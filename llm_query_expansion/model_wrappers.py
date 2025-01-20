# Code is based on https://github.com/jina-ai/late-chunking/blob/main/chunked_pooling/wrappers.py

from mteb.encoder_interface import PromptType
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import AutoModel


def construct_document(doc: dict | str):
    if isinstance(doc, str):
        return doc
    elif 'title' in doc:
        return f'{doc["title"]} {doc["text"].strip()}'
    else:
        return doc['text'].strip()


class JinaEmbeddingsV3Wrapper(nn.Module):
    def __init__(
        self, model_name, tasks=['retrieval.query', 'retrieval.passage'], **model_kwargs
    ):
        super().__init__()
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        )
        self.tasks = tasks

    def encode_queries(
        self,
        sentences: str | list[str],
        *args,
        **kwargs,
    ):
        return self._model.encode(sentences, *args, task=self.tasks[0], **kwargs)

    def encode_corpus(
        self,
        sentences: str | list[str],
        *args,
        **kwargs,
    ):
        _sentences = [construct_document(sentence) for sentence in sentences]
        return self._model.encode(_sentences, *args, task=self.tasks[1], **kwargs)

    def encode(
        self,
        *args,
        prompt_type: PromptType | None = None,
        **kwargs,
    ):
        if prompt_type and prompt_type == PromptType.passage:
            return self.encode_corpus(*args, **kwargs)
        return self.encode_queries(*args, **kwargs)

    @property
    def device(self):
        return self._model.device


MODEL_WRAPPERS: dict[str, any] = {
    'jinaai/jina-embeddings-v3': JinaEmbeddingsV3Wrapper,
    'sentence-transformers/all-MiniLM-L6-v2': SentenceTransformer,
}

MODEL_EVAL_BATCH_SIZES: dict[str, int] = {
    'jinaai/jina-embeddings-v3': 8,
    'sentence-transformers/all-MiniLM-L6-v2': 32,
}

MODELS_WITHOUT_PROMPT_NAME_ARG: list[str] = [
    'jinaai/jina-embeddings-v2-small-en',
    'jinaai/jina-embeddings-v2-base-en',
    'jinaai/jina-embeddings-v3',
]


def remove_unsupported_kwargs(original_encode: callable) -> callable:
    """
    Remove unsupported kwargs from the encode function of the model.

    :param original_encode: The original encode function of the model.
    :return: A wrapper function that removes unsupported kwargs.
    """

    def wrapper(self, *args, **kwargs):
        # Remove 'prompt_name' from kwargs if present
        kwargs.pop('task_name', None)
        kwargs.pop('prompt_name', None)
        kwargs.pop('request_qid', None)
        return original_encode(self, *args, **kwargs)

    return wrapper


def load_model(model_name: str, **model_kwargs) -> nn.Module:
    """
    Load a model from the given model name.

    :param model_name: The name of the model to load.
    :return: The loaded model.
    """
    if model_name in MODEL_WRAPPERS:
        model = MODEL_WRAPPERS[model_name](model_name, **model_kwargs)
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # encode functions of various models do not support all sentence transformers kwargs parameter
    if model_name in MODELS_WITHOUT_PROMPT_NAME_ARG:
        ENCODE_FUNC_NAMES = ['encode', 'encode_queries', 'encode_corpus']
        for func_name in ENCODE_FUNC_NAMES:
            if hasattr(model, func_name):
                setattr(
                    model,
                    func_name,
                    remove_unsupported_kwargs(getattr(model, func_name)),
                )

    return model
