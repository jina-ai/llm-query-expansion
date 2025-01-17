# Code is based on https://github.com/jina-ai/late-chunking/blob/main/chunked_pooling/wrappers.py

import torch
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
        task: str | None = None,
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

    def get_instructions(self):
        return [self._model._task_instructions[x] for x in self.tasks]

    def forward(self, *args, **kwargs):
        task_id = self._model._adaptation_map[self.tasks[1]]
        num_examples = kwargs['input_ids'].shape[0]
        adapter_mask = torch.full(
            (num_examples,), task_id, dtype=torch.int32, device=self._model.device
        )
        return self._model.forward(*args, adapter_mask=adapter_mask, **kwargs)

    @property
    def device(self):
        return self._model.device

    @staticmethod
    def has_instructions():
        return True


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


def remove_unsupported_kwargs(original_encode: callable):
    def wrapper(self, *args, **kwargs):
        # Remove 'prompt_name' from kwargs if present
        kwargs.pop('prompt_name', None)
        kwargs.pop('request_qid', None)
        return original_encode(self, *args, **kwargs)

    return wrapper


def load_model(model_name: str, **model_kwargs):
    if model_name in MODEL_WRAPPERS:
        model = MODEL_WRAPPERS[model_name](model_name, **model_kwargs)
        if hasattr(MODEL_WRAPPERS[model_name], 'has_instructions'):
            has_instructions = MODEL_WRAPPERS[model_name].has_instructions()
        else:
            has_instructions = False
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        has_instructions = False

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

    return model, has_instructions
