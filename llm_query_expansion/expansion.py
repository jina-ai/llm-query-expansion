import json
from os import getenv
from random import shuffle

import google.generativeai as genai
import typing_extensions as typing
from mteb import tasks
from tqdm import tqdm

MAX_RETRIES = 5

SCIFACT_PROMPT_TEMPLATE = """
Please provide additional search keywords and phrases for each of the key aspects of the following queries that make it easier to find the scientific document that supports or rejects the scientific fact in the query field (about {size} words per query):
{query}

Please respond in the following JSON schema:
Expansion = {"qid": str, "additional_info": str}
Return: list[Expansion]

"""  # noqa E501

PROMPT_TEMPLATE = """
Please provide additional search keywords and phrases for each of the key aspects of the following queries that make it easier to find relevant documents (about {size} words per query):
{query}

Please respond in the following JSON schema:
Expansion = {"qid": str, "additional_info": str}
Return: list[Expansion]

"""  # noqa E501

PROMPT_TEMPLATES = {'general': PROMPT_TEMPLATE, 'SciFact': SCIFACT_PROMPT_TEMPLATE}


class QExpansion(typing.TypedDict):
    qid: str
    additional_info: str


def get_prompt(queries: str, size: int, template: str) -> str:
    """
    Generate a prompt to provide additional search keywords and phrases to expand the queries.

    :param queries: A dictionary mapping query ids to queries to expand in JSON format.
    :param size: The number of words per query for the expansion.
    :param template: The template for the prompt to generate query expansions.
    :return: A prompt to provide additional search keywords and phrases for the queries.
    """

    prompt = PROMPT_TEMPLATE.replace("{size}", str(size)).replace(
        "{query}", json.dumps(queries)
    )
    return prompt


def generate_expansions(
    task_name: str, size: int, batch_size: int = 50, prompt_template: str = 'general'
) -> list[dict[str, str]]:
    """
    Generate query expansions using the GEMINI API.

    :param queries: A dictionary mapping query ids to queries to expand.
    :param size: The number of words per query for the expansion.
    :param batch_size: The batch size for generating query expansions.
    :param prompt_template: The template for the prompt to generate query expansions.
    :return: A dictionary mapping query ids to query expansions.
    """

    task = getattr(tasks, task_name, None)()
    task.load_data()
    queries = list(task.queries['test'].items())

    genai.configure(
        api_key=getenv("GEMINI_API_KEY"),
    )
    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    expansions = {}
    for batch in tqdm(
        range(0, len(queries), batch_size),
        desc="Generating expansions",
        total=len(queries) // batch_size,
    ):
        batch_queries = [
            {'qid': x[0], 'query': x[1]} for x in queries[batch : batch + batch_size]
        ]
        for i in range(MAX_RETRIES):
            prompt = get_prompt(
                json.dumps(batch_queries), size, PROMPT_TEMPLATES[prompt_template]
            )
            try:
                model_response = model.generate_content(prompt).text
                model_response = model_response[
                    model_response.find("[") : -model_response[::-1].find("]")
                ]

                expanded_queries = json.loads(model_response)
                for query in expanded_queries:
                    expansions[query['qid']] = query['additional_info']
            except Exception as e:
                print(f"Error generating expansions: {e}")
                shuffle(batch_queries)
                continue
            break
    return [{'qid': k, 'additional_info': v} for (k, v) in expansions.items()]
