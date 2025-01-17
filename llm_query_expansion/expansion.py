import json
from os import getenv

import google.generativeai as genai

PROMPT_TEMPLATE = """
Please provide additional search keywords and phrases for each of the key aspects of the  following queries make it easier to find the document that supports or rejects the fact in the query field (about {size} words per query):
{query}
Please respond in the follwoing JSON format:
[
    {qid: <query-id>, "additional_info": <query-info>},
    [...]
]

Output: 
"""  # noqa E501

GEMINI_BATCH_SIZE = 100


def get_prompt(queries: str, size: int):
    """
    Generate a prompt to provide additional search keywords and phrases to expand the queries.

    :param queries: A dictionary mapping query ids to queries to expand in JSON format.
    :param size: The number of words per query for the expansion.
    :return: A prompt to provide additional search keywords and phrases for the queries.
    """

    prompt = PROMPT_TEMPLATE.replace("{size}", str(size)).replace(
        "{query}", json.dumps(queries)
    )
    return prompt


def generate_expansions(queries: dict, size: int):
    """
    Generate query expansions using the GEMINI API.

    :param queries: A dictionary mapping query ids to queries to expand.
    :param size: The number of words per query for the expansion.
    """
    genai.configure(api_key=getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    expansions = {}
    for batch in range(0, len(queries), GEMINI_BATCH_SIZE):
        batch_queries = queries[batch : batch + GEMINI_BATCH_SIZE]
        prompt = get_prompt(batch_queries, size)
        expanded_queries = model(prompt)
        for query in expanded_queries:
            expansions[query['qid']] = query['additional_info']
    return expansions
