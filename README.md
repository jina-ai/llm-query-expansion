# LLM-Based Query Expansions for Text Embedding Models

TODO write some introduction about Query Expansion

TODO write about that using LLMs for Query Expansion is trivial
Query Expension for Better Query Embedding using LLMs


## Evaluation Results

**SciFact**
| Model     | No Expansion | Gemini-F1.5 (100 words) | Gemini-F1.5 (150 words) | Gemini-F1.5 (250 words) |
|-----------|--------------|-------------------------|-------------------------|-------------------------|
| Jina V3   | 72.74        | 73.39                   | 74.16                   | **74.33**               |
| MiniLM    | 64.508       | **68.72**               | 66.27                   | 68.50                   |


## Run Query Expansion Experiments

### Generate Expansions

We provided a set of expanded queries in the the `query_expansions` folder of this repository that can be used out of the box.

If you want to create custom expansions or expand queries for other tasks, you can use the `expand` command. 
This requires getting a [Gemini API key](https://aistudio.google.com/apikey) (at the moment you can get a key for free for up to 1,500 requests per day) and setting it as environment variable:

```sh
export GEMINI_API_KEY={YOUR_KEY}
```

To apply query expansion to a specific MTEB Retrieval task, you can then call:

```sh
python3 -m llm_query_expansion expand --task {TASK_NAME} --expansion-size {NUMBER_OF_WORDS} --batch-size {QUERIES_PER_LLM_REQUEST} --output-file query_expansions/{FILENAME}
```

### Evalute Models With and Without Expanded Queries

To run evaluations with expanded queries, you can use the file  

```sh
python3 -m llm_query_expansion evaluate  --task {TASK_NAME} --expansions-file query_expansions/{FILE_NAME} --model-name {MODEL_NAME} --output-folder {RESULTS_FOLDER_NAME}
```

