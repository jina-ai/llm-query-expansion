# LLM-Based Query Expansions for Text Embedding Models

This repository contains code for experimenting with LLM-based query expansion.

<p>
<img src="misc/llm-query-expansion.png" alt="Flow Chart - LLM-Based Query Expansion" width="1000" />
</p>
<p>
<em>Figure 1: LLM-Based Query Expansion</em>
</p>

## What is LLL-Based Query Expansion?

*Query Expansion* is a technique that automatically expands or reformulates queries to improve information retrieval (IR) systems, i.e., it makes it easier for your search system to find relevant documents.

Query expansion is well-established in IR.
It is specifically effective in solving problems of keyword-based search systems like BM25 that typically suffer from "zero results" problems: If your search terms don't occur in any document, you don't get any results.
It expands the query with additional relevant terms so that it is less likely that no document contains any of them.
This usually leads to more results (better recall - less "zero result" situations) and often more accurate relevance judgments (higher precision):
Researchers and practitioners have developed a large variety of different methods to find relevant terms for expanding the queries.
Some of the most popular one are:
- **Thesaurus-Based Methods:** A dictionary of synonyms in the domain of the retrieval tasks, called [Thesaurus](https://en.wikipedia.org/wiki/Thesaurus) is used to identify synonymous or related terms that are added to the query string. In the same way knowledge graphs can be used to identify related text values to the terms in the query to expand the query, the class of a term mentioned in the query.
- **Relevance Feedback:** If a use provides feedback on which documents are relevant, the IR system can expand the query with terms extracted from those documents to find related documents. If no user feedback is available, term from the highest-ranked documents can be used instead (pseudo-relevance feedback).
- **Query Logs:** Terms occurring in similar queries in the past can get added to extend the query.
- **User Personalization:** Additional user information, e.g., the users location and preferences, can be added in the form of query terms to identify more relevant documents for this specific user.

As embedding models capture the semantics of words adding synonyms is not so effective.
Moreover, you never get zero results because embeddings allow calculating similarity scores for every document. 
Consequently, the traditional query expansion methods are less popular for embedding models.

LLMs changed this situation a bit.
While the traditional methods are not used that much, LLM-based query expansion can be indeed effective.
Researchers have explored various different LLM-based methods, e.g., using in-context learning (https://arxiv.org/pdf/2305.03653) or progressively increasing the query expansion based on the first retrieval results (https://arxiv.org/pdf/2406.07136)
In contrast, we aim at exploring how simple expansion methods works with current LLMs and embedding models that can easily be applied to standard use cases and basically any retrieval dataset without much fine-grained tuning of hyperparameters etc.

Figure 1 shows how this works:
1. A query (or a set of queries) is inserted into a prompt template to instruct the LLMs generating query expansions (relevant additional info in the form of keywords / key phrases)
2. The LLMs is applied to generate the expansions based on the prompt.
3. Each query gets concatenated with its expansions and the embedding model encodes this into a representation
4. After that the embedding can be used to match document embeddings, as it is usually done in dense retrieval systems.


Compared to traditional query expansion methods this technique:
- Abandons users from the need of finding a suitable thesaurus for your retrieval task that has the domain-specific terminology
- It is much more flexible, allowing you to adjust the prompt to the needs of the retrieval task (whether you want to find documents, duplicate texts, contradicting statements, etc.) as well as freely decide on the length of the expansions


## Apply LLM-Based Query Expansion to Different Retrieval Tasks

To test LLMs for query expansion, we first need to construct a prompt template.
We decided to expand multiple queries at once to reduce the number of LLM calls, although in production, you might prefer to expand every incoming query individually when it is issued to the system.
We use the following prompt, which is generic enough to apply as-is to many different datasets:

<p>
<img src="misc/prompt.png" alt="Prompt" width="500" />
</p>
<p>
<em>Figure 2: Prompt Template to Generate Query Expansions</em>
</p>

It should return a JSON response that maps the query IDs to their expansions.
One can configure the target length of the expansions and test this with 100, 150, and 250 words.

We use the new [Gemini Flash 2.0 (experimental)](https://deepmind.google/technologies/gemini/flash/) language model and two embedding models: [jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3/) and [all-MiniLM-L6-v2](https://huggingface.co/).

We group 20-50 queries into each request (more queries could exceed the output limit of the LLM).
If JSON responses are invalid, we shuffle the queries, reconstruct the prompt, and run the requests again until the LLM returns a valid JSON response that maps the query IDs to their expansions. 

For evaluating how well the method works, we apply our approach to some of the BeIR datasets that have a low number of queries.

**SciFact**
| Model     | No Expansion | Gemini-F1.5 (100 words) | Gemini-F1.5 (150 words) | Gemini-F1.5 (250 words) |
|-----------|--------------|-------------------------|-------------------------|-------------------------|
| Jina V3   | 72.74        | 73.39                   | 74.16                   | **74.33**               |
| MiniLM    | 64.51        | **68.72**               | 66.27                   | 68.50                   |

**TRECCOVID**
| Model     | No Expansion | Gemini-F1.5 (100 words) | Gemini-F1.5 (150 words) | Gemini-F1.5 (250 words) |
|-----------|--------------|-------------------------|-------------------------|-------------------------|
| Jina V3   | 77.55        | 76.74                   | 77.12                   | **79.28**               |
| MiniLM    | 47.25        | 67.90                   | **70.18**               | 69.60                   |

**FiQA**
| Model     | No Expansion | Gemini-F1.5 (100 words) | Gemini-F1.5 (150 words) | Gemini-F1.5 (250 words) |
|-----------|--------------|-------------------------|-------------------------|-------------------------|
| Jina V3   | 47.34        | **47.76**               | 46.03                   | 47.34                   |
| MiniLM    | **36.87**    | 33.96                   | 32.60                   | 31.84                   |

**NFCorpus**
| Model     | No Expansion | Gemini-F1.5 (100 words) | Gemini-F1.5 (150 words) | Gemini-F1.5 (250 words) |
|-----------|--------------|-------------------------|-------------------------|-------------------------|
| Jina V3   | 36.46        | **40.62**               | 39.63                   | 39.20                   |
| MiniLM    | 31.59        | **33.76**               | 33.76                   | 33.35                   |

**Touche2020**
| Model     | No Expansion | Gemini-F1.5 (100 words) | Gemini-F1.5 (150 words) | Gemini-F1.5 (250 words) |
|-----------|--------------|-------------------------|-------------------------|-------------------------|
| Jina V3   | 26.24        | 26.91                   | 27.15                   | **27.54**               |
| MiniLM    | 16.90        | **25.31**               | 23.52                   | 23.23                   |

**Average Improvement**
| Model     | Gemini-F1.5 (100 words) | Gemini-F1.5 (150 words) | Gemini-F1.5 (250 words) |
|-----------|-------------------------|-------------------------|-------------------------|
| Jina V3   | +1.02                   | +0.75                   | **+1.48**               |
| MiniLM    | **+6.51**               | +5.84                   | +5.88                   |


The results show that in general, LLM-based query expansion improves the retrieval results.
However, there are also situations, in which it can decrease performance, e.g., for MiniLM when applied for the FiQA dataset.
We also observe generally larger improvements with the small MiniLM through query expansion than when using the larger jina-embeddings-v3 model.
One reason might be that small models have more difficulties to represent technical terminology and less frequent words with complex meanings accurately.
Another reason might be that it is easier to improve a low-performing model than a model with already good retrieval performance.
Nevertheless, despite higher magnitudes of improvement for MiniLM, the improvements for jina-embeddings-v3 seem more robust, e.g., for FiQA query expansion can lead to small improvements for jina-embeddings-v3 in contrast to a 3% decrease for MiniLM.
Another observation is that jina-embeddings-v3 tends to benefit more from longer expansions whereas MiniLM tends to perform better with shorter expansions. This might be due to the limited context length of MiniLM and its generally poorer performance on long texts.

### Task-Specific Prompt Expansion

While the method with the prompt above usually leads to improvements, it also degrades performance for some datasets.
This might be due to the generic prompt that leads to expansions that not perfectly align with the intended retrieval task and embedding models.
So we want to test whether using more task-specific prompts leads to better result.
To achieve this, we constructed refined task-specific prompts, e.g., for SciFact we made the adjustments presented in Figure 3:

<p>
<img src="misc/prompt-diff.png" alt="Prompt Difference" width="500" />
</p>
<p>
<em>Figure 3: Task-Specific Adjustments in the Prompt Template</em>
</p>

**SciFact - Task-Specific Prompt for Expansion**
| Model     | No Expansion | Gemini-F1.5 (100 words) | Gemini-F1.5 (150 words) | Gemini-F1.5 (250 words) |
|-----------|--------------|-------------------------|-------------------------|-------------------------|
| Jina V3   | 72.74        | **75.85 (+2.46)**       | 75.07 (+0.91)           | 75.13 (+0.80)           |
| MiniLM    | 64.51        | **69.12 (+0.40)*        | 68.10 (+1.83)           | 67.83 (-0.67)           |

**FiQA - Task-Specific Prompt for Expansion**
| Model     | No Expansion | Gemini-F1.5 (100 words) | Gemini-F1.5 (150 words) | Gemini-F1.5 (250 words) |
|-----------|--------------|-------------------------|-------------------------|-------------------------|
| Jina V3   | 47.34        | 47.77 (+0.01)           | **48.20 (+1.99)**       | 47.75 (+0.41)           |
| MiniLM    | **36.87**    | 34.71 (+0.75)           | 34.68 (+2.08)           | 34.50 (+2.66)           |

One can see that the average improvement for the task-specific prompts is higher, although MiniLM still does not to profit from the expansion of the FIQA queries.

## What are the Advantages and Disadvantages of LLM-Based Query Expansion?

Our experimental results show that query expansion with LLMs can indeed improve the retrieval performance of embedding models.
Specifically, they are helpful:
- If your model is able to comprehend long queries well
- If you are not happy with the retrieval performance
- If retrieval performance is more important than speed

In contrast, traditional methods deliver more speed, are potentially more cost efficient, but usually require information from additional resources like a domain-specific thesaurus or relevance feedback from users.

## Further Directions

- Using retrieval models other than embedding models, such as re-ranking models
- Extend the documents instead of the queries, or prepend a summary of the document before embedding (many embedding models have limited context length or focus on the beginning of the document)
- Instead of using LLMs, testing techniques that are faster and cheaper.
- Experiment with the prompting, e.g., clearly define the format for the expansions: whether to include topics, questions, synonyms and how many of them, and test different prompt variations.

## Conclusion

We looked at LLM-based query expansion which is an alternative to traditional methods.
Compared to traditional methods, it is easy to implement as it does not rely on external information, such as thesauri. 
We tested the LLM method on various tasks and observed that it generally improves the retrieval performance of embedding models; however, the performance varies between datasets.
While a generic prompt seems to work, customizing the prompt to generate expansions specifically for the target dataset can further improve performance.

**Links to additional resources:**
- Jina Embeddings V3: https://huggingface.co/jinaai/jina-embeddings-v3
- Jina Embedding API: https://jina.ai/embeddings/
- Gemini API: https://aistudio.google.com/prompts/new_chat?model=gemini-2.0-flash-experimental
- BeIR Benchmark: https://github.com/beir-cellar/beir

## How to Run Query Expansion Experiments and Reproduce the Results?

The `llm_query_expansion` tool in this repository serves as a CLI tool that has two subcommands:
- `expand`: Creates query expansions for different retrieval tasks of the [MTEB benchmark](https://github.com/embeddings-benchmark/mteb) and stores them in a file.
- `evaluate`: Evaluates the tasks with different query expansions (or without any query expansion)

You can be install the tool with `pip install .`

### Generate Expansions

We have provided a set of expanded queries in the `query_expansions` folder of this repository which can be used out of the box.

If you want to create custom expansions or expand queries for other tasks, you can use the `expand` command. 
This requires obtaining a [Gemini API key](https://aistudio.google.com/apikey) (currently, you can get a key for free for up to 1,500 requests per day) and setting it as an environment variable:

```sh
export GEMINI_API_KEY={YOUR_KEY}
```

To apply query expansion to a specific MTEB retrieval task, you can then call:

```sh
python3 -m llm_query_expansion expand --task {TASK_NAME} --expansion-size {NUMBER_OF_WORDS} --batch-size {QUERIES_PER_LLM_REQUEST} --output-file query_expansions/{FILENAME}
```

### Evalute Models With and Without Expanded Queries

To run evaluations with expanded queries, you can use the file:

```sh
python3 -m llm_query_expansion evaluate  --task {TASK_NAME} --expansions-file query_expansions/{FILE_NAME} --model-name {MODEL_NAME} --output-folder {RESULTS_FOLDER_NAME}
```

