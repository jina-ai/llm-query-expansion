from typing import TYPE_CHECKING

from mteb import MTEB, tasks

from llm_query_expansion.model_wrappers import MODEL_EVAL_BATCH_SIZES, load_model

if TYPE_CHECKING:
    from mteb.abstasks import AbsTask


def expand_task(task: AbsTask, expansions: dict[str, str]) -> None:
    """
    Expand the queries in a task using the provided expansions.

    :param task: The task to expand.
    :param expansions: A dictionary mapping query ids to expansions for the queries
    """
    new_queries = {
        qid: f'{query} {expansions[qid]}' for qid, query in task.queries['test'].items()
    }
    task.queries['test'] = new_queries


def add_instruction(task: any, model: any) -> None:
    # TODO implement this function
    ...


def evaluate_model(task_name: str, model_name: str, expansions: dict[str, str]) -> None:
    """
    Evaluate a model on a given task.

    :param task_name: The name of the task to evaluate the model on.
    :param model_name: The name of the model to evaluate.
    :param expansions: A dictionary mapping query ids to expansions for the queries.
    """

    task = getattr(tasks, task_name, None)()
    if task is None:
        raise ValueError(
            f"Invalid task name: {task_name} Selelct a retreival task from "
            + "https://github.com/embeddings-benchmark/mteb/blob/main/docs/tasks.md"
        )

    model, has_instructions = load_model(model_name)
    batch_size = MODEL_EVAL_BATCH_SIZES[model_name]

    expand_task(task, expansions)
    if has_instructions:
        add_instruction(task, model)

    benchmark = MTEB(tasks=[task])
    benchmark.run(model, batch_size=batch_size)
