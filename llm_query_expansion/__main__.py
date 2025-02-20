import argparse
import json

from llm_query_expansion.evaluate import evaluate_model
from llm_query_expansion.expansion import generate_expansions


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="CLI for expanding queries and evaluate task with and without query expansion."
    )

    # Create subparsers for "expand" and "evaluate"
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub-commands: expand or evaluate"
    )

    # Sub-command: expand
    expand_parser = subparsers.add_parser(
        "expand",
        help=(
            "Apply query expansion for a specific task and store the expanded queries to a file "
            "(requires setting GEMINI_API_KEY envrionment variable)."
        ),
    )
    expand_parser.add_argument(
        '--task',
        type=str,
        required=True,
        help="Task name for the task to apply query expansion on (e.g., TRECCOVID).",
    )
    expand_parser.add_argument(
        '--expansion-size',
        type=int,
        required=True,
        default=100,
        help="Target number of words per query for the expansion.",
    )
    expand_parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to store the expanded queries.",
    )
    expand_parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=25,
        help="Batch size for generating query expansions.",
    )
    expand_parser.add_argument(
        "--prompt-template",
        type=str,
        required=False,
        default='general',
        help="The template for the prompt to generate query expansions.",
    )

    # Sub-command: evaluate
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a task with the original query set or an expanded query set.",
    )
    evaluate_parser.add_argument(
        '--task',
        type=str,
        required=True,
        help="Task name for the evaluation task (e.g., TRECCOVID).",
    )
    evaluate_parser.add_argument(
        '--expansions-file',
        type=str,
        required=False,
        help=(
            "Path to the expansions file. If not provided, the original queries "
            "will be used for evaluation."
        ),
    )
    evaluate_parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help="Name of the model to be used for evaluation.",
    )
    evaluate_parser.add_argument(
        '--output-folder',
        type=str,
        required=False,
        default='results',
        help="Folder to store the evaluation results.",
    )

    return parser.parse_args()


def load_expansions_from_file(file_path: str) -> dict[str, str]:
    """
    Load query expansions from a file.

    :param file_path: Path to the file containing the query expansions.
    :return: A dictionary mapping query ids to expanded queries.
    """

    with open(file_path, 'r') as f:
        expansions = json.load(f)
    if isinstance(expansions, dict):
        return expansions
    elif isinstance(expansions, list):
        return {str(query['qid']): query['additional_info'] for query in expansions}


def main() -> None:
    args = get_args()

    if args.command == "expand":
        expansions = generate_expansions(
            args.task,
            args.expansion_size,
            batch_size=args.batch_size,
            prompt_template=args.prompt_template or 'general',
        )
        with open(args.output_file, 'w') as f:
            json.dump(expansions, f, indent=2)
    elif args.command == "evaluate":
        expansions = (
            load_expansions_from_file(args.expansions_file)
            if args.expansions_file
            else None
        )
        if expansions:
            print("Loaded", len(expansions), "query expansions")
        else:
            print("No query expansions loaded")
        evaluate_model(
            args.task, args.model_name, expansions, output_folder=args.output_folder
        )
    else:
        raise ValueError(f"Invalid command: {args.command}")


if __name__ == '__main__':
    main()
