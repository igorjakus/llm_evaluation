import json
from model import LLMModel
from evaluator import Evaluator


def main():
    evaluator = Evaluator()

    tasks = [
        {
            "name": "Summarization",
            "dataset": "../dataset/summarization.json",
            "model_fn": LLMModel.summarize,
            "metrics": ["bleu", "rouge", "bertscore"],
        },
        {
            "name": "Math",
            "dataset": "../dataset/math.json",
            "model_fn": LLMModel.solve_math,
            "metrics": ["math_accuracy", "llm_judge"],
        },
    ]

    for task in tasks:
        try:
            with open(task["dataset"]) as f:
                dataset = json.load(f)

            results = evaluator.evaluate(task["model_fn"], dataset, task["metrics"])
            evaluator.print_summary(results, task["name"])
        except FileNotFoundError:
            print(f"\n{task['name']} dataset not found. Skipping.")


if __name__ == "__main__":
    main()
