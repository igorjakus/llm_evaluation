import json
from ollama_client import OllamaModel
from rouge_score import rouge_scorer
import bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download("punkt")


def compute_bleu(references: list[str], prediction: str) -> float:
    reference_tokens_list = [nltk.word_tokenize(ref.lower()) for ref in references]
    prediction_tokens = nltk.word_tokenize(prediction.lower())
    smoothie = SmoothingFunction().method1
    return sentence_bleu(
        reference_tokens_list, prediction_tokens, smoothing_function=smoothie
    )


def compute_rouge(reference: str, prediction: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores


def compute_bertscore(references: list[str], predictions: list[str]) -> float:
    P, R, F1 = bert_score.score(predictions, references, lang="en", verbose=False)
    return F1.tolist()[0]


def evaluate_model(model_fn, dataset, metrics: list[str]) -> None:
    predictions = []
    references = []
    results = {m: [] for m in metrics}

    for data in dataset:
        pred = model_fn(data["prompt"])
        predictions.append(pred)

        reference = data["reference"]
        references.append(reference)

        # BLEU and ROUGE are per-sample
        if "bleu" in metrics:
            results["bleu"].append(compute_bleu([reference], pred))

        if "rouge" in metrics:
            results["rouge"].append(compute_rouge(reference, pred))

    # BERTScore is batch-based
    if "bertscore" in metrics:
        results["bertscore"] = compute_bertscore(references, predictions)

    return results


def print_results_summary(results: dict) -> None:
    print("\n=== Evaluation Results Summary ===")

    for metric, scores in results.items():
        if metric == "bleu":
            avg_score = sum(scores) / len(scores)
            print(f"BLEU Score: {avg_score:.4f}")
        elif metric == "rouge":
            rouge1_scores = [s["rouge1"].fmeasure for s in scores]
            rougeL_scores = [s["rougeL"].fmeasure for s in scores]
            print(f"ROUGE-1 F1: {sum(rouge1_scores) / len(rouge1_scores):.4f}")
            print(f"ROUGE-L F1: {sum(rougeL_scores) / len(rougeL_scores):.4f}")
        elif metric == "bertscore":
            print(f"BERTScore F1: {scores:.4f}")


def main():
    with open("../dataset/summarization.json", "r") as file:
        summarize_dataset = json.load(file)

    results = evaluate_model(
        model_fn=OllamaModel.summarize,
        dataset=summarize_dataset,
        metrics=["bleu", "rouge", "bertscore"],
    )

    print_results_summary(results)


if __name__ == "__main__":
    main()
