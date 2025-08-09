import re
from typing import Dict, List, Callable
from rouge_score import rouge_scorer
import bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download("punkt")


class Evaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"], use_stemmer=True
        )

    def bleu(self, references: List[str], prediction: str) -> float:
        ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]
        pred_tokens = nltk.word_tokenize(prediction.lower())
        return sentence_bleu(
            ref_tokens, pred_tokens, smoothing_function=SmoothingFunction().method1
        )

    def rouge(self, reference: str, prediction: str) -> Dict:
        return self.rouge_scorer.score(reference, prediction)

    def bertscore(self, references: List[str], predictions: List[str]) -> float:
        _, _, F1 = bert_score.score(predictions, references, lang="en", verbose=False)
        return F1.mean().item()

    def math_accuracy(self, reference: str, prediction: str) -> bool:
        def extract_number(text: str) -> str:
            numbers = re.findall(r"-?\d+\.?\d*", text.strip())
            return numbers[-1] if numbers else ""

        ref_num = extract_number(reference)
        pred_num = extract_number(prediction)
        return ref_num == pred_num and ref_num != ""

    def evaluate(
        self, model_fn: Callable, dataset: List[Dict], metrics: List[str]
    ) -> Dict:
        predictions, references = [], []
        results = {m: [] for m in metrics}

        for data in dataset:
            pred = model_fn(data["prompt"])
            ref = data["reference"]
            predictions.append(pred)
            references.append(ref)

            if "bleu" in metrics:
                results["bleu"].append(self.bleu([ref], pred))
            if "rouge" in metrics:
                results["rouge"].append(self.rouge(ref, pred))
            if "math_accuracy" in metrics:
                results["math_accuracy"].append(self.math_accuracy(ref, pred))

        if "bertscore" in metrics:
            results["bertscore"] = self.bertscore(references, predictions)

        return results

    def print_summary(self, results: Dict, task_name: str) -> None:
        print(f"\n=== {task_name} Results ===")

        if "bleu" in results:
            print(f"BLEU: {sum(results['bleu']) / len(results['bleu']):.4f}")

        if "rouge" in results:
            rouge1 = [s["rouge1"].fmeasure for s in results["rouge"]]
            rougeL = [s["rougeL"].fmeasure for s in results["rouge"]]
            print(f"ROUGE-1: {sum(rouge1) / len(rouge1):.4f}")
            print(f"ROUGE-L: {sum(rougeL) / len(rougeL):.4f}")

        if "bertscore" in results:
            print(f"BERTScore: {results['bertscore']:.4f}")

        if "math_accuracy" in results:
            print(
                f"Accuracy: {sum(results['math_accuracy']) / len(results['math_accuracy']):.4f}"
            )
