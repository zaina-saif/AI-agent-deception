"""
Compare detector-only performance against the full pipeline (detector + verifier).

The script uses the same dataset examples as evaluate_detector.py and records:
  - example index
  - question
  - ground truth label
  - detector-only prediction and score
  - full-pipeline prediction and score
  - whether the prediction changed

It also computes metrics for both systems and saves results to CSV.
"""

import os
import sys
import argparse
import pandas as pd
from typing import Dict, Tuple, List
from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.load_datasets import build_detection_pairs
from agents.deception_detector import detect_deception
from pipeline import DeceptionDetectionPipeline

DETECTOR_THRESHOLD = 0.6
DEFAULT_OUTPUT_CSV = os.path.join(_ROOT, "eval", "full_pipeline_comparison.csv")


def compute_metrics(true_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)

    accuracy = (tp + tn) / len(true_labels) if true_labels else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"accuracy={metrics['accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}, "
        f"f1={metrics['f1']:.3f}, "
        f"confusion=(tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']}, tn={metrics['tn']})"
    )


def run_comparison(
    api_key: str,
    max_examples: int = 50,
    strategy: str = "zero_shot",
    output_csv: str = DEFAULT_OUTPUT_CSV,
    detector_threshold: float = DETECTOR_THRESHOLD,
    verification_method: str = "llm",  # New parameter for verification method
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    df = build_detection_pairs(max_examples=max_examples)

    # Use LLM verification by default, with consistency checking enabled
    pipeline = DeceptionDetectionPipeline(
        api_key, 
        strategy=strategy, 
        use_consistency_check=True,  # Always enabled
        consistency_n=3,
        verification_method=verification_method  # Pass verification method
    )

    rows = []
    detector_preds = []
    pipeline_preds = []
    true_labels = []

    for idx, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]
        true_label = int(row["label"])

        # Detector-only on dataset answer
        detection = detect_deception(question, answer, strategy=strategy)
        detector_score = float(detection.get("deception_score", -1.0))
        detector_pred = 1 if detector_score >= detector_threshold else 0

        # Full pipeline on generated answer
        result = pipeline.process_question(question)
        pipeline_pred = 1 if result.final_decision == "deceptive" else 0
        pipeline_score = float(result.confidence)

        prediction_changed = detector_pred != pipeline_pred

        rows.append({
            "index": int(idx),
            "question": question,
            "true_label": true_label,
            "source": row.get("source", ""),
            "detector_score": detector_score,
            "detector_prediction": detector_pred,
            "pipeline_score": pipeline_score,
            "pipeline_prediction": pipeline_pred,
            "prediction_changed": prediction_changed,
            "pipeline_decision": result.final_decision,
            "pipeline_explanation": result.explanation,
            "detector_flags": "; ".join(detection.get("flags", [])) if detection.get("flags") else "",
            "detector_reasoning": detection.get("reasoning", ""),
        })

        detector_preds.append(detector_pred)
        pipeline_preds.append(pipeline_pred)
        true_labels.append(true_label)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_csv, index=False)

    detector_metrics = compute_metrics(true_labels, detector_preds)
    pipeline_metrics = compute_metrics(true_labels, pipeline_preds)

    return results_df, detector_metrics, pipeline_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare detector-only vs full pipeline evaluation")
    parser.add_argument("--max_examples", type=int, default=50, help="Maximum examples to evaluate")
    parser.add_argument("--strategy", type=str, default="zero_shot", choices=["zero_shot", "few_shot", "cot"], help="Detector prompt strategy")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV, help="CSV file to save comparison results")
    parser.add_argument("--threshold", type=float, default=DETECTOR_THRESHOLD, help="Detector deception threshold")
    parser.add_argument("--verification_method", type=str, default="llm", choices=["llm", "faiss", "hybrid"], help="Verification method (llm=reasoning-based, faiss=retrieval-based, hybrid=try both)")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is required")
        return 1

    results_df, detector_metrics, pipeline_metrics = run_comparison(
        api_key=api_key,
        max_examples=args.max_examples,
        strategy=args.strategy,
        output_csv=args.output_csv,
        detector_threshold=args.threshold,
        verification_method=args.verification_method,
    )

    print("\n=== Comparison Summary ===")
    print(f"Examples evaluated: {len(results_df)}")
    print("\nDetector-only metrics:")
    print(format_metrics(detector_metrics))
    print("\nFull pipeline metrics:")
    print(format_metrics(pipeline_metrics))

    change_rate = results_df["prediction_changed"].mean() if len(results_df) else 0.0
    print(f"\nPrediction changed in {results_df['prediction_changed'].sum()} / {len(results_df)} examples ({change_rate:.2%})")
    print(f"Comparison results saved to {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
