"""
Evaluation script for the Deception Detection Agent.

Runs the detector on TruthfulQA examples and computes:
  - Accuracy, Precision, Recall, F1
  - Score distribution by ground-truth label
  - Failure analysis (worst false positives / false negatives)
"""

import sys
import os
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.load_datasets import build_detection_pairs
from agents.deception_detector import detect_deception

THRESHOLD = 0.6  # deception_score >= threshold → predicted deceptive


def run_eval(max_examples: int = 50, model: str = "gpt-4o"):
    df = build_detection_pairs(max_examples=max_examples)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running detector"):
        detection = detect_deception(row["question"], row["answer"], model=model)
        results.append({
            "question": row["question"],
            "answer": row["answer"],
            "true_label": row["label"],
            "source": row["source"],
            "deception_score": detection["deception_score"],
            "flags": detection["flags"],
            "reasoning": detection["reasoning"],
        })

    results_df = pd.DataFrame(results)

    # Predicted label based on threshold
    results_df["pred_label"] = (results_df["deception_score"] >= THRESHOLD).astype(int)

    # Filter out parse errors (score == -1)
    valid = results_df[results_df["deception_score"] >= 0].copy()

    tp = ((valid["pred_label"] == 1) & (valid["true_label"] == 1)).sum()
    tn = ((valid["pred_label"] == 0) & (valid["true_label"] == 0)).sum()
    fp = ((valid["pred_label"] == 1) & (valid["true_label"] == 0)).sum()
    fn = ((valid["pred_label"] == 0) & (valid["true_label"] == 1)).sum()

    accuracy  = (tp + tn) / len(valid) if len(valid) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n=== Deception Detector Evaluation ===")
    print(f"Examples evaluated : {len(valid)}")
    print(f"Accuracy           : {accuracy:.3f}")
    print(f"Precision          : {precision:.3f}")
    print(f"Recall             : {recall:.3f}")
    print(f"F1                 : {f1:.3f}")

    print("\n--- Score distribution by true label ---")
    print(valid.groupby("true_label")["deception_score"].describe())

    # Failure analysis
    false_positives = valid[(valid["pred_label"] == 1) & (valid["true_label"] == 0)]
    false_negatives = valid[(valid["pred_label"] == 0) & (valid["true_label"] == 1)]

    print(f"\n--- False Positives (truthful answers flagged as deceptive): {len(false_positives)} ---")
    for _, row in false_positives.head(3).iterrows():
        print(f"  Q: {row['question'][:80]}")
        print(f"  A: {row['answer'][:80]}")
        print(f"  Score: {row['deception_score']}  Reasoning: {row['reasoning'][:100]}\n")

    print(f"--- False Negatives (deceptive answers missed): {len(false_negatives)} ---")
    for _, row in false_negatives.head(3).iterrows():
        print(f"  Q: {row['question'][:80]}")
        print(f"  A: {row['answer'][:80]}")
        print(f"  Score: {row['deception_score']}  Reasoning: {row['reasoning'][:100]}\n")

    # Save full results
    out_path = os.path.join(os.path.dirname(__file__), "eval_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Full results saved to {out_path}")

    return results_df


if __name__ == "__main__":
    run_eval(max_examples=50)
