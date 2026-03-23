"""
Dataset loading utilities for the AI deception detection project.
Loads TruthfulQA and HaluEval for training and evaluation.
"""

from datasets import load_dataset
import pandas as pd


def load_truthfulqa(split="validation"):
    """
    Load TruthfulQA dataset.

    Each example has:
      - question: the question string
      - best_answer: the correct, truthful answer
      - incorrect_answers: list of plausible but false answers
      - category: topic category (e.g. 'Health', 'Law', etc.)
    """
    ds = load_dataset("truthful_qa", "generation", split=split)
    return ds


def load_truthfulqa_mc(split="validation"):
    """
    Load TruthfulQA multiple-choice variant.
    Useful for quick label-based evaluation.

    Each example has:
      - question
      - mc1_targets: {'choices': [...], 'labels': [...]}  (1 correct answer)
      - mc2_targets: {'choices': [...], 'labels': [...]}  (multiple correct answers)
    """
    ds = load_dataset("truthful_qa", "multiple_choice", split=split)
    return ds


def load_halueval(task="qa"):
    """
    Load HaluEval — a dataset of hallucinated vs. faithful responses.
    Task options: 'qa', 'dialogue', 'summarization'

    Each QA example has:
      - knowledge: supporting context
      - question
      - right_answer: faithful answer
      - hallucinated_answer: a plausible but false answer
    """
    ds = load_dataset("pminervini/HaluEval", task)
    return ds


def build_detection_pairs(max_examples=200):
    """
    Build a list of (question, answer, label) tuples for the deception detector.
    label=1 means deceptive/hallucinated, label=0 means truthful.

    Pulls from:
      - TruthfulQA: incorrect answers as deceptive, best_answer as truthful
      - HaluEval QA: hallucinated_answer as deceptive, right_answer as truthful
    """
    pairs = []

    # --- TruthfulQA ---
    tqa = load_truthfulqa()
    for ex in tqa:
        question = ex["question"]
        # truthful example
        pairs.append({
            "question": question,
            "answer": ex["best_answer"],
            "label": 0,
            "source": "truthfulqa_truthful",
        })
        # deceptive example (take first incorrect answer)
        if ex["incorrect_answers"]:
            pairs.append({
                "question": question,
                "answer": ex["incorrect_answers"][0],
                "label": 1,
                "source": "truthfulqa_deceptive",
            })
        if len(pairs) >= max_examples:
            break

    # --- HaluEval ---
    try:
        halu = load_halueval("qa")
        halu_split = halu["data"] if "data" in halu else halu[list(halu.keys())[0]]
        for ex in halu_split:
            pairs.append({
                "question": ex["question"],
                "answer": ex["right_answer"],
                "label": 0,
                "source": "halueval_truthful",
            })
            pairs.append({
                "question": ex["question"],
                "answer": ex["hallucinated_answer"],
                "label": 1,
                "source": "halueval_deceptive",
            })
            if len(pairs) >= max_examples * 2:
                break
    except Exception as e:
        print(f"HaluEval load skipped: {e}")

    df = pd.DataFrame(pairs)
    print(f"Loaded {len(df)} total examples ({df['label'].sum()} deceptive, {(df['label']==0).sum()} truthful)")
    return df


if __name__ == "__main__":
    df = build_detection_pairs(max_examples=100)
    print(df.head())
    print("\nCategory breakdown:")
    print(df["source"].value_counts())
