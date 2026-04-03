"""
Integration test for the complete multi-agent pipeline

Tests the integration between:
- Generator (GeneratorAgent)
- Ananya's Deception Detector
- Zaina's Verifier + Retrieval

Uses the same evaluation data that Ananya used for her detector.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline import DeceptionDetectionPipeline
from data.load_datasets import build_detection_pairs

def test_pipeline_integration():
    """Test the integrated pipeline on a small subset of evaluation data"""

    print("Testing Multi-Agent Pipeline Integration")
    print("=" * 50)

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Initialize pipeline
    pipeline = DeceptionDetectionPipeline(api_key)

    # Load evaluation data (same as Ananya used)
    eval_data = build_detection_pairs(max_examples=20)  # Small subset for testing

    print(f"Testing on {len(eval_data)} examples from evaluation dataset")
    print()

    results = []
    correct_predictions = 0
    total_predictions = 0

    for i, (_, row) in enumerate(eval_data.iterrows()):
        question = row["question"]
        answer = row["answer"]
        true_label = row["label"]  # 0 = truthful, 1 = deceptive

        print(f"Example {i+1}: {question[:60]}...")
        print(f"True label: {'deceptive' if true_label else 'truthful'}")

        try:
            # Run through pipeline
            result = pipeline.process_question(question)

            # For testing, we'll use the generated answer instead of the dataset answer
            # This tests the full pipeline: Generator → Detector → Verifier
            generated_answer = result.generated_answer
            detection_result = pipeline.deception_detector.analyze_response(question, generated_answer)

            # Use detection result to determine if verification needed
            if detection_result["needs_verification"] and detection_result["claims_to_verify"]:
                verification_results = pipeline.verifier.verify_claims_batch(detection_result["claims_to_verify"])
            else:
                verification_results = []

            # Make final decision
            final_decision, confidence, explanation = pipeline._make_final_decision(
                detection_result["deception_score"], verification_results
            )

            # Compare with ground truth (simplified: deceptive if score > 0.5)
            pred_label = 1 if final_decision == "deceptive" else 0
            is_correct = (pred_label == true_label)

            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            print(f"Generated answer: {generated_answer[:100]}...")
            print(f"Deception score: {detection_result['deception_score']:.2f}")
            print(f"Claims to verify: {len(detection_result['claims_to_verify'])}")
            print(f"Final decision: {final_decision} (confidence: {confidence:.2f})")
            print(f"Correct: {is_correct}")
            print("-" * 50)

            results.append({
                "question": question,
                "generated_answer": generated_answer,
                "true_label": true_label,
                "deception_score": detection_result["deception_score"],
                "claims_count": len(detection_result["claims_to_verify"]),
                "final_decision": final_decision,
                "confidence": confidence,
                "correct": is_correct
            })

        except Exception as e:
            print(f"Error processing example: {e}")
            continue

    # Summary
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nIntegration Test Results:")
    print(f"Accuracy: {accuracy:.2f} ({correct_predictions}/{total_predictions})")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("integration_test_results.csv", index=False)
    print("Detailed results saved to integration_test_results.csv")

def compare_single_vs_multi_agent():
    """Compare single-agent (deception detector only) vs multi-agent performance"""

    print("\nComparing Single-Agent vs Multi-Agent Performance")
    print("=" * 50)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return

    pipeline = DeceptionDetectionPipeline(api_key)
    eval_data = build_detection_pairs(max_examples=20)

    single_agent_correct = 0
    multi_agent_correct = 0
    total = 0

    for _, row in eval_data.iterrows():
        question = row["question"]
        true_label = row["label"]

        try:
            result = pipeline.process_question(question)
            generated_answer = result.generated_answer

            # Single agent: just deception detector
            detection = pipeline.deception_detector.analyze_response(question, generated_answer)
            single_pred = 1 if detection["deception_score"] > 0.5 else 0

            # Multi agent: full pipeline
            multi_pred = 1 if result.final_decision == "deceptive" else 0

            if single_pred == true_label:
                single_agent_correct += 1
            if multi_pred == true_label:
                multi_agent_correct += 1

            total += 1

        except Exception as e:
            continue

    single_acc = single_agent_correct / total if total > 0 else 0
    multi_acc = multi_agent_correct / total if total > 0 else 0

    print(f"Single Agent Accuracy: {single_acc:.2f}")
    print(f"Multi Agent Accuracy: {multi_acc:.2f}")
    print(f"Improvement: {multi_acc - single_acc:.2f}")

if __name__ == "__main__":
    test_pipeline_integration()
    compare_single_vs_multi_agent()