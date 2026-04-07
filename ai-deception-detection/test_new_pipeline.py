#!/usr/bin/env python3
"""
Test script for improved multi-agent pipeline with:
1. Consistency-based verification (primary signal)
2. LLM-based verification (reasoning-based, not retrieval)
3. Better decision fusion logic
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.load_datasets import build_detection_pairs
from agents.deception_detector import detect_deception
from pipeline import DeceptionDetectionPipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

api_key = os.getenv('OPENAI_API_KEY')

print("=" * 70)
print("IMPROVED MULTI-AGENT PIPELINE TEST")
print("Features: Consistency checking + LLM-based verification")
print("=" * 70)

# Load examples
df = build_detection_pairs(max_examples=10)
print(f"\nLoaded {len(df)} examples ({sum(df['label'])} deceptive, {len(df)-sum(df['label'])} truthful)")

# Initialize pipeline with new settings
print("\nInitializing pipeline...")
print("  - Consistency checking: ENABLED (primary signal)")
print("  - Verification method: llm (reasoning-based)")
print("  - Decision logic: Consistency-first fusion")

pipeline = DeceptionDetectionPipeline(
    api_key, 
    strategy='zero_shot', 
    use_consistency_check=True,  # Now primary signal
    consistency_n=3,
    verification_method='llm'  # LLM-based verification
)

# Run evaluation
detector_preds = []
pipeline_preds = []
true_labels = []
results = []

for idx, row in df.iterrows():
    question = row['question']
    answer = row['answer']
    true_label = int(row['label'])
    
    print(f"\n[Example {idx+1}/{len(df)}] {question[:50]}...")
    
    # Detector-only
    detection = detect_deception(question, answer, strategy='zero_shot')
    detector_score = float(detection.get('deception_score', -1.0))
    detector_pred = 1 if detector_score >= 0.6 else 0
    
    # Full pipeline with consistency + LLM verification
    result = pipeline.process_question(question)
    pipeline_pred = 1 if result.final_decision == 'deceptive' else 0
    
    detector_preds.append(detector_pred)
    pipeline_preds.append(pipeline_pred)
    true_labels.append(true_label)
    
    label_name = ['truthful', 'deceptive'][true_label]
    det_name = ['truthful', 'deceptive'][detector_pred]
    pipe_name = result.final_decision
    pipe_conf = result.confidence
    consistency = result.consistency_score if hasattr(result, 'consistency_score') else -1
    
    results.append({
        'idx': idx,
        'true': label_name,
        'detector': det_name,
        'pipeline': pipe_name,
        'confidence': pipe_conf,
        'consistency': consistency,
    })
    
    print(f"  Label: {label_name:8s}")
    print(f"  Detector: {det_name:8s} (score={detector_score:.2f})")
    print(f"  Pipeline: {pipe_name:8s} (conf={pipe_conf:.2f})")
    if consistency >= 0:
        print(f"  Consistency: {consistency:.2f}")

# Compute metrics
print("\n" + "=" * 70)
print("PERFORMANCE METRICS")
print("=" * 70)

detector_acc = accuracy_score(true_labels, detector_preds)
detector_f1 = f1_score(true_labels, detector_preds)
detector_cm = confusion_matrix(true_labels, detector_preds)

print("\nDETECTOR-ONLY (Baseline):")
print(f"  Accuracy: {detector_acc:.1%}")
print(f"  F1 Score: {detector_f1:.3f}")
print(f"  Confusion Matrix:")
print(f"    TP={detector_cm[1][1]}, FP={detector_cm[0][1]}")
print(f"    FN={detector_cm[1][0]}, TN={detector_cm[0][0]}")

pipeline_acc = accuracy_score(true_labels, pipeline_preds)
pipeline_f1 = f1_score(true_labels, pipeline_preds)
pipeline_cm = confusion_matrix(true_labels, pipeline_preds)

print("\nIMPROVED PIPELINE (Consistency + LLM Verification):")
print(f"  Accuracy: {pipeline_acc:.1%}")
print(f"  F1 Score: {pipeline_f1:.3f}")
print(f"  Confusion Matrix:")
print(f"    TP={pipeline_cm[1][1]}, FP={pipeline_cm[0][1]}")
print(f"    FN={pipeline_cm[1][0]}, TN={pipeline_cm[0][0]}")

print("\nIMPROVEMENT:")
acc_change = (pipeline_acc - detector_acc) * 100
f1_change = (pipeline_f1 - detector_f1)
print(f"  Accuracy change: {acc_change:+.1f}%")
print(f"  F1 change: {f1_change:+.3f}")

if pipeline_acc > detector_acc:
    print(f"\n✅ SUCCESS: Pipeline outperforms baseline!")
else:
    print(f"\n⚠️  Pipeline does not beat baseline yet")

# Show decision distribution
print("\nDecision Distribution:")
print("  Detector:")
print(f"    Truthful: {sum(1 for p in detector_preds if p==0)}")
print(f"    Deceptive: {sum(1 for p in detector_preds if p==1)}")
print("  Pipeline:")
print(f"    Truthful: {sum(1 for d in [r['pipeline'] for r in results] if d=='truthful')}")
print(f"    Deceptive: {sum(1 for d in [r['pipeline'] for r in results] if d=='deceptive')}")
print(f"    Uncertain: {sum(1 for d in [r['pipeline'] for r in results] if d=='uncertain')}")

print("\n" + "=" * 70)
