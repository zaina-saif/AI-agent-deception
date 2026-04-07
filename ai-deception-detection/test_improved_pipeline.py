#!/usr/bin/env python3
"""Test the improved pipeline on 10 examples"""

import pandas as pd
import os
import sys
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
print('Testing improved pipeline on 10 examples...\n')

df = build_detection_pairs(max_examples=10)
pipeline = DeceptionDetectionPipeline(api_key, strategy='zero_shot', use_consistency_check=False)

detector_preds = []
pipeline_preds = []
true_labels = []

for idx, row in df.iterrows():
    question = row['question']
    answer = row['answer']
    true_label = int(row['label'])
    
    detection = detect_deception(question, answer, strategy='zero_shot')
    detector_score = float(detection.get('deception_score', -1.0))
    detector_pred = 1 if detector_score >= 0.6 else 0
    
    result = pipeline.process_question(question)
    pipeline_pred = 1 if result.final_decision == 'deceptive' else 0
    
    detector_preds.append(detector_pred)
    pipeline_preds.append(pipeline_pred)
    true_labels.append(true_label)
    
    label_name = ['truthful', 'deceptive'][true_label]
    det_name = ['truthful', 'deceptive'][detector_pred]
    pipe_name = result.final_decision
    print(f'Example {idx+1}: True={label_name:8s} | Detector={det_name:8s} | Pipeline={pipe_name:8s}')

print('\n=== METRICS ===')
print(f'Detector Accuracy: {accuracy_score(true_labels, detector_preds):.3f}')
print(f'Pipeline Accuracy: {accuracy_score(true_labels, pipeline_preds):.3f}')
print(f'Detector F1: {f1_score(true_labels, detector_preds):.3f}')
print(f'Pipeline F1: {f1_score(true_labels, pipeline_preds):.3f}')

print(f'\nDetector Confusion Matrix:\n{confusion_matrix(true_labels, detector_preds)}')
print(f'\nPipeline Confusion Matrix:\n{confusion_matrix(true_labels, pipeline_preds)}')
