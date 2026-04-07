import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_csv('eval/full_pipeline_comparison.csv')
true_labels = df['true_label'].values
detector_preds = df['detector_prediction'].values
pipeline_preds = df['pipeline_prediction'].values

print('=== IMPROVED PIPELINE RESULTS ===\n')
print(f'Total examples: {len(df)}')
print(f'Deceptive: {sum(true_labels)}, Truthful: {len(true_labels) - sum(true_labels)}\n')

print('DETECTOR-ONLY:')
det_acc = accuracy_score(true_labels, detector_preds)
det_f1 = f1_score(true_labels, detector_preds)
print(f'  Accuracy: {det_acc:.3f}')
print(f'  F1 Score: {det_f1:.3f}')
det_cm = confusion_matrix(true_labels, detector_preds)
print(f'  TP: {det_cm[1][1]}, FP: {det_cm[0][1]}, FN: {det_cm[1][0]}, TN: {det_cm[0][0]}')

print('\nPIPELINE (IMPROVED):')
pipe_acc = accuracy_score(true_labels, pipeline_preds)
pipe_f1 = f1_score(true_labels, pipeline_preds)
print(f'  Accuracy: {pipe_acc:.3f}')
print(f'  F1 Score: {pipe_f1:.3f}')
pipe_cm = confusion_matrix(true_labels, pipeline_preds)
print(f'  TP: {pipe_cm[1][1]}, FP: {pipe_cm[0][1]}, FN: {pipe_cm[1][0]}, TN: {pipe_cm[0][0]}')

print(f'\n✅ IMPROVEMENT: {(pipe_acc - det_acc)*100:+.1f}% accuracy')

# Show decision distribution
print(f'\nDecision Distribution:')
for decision in ['truthful', 'deceptive', 'uncertain']:
    count = sum(1 for d in df['pipeline_decision'] if d == decision)
    print(f'  {decision}: {count}')
