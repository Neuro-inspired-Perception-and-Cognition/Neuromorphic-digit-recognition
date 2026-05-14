"""
This module synchronizes event-based predictions with ground-truth data
using timestamps. It calculates overall accuracy, per-class precision,
and a confusion matrix to identify systematic classification errors.
"""

import numpy as np
from collections import defaultdict

correct_per_class = defaultdict(int)
total_per_class = defaultdict(int)
confusion = np.zeros((10, 10), dtype=int)

pred_file = r"C:\Users\vedme\OneDrive\Робочий стіл\BT\experiments\detections_test_validation.txt"
true_file = r"C:\Users\vedme\OneDrive\Робочий стіл\BT\experiments\digit_times_test_validation.txt"
shift = 0

def load_preds(path):
    preds = []
    with open(path) as f:
        for line in f:
            parts = line.split(",")
            digit = int(parts[0])
            t = float(parts[1])
            preds.append((digit, t))
    return preds

def load_true(path):
    true = []
    with open(path) as f:
        for line in f:
            digit, t = line.strip().split(",")
            true.append((int(digit), float(t)))
    return true

preds = load_preds(pred_file)
truth = load_true(true_file)
truth_times = np.array([t for _, t in truth])
correct = 0
total = 0
matches = []

for p_digit, p_time in preds:
    p_time = p_time - shift
    # find last true time before prediction
    idx = np.searchsorted(truth_times, p_time) - 1

    if idx < 0:
        continue

    true_digit = truth[idx][0]
    matches.append((p_time, p_digit, true_digit))
    total_per_class[true_digit] += 1

    if p_digit == true_digit:
        correct += 1
        correct_per_class[true_digit] += 1

    total += 1
    confusion[true_digit, p_digit] += 1

print("Total:", total)
print("Correct:", correct)
print("Accuracy:", correct / total)

print("\nPer-class accuracy:")

for digit in sorted(total_per_class.keys()):
    acc = correct_per_class[digit] / total_per_class[digit]
    print(f"Digit {digit}: {acc:.3f} ({correct_per_class[digit]}/{total_per_class[digit]})")

print("\nConfusion matrix:")
print(confusion)