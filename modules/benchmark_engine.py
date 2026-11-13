"""
Empirical Benchmark Evaluation Suite for Misinformation Detection.

Provides ground-truth labeled test sets (True, False, Unverified) and computes
formal machine learning performance metrics:
  - Confusion Matrix (TP, FP, TN, FN)
  - Accuracy (%)
  - Precision / Positive Predictive Value (%)
  - Recall / Sensitivity (%)
  - Specificity / True Negative Rate (%)
  - F1 Score
  - False Positive Rate (FPR)
  - Brier Calibration Score
"""

import json
from typing import Any, Callable

# Ground-truth labeled benchmark dataset
BENCHMARK_DATASET: list[dict[str, Any]] = [
    # --- Class: Misinformation / False (Label: "False") ---
    {"claim": "Drinking bleach cures COVID-19 infection", "ground_truth": "False", "category": "Health Hoax"},
    {"claim": "Vaccines implant 5G microchips into patients", "ground_truth": "False", "category": "Conspiracy"},
    {"claim": "The Earth is flat and NASA fakes all satellite images", "ground_truth": "False", "category": "Science Denial"},
    {"claim": "5G cellular networks transmit respiratory viruses", "ground_truth": "False", "category": "Tech Hoax"},
    {"claim": "Lemons cure cancer 10000 times better than chemotherapy", "ground_truth": "False", "category": "Medical Fraud"},
    {"claim": "Secret government aerosol spraying called chemtrails controls human minds", "ground_truth": "False", "category": "Conspiracy"},
    {"claim": "Eating garlic prevents coronavirus infection completely", "ground_truth": "False", "category": "Health Hoax"},
    {"claim": "COVID-19 was patented by the CDC years before the outbreak", "ground_truth": "False", "category": "Fabrication"},
    {"claim": "Putting onions in your socks cures the flu overnight", "ground_truth": "False", "category": "Folk Myth"},
    {"claim": "Fluoride in tap water is a secret weapon designed to lower population IQ", "ground_truth": "False", "category": "Conspiracy"},

    # --- Class: Credible / True (Label: "True") ---
    {"claim": "Water consists of hydrogen and oxygen molecules", "ground_truth": "True", "category": "Basic Science"},
    {"claim": "Smoking tobacco cigarettes increases the risk of lung cancer", "ground_truth": "True", "category": "Public Health"},
    {"claim": "Penicillin was discovered by Alexander Fleming in 1928", "ground_truth": "True", "category": "History"},
    {"claim": "Photosynthesis is the process by which green plants transform light energy into chemical energy", "ground_truth": "True", "category": "Biology"},
    {"claim": "The United States Constitution was drafted in Philadelphia in 1787", "ground_truth": "True", "category": "History"},
    {"claim": "The human heart contains four chambers: two atria and two ventricles", "ground_truth": "True", "category": "Anatomy"},
    {"claim": "The speed of light in a vacuum is approximately 299,792 kilometers per second", "ground_truth": "True", "category": "Physics"},
    {"claim": "Regular physical exercise is associated with lower cardiovascular disease mortality", "ground_truth": "True", "category": "Medicine"},
    {"claim": "The Earth orbits around the Sun once every 365.25 days approximately", "ground_truth": "True", "category": "Astronomy"},
    {"claim": "DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine", "ground_truth": "True", "category": "Genetics"},

    # --- Class: Unverified / Context-Dependent (Label: "Unverified") ---
    {"claim": "indians are born in america", "ground_truth": "Unverified", "category": "Context Ambiguity"},
    {"claim": "Eating chocolate every day makes you live longer", "ground_truth": "Unverified", "category": "Sensationalized Study"},
    {"claim": "Artificial intelligence will replace 80 percent of all jobs by next year", "ground_truth": "Unverified", "category": "Speculative Hyperbole"},
    {"claim": "Ancient civilizations possessed advanced electrical technology", "ground_truth": "Unverified", "category": "Pseudohistory"},
    {"claim": "Stock market will experience record growth this quarter according to analysts", "ground_truth": "Unverified", "category": "Market Rumor"},
]


def evaluate_prediction(pred_score: float) -> str:
    """
    Map a continuous credibility score (0-100) to a discrete prediction label.
      - Score < 45: "False" (Likely Misinformation)
      - Score >= 65: "True" (Likely Credible)
      - 45 <= Score < 65: "Unverified" (Needs Verification)
    """
    if pred_score < 45.0:
        return "False"
    elif pred_score >= 65.0:
        return "True"
    else:
        return "Unverified"


def run_benchmark_evaluation(pipeline_fn: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """
    Run empirical evaluation across the full benchmark test set.

    Args:
        pipeline_fn: Function that takes a claim string and returns a dict with 'score'.

    Returns:
        Structured evaluation report with precision, recall, F1, accuracy, and confusion matrix.
    """
    results: list[dict[str, Any]] = []

    # Binary classification metrics: Misinformation (Positive Class = "False")
    # True Positive (TP): Ground Truth is False, Model predicts False (<45)
    # False Positive (FP): Ground Truth is NOT False, Model predicts False (<45)
    # True Negative (TN): Ground Truth is NOT False, Model predicts NOT False (>=45)
    # False Negative (FN): Ground Truth is False, Model predicts NOT False (>=45)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Multi-class accuracy
    exact_matches = 0
    total = len(BENCHMARK_DATASET)
    brier_sum = 0.0

    for item in BENCHMARK_DATASET:
        claim = item["claim"]
        gt = item["ground_truth"]
        category = item["category"]

        try:
            pred_res = pipeline_fn(claim)
            score = float(pred_res.get("score", 50.0))
        except Exception:
            score = 50.0

        pred_label = evaluate_prediction(score)
        is_exact = (pred_label == gt)
        if is_exact:
            exact_matches += 1

        # Binary Confusion Matrix where Positive = "Misinformation/False"
        is_gt_false = (gt == "False")
        is_pred_false = (pred_label == "False")

        if is_gt_false and is_pred_false:
            tp += 1
            cm_status = "TP"
        elif not is_gt_false and is_pred_false:
            fp += 1
            cm_status = "FP"
        elif not is_gt_false and not is_pred_false:
            tn += 1
            cm_status = "TN"
        else:
            fn += 1
            cm_status = "FN"

        # Brier score calculation (target probability: False -> 1.0 prob of misinfo, True -> 0.0)
        prob_misinfo = (100.0 - score) / 100.0
        target = 1.0 if gt == "False" else (0.5 if gt == "Unverified" else 0.0)
        brier_sum += (prob_misinfo - target) ** 2

        results.append({
            "claim": claim,
            "ground_truth": gt,
            "category": category,
            "pred_score": score,
            "pred_label": pred_label,
            "cm_status": cm_status,
            "exact_match": is_exact,
        })

    # Mathematical metric calculations
    accuracy = (exact_matches / total * 100.0) if total > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    brier_score = round(brier_sum / total, 4) if total > 0 else 0.0

    return {
        "total_samples": total,
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "metrics": {
            "accuracy_pct": round(accuracy, 2),
            "precision_pct": round(precision, 2),
            "recall_sensitivity_pct": round(recall, 2),
            "specificity_pct": round(specificity, 2),
            "f1_score": round(f1 / 100.0, 3),  # standard 0-1 scale
            "false_positive_rate_pct": round(fpr, 2),
            "brier_calibration_score": brier_score,
        },
        "detailed_results": results,
    }
