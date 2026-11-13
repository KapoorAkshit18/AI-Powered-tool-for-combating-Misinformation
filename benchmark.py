"""
Benchmark CLI Runner for Misinformation Detection Engine.

Executes ground-truth benchmark evaluation and outputs formal statistical metrics:
Accuracy, Precision, Recall, Specificity, F1-Score, and Confusion Matrix.

Usage:
    python benchmark.py
"""

import sys
from modules.benchmark_engine import run_benchmark_evaluation
from modules.fact_checker import check_claim
from modules.input_processor import process_input
from modules.linguistic_analyzer import analyze_linguistics
from modules.metrics_calculator import calculate_text_metrics
from modules.ml_classifier import predict_misinformation_probability
from modules.nlp_analyzer import analyze_text
from modules.scoring_engine import calculate_credibility_score
from modules.wiki_verifier import verify_with_wikipedia


def evaluate_single_claim(claim_text: str) -> dict:
    """Run pipeline without requiring external API keys for benchmark consistency."""
    processed = process_input(claim_text, "claim")
    text = processed.get("text", claim_text)
    nlp = analyze_text(text)
    fc = check_claim(text, api_key=None)
    ling = analyze_linguistics(text)
    wiki = verify_with_wikipedia(text)
    ml_res = predict_misinformation_probability(text)
    metrics_res = calculate_text_metrics(text)

    score_res = calculate_credibility_score(
        nlp_result=nlp,
        fact_check_result=fc,
        source_result=None,
        linguistic_result=ling,
        wiki_result=wiki,
        ml_result=ml_res,
        metrics_result=metrics_res,
    )
    return {"score": score_res.get("overall_score", 50.0)}


def main():
    print("=" * 75)
    print(" 📊 MISINFORMATION DETECTION ENGINE — EMPIRICAL BENCHMARK EVALUATION")
    print("=" * 75)
    print("[*] Running evaluation across 25 ground-truth benchmark claims...")

    report = run_benchmark_evaluation(evaluate_single_claim)
    m = report["metrics"]
    cm = report["confusion_matrix"]

    print("\n" + "─" * 75)
    print(" 📈 FORMAL STATISTICAL PERFORMANCE METRICS")
    print("─" * 75)
    print(f"  Total Benchmark Samples  : {report['total_samples']}")
    print(f"  Accuracy                 : {m['accuracy_pct']:>6.2f} %")
    print(f"  Precision (PPV)          : {m['precision_pct']:>6.2f} %")
    print(f"  Recall (Sensitivity)     : {m['recall_sensitivity_pct']:>6.2f} %")
    print(f"  Specificity (TNR)        : {m['specificity_pct']:>6.2f} %")
    print(f"  F1 Score                 : {m['f1_score']:>6.3f} (0.0 to 1.0)")
    print(f"  False Positive Rate (FPR): {m['false_positive_rate_pct']:>6.2f} %")
    print(f"  Brier Calibration Score  : {m['brier_calibration_score']:>6.4f} (lower is better)")

    print("\n" + "─" * 75)
    print(" 🔲 CONFUSION MATRIX (Positive Class = Misinformation/False)")
    print("─" * 75)
    print(f"                       Predicted False    Predicted Not-False")
    print(f"  Actual False (Misinfo):   TP = {cm['true_positives']:<3}             FN = {cm['false_negatives']:<3}")
    print(f"  Actual True/Unverified:   FP = {cm['false_positives']:<3}             TN = {cm['true_negatives']:<3}")

    print("\n" + "─" * 75)
    print(" 📋 SAMPLE BREAKDOWN")
    print("─" * 75)
    print(f"  {'Claim':<45} | {'Actual':<10} | {'Pred':<10} | {'Score':<5}")
    print("  " + "-" * 72)
    for r in report["detailed_results"][:10]:
        claim_short = (r["claim"][:42] + "...") if len(r["claim"]) > 42 else r["claim"]
        print(f"  {claim_short:<45} | {r['ground_truth']:<10} | {r['pred_label']:<10} | {r['pred_score']:>5.1f}")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
