"""
AI-Powered Misinformation Detection Engine — CLI Interface.

Allows running analysis directly from the command line without launching
the Streamlit web interface.

Usage:
    python main.py --claim "Drinking bleach cures COVID-19"
    python main.py --url "https://example.com/article"
    python main.py --interactive
"""

import argparse
import json
import sys
from datetime import datetime

from config import Config
from knowledge_base.db import init_db, save_analysis, get_history
from modules.fact_checker import check_claim
from modules.gemini_verifier import analyze_with_gemini
from modules.wiki_verifier import verify_with_wikipedia
from modules.metrics_calculator import calculate_text_metrics
from modules.ml_classifier import predict_misinformation_probability
from modules.input_processor import process_input
from modules.linguistic_analyzer import analyze_linguistics
from modules.nlp_analyzer import analyze_text
from modules.scoring_engine import calculate_credibility_score
from modules.source_credibility import assess_source


def print_banner():
    banner = r"""
  __  __ _     ___        __         _____       _            _             
 |  \/  (_)   |_ _|_ __  / _| ___   |  __ \  ___| |_ ___  ___| |_ ___  _ __ 
 | |\/| | |___ | || '_ \| |_ / _ \  | |  | |/ _ \ __/ _ \/ __| __/ _ \| '__|
 | |  | | |___|| || | | |  _| (_) | | |__| |  __/ ||  __/ (__| || (_) | |   
 |_|  |_|_|   |___|_| |_|_|  \___/  |_____/ \___|\__\___|\___|\__\___/|_|   
    """
    print(banner)
    print("=" * 75)
    print(" AI-Powered Misinformation Detection Engine (CLI)")
    print("=" * 75)


def run_pipeline(
    text_input: str,
    input_type: str = "claim",
    api_key: str = None,
    gemini_key: str = None,
) -> dict:
    """Run the complete detection pipeline and return results."""
    print(f"\n[*] Processing input ({input_type})...")
    processed = process_input(text_input, input_type)

    if processed.get("error") and not processed.get("text"):
        print(f"[!] Error: {processed['error']}")
        return {}

    text = processed.get("text", text_input)
    domain = processed.get("domain")
    source_url = processed.get("source_url")

    print("[*] Running NLP Analysis (sentiment, subjectivity, emotions)...")
    nlp_result = analyze_text(text)

    print("[*] Checking Fact-Check database...")
    fact_result = check_claim(text, api_key=api_key or Config.GOOGLE_FACTCHECK_API_KEY)

    source_result = None
    if domain:
        print(f"[*] Assessing source credibility for: {domain}...")
        source_result = assess_source(domain)

    print("[*] Running Linguistic Analysis (clickbait, fallacies, tone)...")
    ling_result = analyze_linguistics(text)

    print("[*] Cross-referencing Wikipedia Knowledge Graph...")
    wiki_result = verify_with_wikipedia(text)

    print("[*] Computing Quantitative Linguistic & Readability Metrics...")
    metrics_result = calculate_text_metrics(text)

    print("[*] Predicting Statistical Misinformation Probability (ML Model)...")
    ml_result = predict_misinformation_probability(text)

    # Gemini analysis
    gemini_key = gemini_key or Config.GEMINI_API_KEY
    gemini_result = None
    if gemini_key:
        print("[*] Running Gemini Deep Reasoning & Multi-Claim Extraction...")
        gemini_result = analyze_with_gemini(text, api_key=gemini_key)

    print("[*] Aggregating credibility scores...")
    score_result = calculate_credibility_score(
        nlp_result=nlp_result,
        fact_check_result=fact_result,
        source_result=source_result,
        linguistic_result=ling_result,
        gemini_result=gemini_result,
        wiki_result=wiki_result,
        ml_result=ml_result,
        metrics_result=metrics_result,
    )

    overall_score = score_result.get("overall_score", 50)
    verdict = score_result.get("verdict", "Needs Verification")

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_type": input_type,
        "claim": text_input[:300],
        "score": overall_score,
        "verdict": verdict,
        "component_scores": score_result.get("component_scores", {}),
        "key_findings": score_result.get("key_findings", []),
        "nlp_analysis": nlp_result,
        "fact_check": fact_result,
        "source_credibility": source_result,
        "linguistic_analysis": ling_result,
        "gemini_analysis": gemini_result,
        "wiki_analysis": wiki_result,
        "metrics_analysis": metrics_result,
        "ml_analysis": ml_result,
        "source_url": source_url,
    }

    try:
        save_analysis(
            claim=text_input[:300],
            verdict=verdict,
            score=overall_score,
            details=result,
            source_url=source_url,
        )
        print("[+] Result saved to Knowledge Base.")
    except Exception as e:
        print(f"[!] Warning: Could not save to KB: {e}")

    return result


def display_results(result: dict):
    """Pretty-print the analysis results to terminal."""
    if not result:
        return

    score = result.get("score", 0)
    verdict = result.get("verdict", "N/A")

    print("\n" + "=" * 75)
    print(" ANALYSIS VERDICT")
    print("=" * 75)
    print(f"  Overall Score : {score:.1f} / 100")
    print(f"  Verdict       : {verdict}")
    print("-" * 75)

    print("\n[Component Breakdown]")
    for component, comp_score in result.get("component_scores", {}).items():
        bar_len = int(comp_score / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {component:<22} [{bar}] {comp_score:>5.1f}/100")

    findings = result.get("key_findings", [])
    if findings:
        print("\n[Key Findings]")
        for finding in findings:
            print(f"  • {finding}")

    # Display Measurable Quantitative Metrics
    met = result.get("metrics_analysis", {})
    if met:
        print("\n[Measurable Linguistic & Readability Metrics]")
        print(f"  Flesch Reading Ease   : {met.get('flesch_reading_ease')}/100")
        print(f"  Grade Level           : Grade {met.get('flesch_kincaid_grade')}")
        print(f"  Lexical Diversity TTR : {met.get('lexical_diversity_ttr'):.3f}")
        print(f"  Uppercase Ratio       : {met.get('uppercase_ratio_pct')}%")
        print(f"  Punctuation Density   : {met.get('punctuation_density_per_100w')} / 100w")
        print(f"  95% Confidence Margin : ±{met.get('margin_of_error_pct')}%")

    # Display ML Statistical Model Probability
    ml_res = result.get("ml_analysis", {})
    if ml_res and ml_res.get("available"):
        print("\n[Statistical Machine Learning Model]")
        print(f"  Misinformation Probability : {ml_res.get('misinfo_probability_pct'):.2f}%")
        print(f"  Credibility Probability    : {ml_res.get('credible_probability_pct'):.2f}%")
        ci = ml_res.get("confidence_interval_95", (0, 0))
        print(f"  95% Confidence Bounds      : [{ci[0]}% - {ci[1]}%]")
        feats = ml_res.get("top_predictive_features", [])
        if feats:
            print("  Top N-Gram Indicators      :")
            for f_item in feats[:3]:
                print(f"    • '{f_item['term']}': {f_item['impact']} (coeff: {f_item['weight']:+.3f})")

    # Display Gemini Atomic Claims if available
    gem = result.get("gemini_analysis", {})
    if gem and gem.get("available"):
        print("\n[Gemini Deep Reasoning & Atomic Claims]")
        summary = gem.get("verdict_summary")
        if summary:
            print(f"  Summary : {summary}")
        claims = gem.get("atomic_claims", [])
        if claims:
            print("  Atomic Claims Breakdown:")
            for i, c in enumerate(claims, 1):
                print(f"    {i}. {c.get('claim')} -> [{c.get('assessment')}] (Verifiability: {c.get('verifiability')})")
        fallacies = [f for f in gem.get("logical_fallacies", []) if f and f.lower() != "none"]
        if fallacies:
            print(f"  Logical Fallacies Detected : {', '.join(fallacies)}")

    print("\n" + "=" * 75)


def interactive_mode():
    """Run an interactive CLI session."""
    print_banner()
    init_db()

    while True:
        print("\nOptions:")
        print("  1. Analyze a Text Claim")
        print("  2. Analyze an Article URL")
        print("  3. View Recent Analysis History")
        print("  4. Launch Streamlit Web UI")
        print("  5. Exit")

        choice = input("\nEnter choice [1-5]: ").strip()

        if choice == "1":
            claim = input("\nEnter claim: ").strip()
            if claim:
                res = run_pipeline(claim, "claim")
                display_results(res)
        elif choice == "2":
            url = input("\nEnter article URL: ").strip()
            if url:
                res = run_pipeline(url, "url")
                display_results(res)
        elif choice == "3":
            history = get_history(limit=10)
            if not history:
                print("\n[!] No history records found.")
            else:
                print("\n" + "=" * 75)
                print(" RECENT ANALYSES")
                print("=" * 75)
                for item in history:
                    print(
                        f"  [{item.get('id')}] {item.get('timestamp')[:19]} | "
                        f"Score: {item.get('credibility_score'):>5.1f} | "
                        f"{item.get('verdict'):<22} | "
                        f"{item.get('claim_text', '')[:40]}..."
                    )
        elif choice == "4":
            print("\n[*] Starting Streamlit Web App...")
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
            break
        elif choice == "5":
            print("\nGoodbye!")
            break
        else:
            print("[!] Invalid option. Please choose 1-5.")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Misinformation Detection Tool (CLI)"
    )
    parser.add_argument("--claim", type=str, help="Text claim to analyze")
    parser.add_argument("--url", type=str, help="Article URL to analyze")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI")
    parser.add_argument("--api-key", type=str, help="Google Fact Check Tools API key")
    parser.add_argument("--gemini-key", type=str, help="Google AI Studio Gemini API key")
    parser.add_argument("--json", action="store_true", help="Output full JSON result")

    args = parser.parse_args()
    init_db()

    if args.claim:
        res = run_pipeline(args.claim, "claim", api_key=args.api_key, gemini_key=args.gemini_key)
        if args.json:
            print(json.dumps(res, indent=2, default=str))
        else:
            display_results(res)
    elif args.url:
        res = run_pipeline(args.url, "url", api_key=args.api_key, gemini_key=args.gemini_key)
        if args.json:
            print(json.dumps(res, indent=2, default=str))
        else:
            display_results(res)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
