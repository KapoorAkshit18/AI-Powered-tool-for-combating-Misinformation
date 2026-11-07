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


def run_pipeline(text_input: str, input_type: str = "claim", api_key: str = None) -> dict:
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

    print("[*] Aggregating credibility scores...")
    score_result = calculate_credibility_score(
        nlp_result=nlp_result,
        fact_check_result=fact_result,
        source_result=source_result,
        linguistic_result=ling_result,
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
    parser.add_argument("--json", action="store_true", help="Output full JSON result")

    args = parser.parse_args()
    init_db()

    if args.claim:
        res = run_pipeline(args.claim, "claim", api_key=args.api_key)
        if args.json:
            print(json.dumps(res, indent=2, default=str))
        else:
            display_results(res)
    elif args.url:
        res = run_pipeline(args.url, "url", api_key=args.api_key)
        if args.json:
            print(json.dumps(res, indent=2, default=str))
        else:
            display_results(res)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
