"""Smoke test for the misinformation detection tool."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=== TESTING PACKAGE IMPORTS ===")
import streamlit
import textblob
import plotly
import sklearn
import trafilatura
import bs4
import pandas
import nltk
import dotenv
print(f"Streamlit: {streamlit.__version__}")
print(f"Plotly: {plotly.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print("All package imports OK!")

print("\n=== TESTING PROJECT MODULES ===")
from modules.input_processor import process_input
from modules.nlp_analyzer import analyze_text
from modules.fact_checker import check_claim
from modules.source_credibility import assess_source
from modules.linguistic_analyzer import analyze_linguistics
from modules.scoring_engine import calculate_credibility_score
from knowledge_base.db import init_db, save_analysis, get_history
from config import Config
print("All project modules imported OK!")

print("\n=== FUNCTIONAL TEST ===")
init_db()
print("Database initialized OK")

test_claim = "BREAKING: Scientists PROVE that vaccines cause autism!!! This is HUGE!!!"
print(f"Test claim: {test_claim}")

nlp = analyze_text(test_claim)
print(f"NLP Analysis: credibility={nlp.get('nlp_credibility', 'N/A')}")

fact = check_claim(test_claim)
print(f"Fact Check: available={fact.get('available', 'N/A')}")

source = assess_source("infowars.com")
print(f"Source Check (infowars.com): score={source.get('credibility_score', 'N/A')}")

ling = analyze_linguistics(test_claim)
print(f"Linguistic Analysis: credibility={ling.get('linguistic_credibility', 'N/A')}")

score = calculate_credibility_score(nlp, fact, source, ling)
print(f"Overall Verdict: {score.get('verdict', 'N/A')} (score={score.get('overall_score', 'N/A')})")

print("\n=== SMOKE TEST PASSED ===")
