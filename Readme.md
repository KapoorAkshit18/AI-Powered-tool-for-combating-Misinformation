# 🛡️ AI-Powered Misinformation Detection Tool

An intelligent, multi-layered misinformation detection engine built with Python and Streamlit. This is a **detection and analysis tool** (not a chatbot) that evaluates claims, articles, and URLs to produce structured credibility verdicts backed by multiple analytical lenses.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Overview

The tool evaluates content through **5 independent analysis modules**:

| Module | What It Analyzes |
|---|---|
| **🤖 Machine Learning Model** | Trained TF-IDF + Logistic Regression calculating exact mathematical probability $P(\text{Misinfo} \mid x) \in [0, 1]$ and top n-gram feature weights |
| **📏 Quantitative Text Metrics** | Flesch Reading Ease, Flesch-Kincaid Grade Level, Lexical Diversity (TTR), Uppercase %, Punctuation density, and 95% Confidence Intervals |
| **📊 Benchmark Evaluation** | Ground-truth validation across 25 benchmark claims measuring Accuracy, Precision, Recall, Specificity, F1-Score, and Confusion Matrix |
| **🤖 Gemini AI Verifier** | Multi-claim atomic decomposition, logical fallacies (strawman, false dilemma, etc.), and manipulation tactics |
| **📚 Wikipedia Knowledge Graph** | Entity disambiguation and knowledge consensus checking against Wikipedia |
| **🧠 NLP Analyzer** | Sentiment polarity, subjectivity, and density of emotional/loaded language |
| **✅ Fact Checker** | Google Fact Check Tools API integration for matching against verified verdicts |
| **🌐 Source Credibility** | Domain reputation lookup against verified reliable and unreliable databases |
| **📝 Linguistic Analyzer** | Clickbait patterns (ALL CAPS, excessive punctuation, superlatives), sensationalism, and logical fallacies |
| **⚖️ Scoring Engine** | Weighted aggregate credibility score (0–100) with tailored recommendations |
| **⚖️ Scoring Engine** | Weighted aggregate credibility score (0–100) with calibrated safety guards |

All analyses are logged in a local **SQLite Knowledge Base** for historical tracking and review.

---

## 🏗️ Architecture

```
User Input (Claim / URL / Pasted Article)
                    │
                    ▼
          ┌───────────────────┐
          │  Input Processor  │
          └─────────┬─────────┘
                    │
       ┌────────────┼────────────┬────────────┐
       ▼            ▼            ▼            ▼
 ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐
 │   NLP    │ │   Fact   │ │  Source  │ │Linguistic │
 │ Analyzer │ │ Checker  │ │Credibil. │ │ Analyzer  │
 └─────┬────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘
       │           │            │             │
       └───────────┼────────────┴─────────────┘
                   ▼
         ┌───────────────────┐
         │  Scoring Engine   │
         └─────────┬─────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│ Streamlit UI    │ │ SQLite KB       │
│ - Verdict       │ │ - History       │
│ - Gauge Chart   │ │ - Search        │
│ - Radar Chart   │ │                 │
│ - Export Report │ │                 │
└─────────────────┘ └─────────────────┘
```

---

## 📂 Repository Structure

```
AI-Powered-tool-for-combating-Misinformation/
├── app.py                      # Main Streamlit dashboard application
├── config.py                   # Configuration & environment management
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── smoke_test.py               # Verification test suite
│
├── modules/
│   ├── input_processor.py      # Claim, URL (trafilatura), and article handler
│   ├── nlp_analyzer.py         # Sentiment, subjectivity, emotional language
│   ├── fact_checker.py         # Google Fact Check Tools API client
│   ├── source_credibility.py   # Domain reputation evaluator
│   ├── linguistic_analyzer.py  # Clickbait, sensationalism, fallacies
│   └── scoring_engine.py       # Aggregate scoring and verdict generation
│
├── knowledge_base/
│   └── db.py                   # SQLite database operations (CRUD & search)
│
├── data/
│   ├── reliable_sources.json   # Whitelist of credible domains & categories
│   ├── unreliable_sources.json # Blacklist of flagged domains & categories
│   └── misinformation_kb.db    # Local SQLite database (auto-created)
│
└── GCP_Labs/                   # Google Cloud reference notebooks
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- pip package manager

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/AI-Powered-tool-for-combating-Misinformation.git
cd AI-Powered-tool-for-combating-Misinformation

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Tool

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🔑 Authentication & API Keys (Optional)

The tool **works fully out-of-the-box** without any API keys. 

To enable the **Google Fact Check module**:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable the **Fact Check Tools API**
4. Create an API Key under **APIs & Services → Credentials**
5. Enter the key either:
   - In the app: open the sidebar **⚙️ Settings** and paste it, or
   - In a `.env` file:
     ```bash
     cp .env.example .env
     # Add your key:
     # GOOGLE_FACTCHECK_API_KEY=your_key_here
     ```

> 💡 **Note on Google AI Studio vs Google Cloud Keys:**  
> A Google AI Studio key is for Gemini model inference. For fact-checking, you need the **Google Fact Check Tools API** from Google Cloud Console. Both are free.

---

## 📊 Score Interpretation

| Score | Verdict | Meaning |
|---|---|---|
| **70 – 100** | 🟢 **Likely Credible** | Content shows strong markers of reliable reporting |
| **55 – 69** | 🟡 **Needs Caution** | Minor concerns; verify with secondary sources |
| **40 – 54** | 🟠 **Questionable** | Multiple red flags detected; high skepticism advised |
| **0 – 39** | 🔴 **Likely Misinformation** | Strong patterns of misleading content or known bad source |

---

## 🧪 Testing

Run the included smoke test to verify all modules and dependencies:

```bash
python smoke_test.py
```

---

## ⚠️ Disclaimer

This tool is designed to **assist human judgment**, not replace it. Credibility scores are probabilistic indicators based on linguistic patterns, domain reputation, and available fact-checking databases. Always cross-reference critical claims across multiple independent, established sources.

---

## 📄 License

This project is licensed under the MIT License.
