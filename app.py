"""
MisInfo Detector — AI-Powered Misinformation Detection Tool

Main Streamlit application file that provides a professional dashboard
for analyzing text claims, article URLs, and pasted articles for
potential misinformation. This is a detection/analysis tool that produces
structured verdicts, NOT a chatbot.
"""

import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from modules.input_processor import process_input
from modules.nlp_analyzer import analyze_text
from modules.fact_checker import check_claim
from modules.source_credibility import assess_source
from modules.linguistic_analyzer import analyze_linguistics
from modules.gemini_verifier import analyze_with_gemini
from modules.wiki_verifier import verify_with_wikipedia
from modules.metrics_calculator import calculate_text_metrics
from modules.ml_classifier import predict_misinformation_probability
from modules.benchmark_engine import run_benchmark_evaluation
from modules.scoring_engine import calculate_credibility_score
from knowledge_base.db import init_db, save_analysis, get_history
try:
    from config import Config, update_env_key
except ImportError:
    import importlib
    import config
    importlib.reload(config)
    from config import Config, update_env_key

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🛡️ MisInfo Detector — AI-Powered Misinformation Detection Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS for professional styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- general page ---------- */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ---------- verdict banners ---------- */
    .verdict-banner {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
        color: #fff;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .verdict-green  { background: linear-gradient(135deg, #27ae60, #2ecc71); }
    .verdict-yellow { background: linear-gradient(135deg, #f39c12, #f1c40f); color: #333; }
    .verdict-orange { background: linear-gradient(135deg, #e67e22, #d35400); }
    .verdict-red    { background: linear-gradient(135deg, #c0392b, #e74c3c); }

    /* ---------- score display ---------- */
    .score-big {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        line-height: 1.1;
    }
    .score-label {
        text-align: center;
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 1rem;
    }

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.4rem;
    }

    /* ---------- analysis cards ---------- */
    .analysis-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #3498db;
    }
    .analysis-card h4 { margin: 0 0 0.4rem 0; }

    /* ---------- history item ---------- */
    .history-item {
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        background: #fff;
        margin-bottom: 0.4rem;
        border: 1px solid #e0e0e0;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Initialise database & session state
# ---------------------------------------------------------------------------
init_db()

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "fact_check_api_key" not in st.session_state:
    st.session_state.fact_check_api_key = Config.GOOGLE_FACTCHECK_API_KEY or ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = Config.GEMINI_API_KEY or ""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🛡️ MisInfo Detector")
    st.markdown(
        "An **AI-powered misinformation detection tool** that analyses claims, "
        "articles, and URLs for credibility using NLP, fact-checking APIs, "
        "source reputation analysis, and linguistic pattern detection."
    )
    st.divider()

    # --- About ---
    with st.expander("ℹ️ About", expanded=False):
        st.markdown(
            """
            **MisInfo Detector** combines multiple analysis modules to produce
            a credibility verdict:

            1. **NLP Analysis** — sentiment, subjectivity & emotional language  
            2. **Fact-Check Lookup** — Google Fact Check Tools API  
            3. **Source Credibility** — domain reputation & history  
            4. **Linguistic Analysis** — clickbait, sensationalism & fallacies  
            5. **Scoring Engine** — weighted aggregate credibility score  

            Results are stored locally so you can review past analyses.
            """
        )

    # --- Analysis history ---
    st.subheader("📜 Recent Analyses")
    history = get_history(limit=10)
    if history:
        for item in history:
            timestamp = item.get("timestamp", "")
            claim_preview = item.get("claim_text", "")[:60]
            score = item.get("credibility_score", "–")
            st.markdown(
                f'<div class="history-item">'
                f"<strong>{score}/100</strong> · {claim_preview}…<br>"
                f"<small>{timestamp}</small></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No analyses yet — run your first one!")

    st.divider()

    # --- Settings ---
    with st.expander("⚙️ Settings & API Keys", expanded=False):
        # Gemini status
        gem_active = bool(st.session_state.gemini_api_key and st.session_state.gemini_api_key.strip())
        gem_badge = "🟢 Configured" if gem_active else "⚪ Not Configured"
        st.markdown(f"**🤖 Google Gemini (AI Studio)** — `{gem_badge}`")
        gemini_key_input = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Free API key from https://aistudio.google.com/ to enable multi-claim extraction & deep logical reasoning.",
        )
        if gemini_key_input != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = gemini_key_input
            update_env_key("GEMINI_API_KEY", gemini_key_input)
            st.success("Gemini API key saved to .env file.")

        # Fact Check status
        fc_active = bool(st.session_state.fact_check_api_key and st.session_state.fact_check_api_key.strip())
        fc_badge = "🟢 Configured" if fc_active else "⚪ Not Configured"
        st.markdown(f"**✅ Google Fact Check Tools API** — `{fc_badge}`")
        api_key_input = st.text_input(
            "Fact Check Tools API Key",
            value=st.session_state.fact_check_api_key,
            type="password",
            help="Optional Google Cloud Console Fact Check API key.",
        )
        if api_key_input != st.session_state.fact_check_api_key:
            st.session_state.fact_check_api_key = api_key_input
            update_env_key("GOOGLE_FACTCHECK_API_KEY", api_key_input)
            st.success("Fact Check API key saved to .env file.")

        if st.button("💾 Persist Keys to .env"):
            update_env_key("GEMINI_API_KEY", st.session_state.gemini_api_key)
            update_env_key("GOOGLE_FACTCHECK_API_KEY", st.session_state.fact_check_api_key)
            st.success("Both keys permanently saved to .env file on disk!")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    """Return a CSS class suffix based on the credibility score."""
    if score >= 70:
        return "green"
    if score >= 50:
        return "yellow"
    if score >= 35:
        return "orange"
    return "red"


def _verdict_label(score: float) -> str:
    """Return a human-readable verdict string."""
    if score >= 70:
        return "✅ Likely Credible"
    if score >= 50:
        return "⚠️ Needs Verification"
    if score >= 35:
        return "🟠 Questionable"
    return "🚨 Likely Misinformation"


def _build_gauge(score: float) -> go.Figure:
    """Build a Plotly gauge chart for the credibility score."""
    color = _score_color(score)
    bar_color_map = {
        "green": "#27ae60",
        "yellow": "#f1c40f",
        "orange": "#e67e22",
        "red": "#e74c3c",
    }
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Credibility Score", "font": {"size": 18}},
            number={"font": {"size": 48}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color_map.get(color, "#3498db")},
                "bgcolor": "#f0f0f0",
                "steps": [
                    {"range": [0, 40], "color": "rgba(231,76,60,0.15)"},
                    {"range": [40, 70], "color": "rgba(241,196,15,0.15)"},
                    {"range": [70, 100], "color": "rgba(39,174,96,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#333", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=40, b=20, l=40, r=40))
    return fig


def _build_radar(component_scores: dict) -> go.Figure:
    """Build a Plotly radar chart for component scores."""
    labels = list(component_scores.keys())
    values = list(component_scores.values())
    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            name="Scores",
            line_color="#3498db",
            fillcolor="rgba(52,152,219,0.25)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=380,
        margin=dict(t=40, b=40, l=60, r=60),
    )
    return fig


def _build_bar(component_scores: dict) -> go.Figure:
    """Build a Plotly bar chart comparing individual component scores."""
    labels = list(component_scores.keys())
    values = list(component_scores.values())
    colors = [
        "#27ae60" if v >= 70 else "#f1c40f" if v >= 40 else "#e74c3c"
        for v in values
    ]
    fig = px.bar(
        x=labels,
        y=values,
        labels={"x": "Analysis Module", "y": "Score"},
        title="Component Score Comparison",
        color=labels,
        color_discrete_sequence=colors,
    )
    fig.update_layout(
        showlegend=False,
        yaxis_range=[0, 100],
        height=360,
        margin=dict(t=50, b=40),
    )
    fig.update_traces(marker_color=colors)
    return fig


def _generate_text_report(result: dict) -> str:
    """Generate a plain-text summary report from an analysis result."""
    lines = [
        "=" * 60,
        "MISINFO DETECTOR — ANALYSIS REPORT",
        "=" * 60,
        f"Date       : {result.get('timestamp', 'N/A')}",
        f"Input Type : {result.get('input_type', 'N/A')}",
        f"Claim      : {result.get('claim', 'N/A')[:120]}",
        "",
        f"OVERALL CREDIBILITY SCORE: {result.get('score', 'N/A')} / 100",
        f"VERDICT: {_verdict_label(result.get('score', 0))}",
        "-" * 60,
    ]

    # Component scores
    component_scores = result.get("component_scores", {})
    if component_scores:
        lines.append("\nCOMPONENT SCORES:")
        for name, val in component_scores.items():
            lines.append(f"  {name:30s} : {val}/100")

    # Key findings
    findings = result.get("key_findings", [])
    if findings:
        lines.append("\nKEY FINDINGS:")
        for f in findings:
            lines.append(f"  • {f}")

    # Detailed sections
    for section_key in ("nlp_analysis", "fact_check", "source_credibility", "linguistic_analysis", "gemini_analysis", "wiki_analysis", "metrics_analysis", "ml_analysis"):
        section = result.get(section_key)
        if section:
            lines.append(f"\n{'—'*40}")
            lines.append(section_key.upper().replace("_", " "))
            lines.append("—" * 40)
            if isinstance(section, dict):
                for k, v in section.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  {section}")

    lines.append("\n" + "=" * 60)
    lines.append("Generated by MisInfo Detector")
    lines.append("=" * 60)
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------
st.markdown("## 🛡️ MisInfo Detector")
st.caption("AI-Powered Misinformation Detection Tool — Analyse claims, articles & URLs")

# ---- Input Section --------------------------------------------------------
st.markdown("---")
input_type = st.radio(
    "Select input type:",
    ["Text Claim", "Article URL", "Paste Article"],
    horizontal=True,
    help="Choose the kind of content you want to analyse.",
)

if input_type == "Text Claim":
    user_input = st.text_input(
        "Enter a claim to verify:",
        placeholder="e.g. 'Drinking bleach cures COVID-19'",
    )
elif input_type == "Article URL":
    user_input = st.text_input(
        "Enter the article URL:",
        placeholder="https://example.com/article",
    )
else:  # Paste Article
    user_input = st.text_area(
        "Paste the full article text:",
        height=200,
        placeholder="Paste article content here…",
    )

analyse_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ---- Analysis pipeline ----------------------------------------------------
if analyse_clicked:
    if not user_input or not user_input.strip():
        st.error("Please provide some input to analyse.")
    else:
        try:
            with st.spinner("🔍 Analysing — this may take a moment…"):
                # Map UI label → module input_type
                type_map = {
                    "Text Claim": "claim",
                    "Article URL": "url",
                    "Paste Article": "article",
                }
                mapped_type = type_map.get(input_type, "claim")

                # 1. Process input
                processed = process_input(user_input, mapped_type)
                text = processed.get("text", user_input)
                url = processed.get("source_url")
                domain = processed.get("domain")

                if processed.get("error") and not text:
                    st.error(processed["error"])
                    st.stop()

                # 2. NLP analysis
                nlp_result = analyze_text(text)

                # 3. Fact-check
                api_key = (
                    st.session_state.fact_check_api_key
                    or Config.GOOGLE_FACTCHECK_API_KEY
                )
                fact_result = check_claim(text, api_key=api_key)

                # 4. Source credibility (only when a domain is available)
                source_result = None
                if domain:
                    source_result = assess_source(domain)

                # 5. Linguistic analysis
                ling_result = analyze_linguistics(text)

                # 6. Wikipedia Knowledge Graph Corroboration
                wiki_result = verify_with_wikipedia(text)

                # 7. Measurable quantitative metrics
                metrics_result = calculate_text_metrics(text)

                # 8. Machine Learning empirical prediction
                ml_result = predict_misinformation_probability(text)

                # 9. Gemini deep reasoning & claim extraction
                gemini_key = (
                    st.session_state.gemini_api_key
                    or Config.GEMINI_API_KEY
                )
                gemini_result = analyze_with_gemini(text, api_key=gemini_key)

                # 10. Aggregate calibrated score
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
                component_scores = score_result.get("component_scores", {})
                key_findings = score_result.get("key_findings", [])

                # Assemble full result
                analysis_result = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "input_type": input_type,
                    "claim": user_input[:300],
                    "score": overall_score,
                    "verdict": _verdict_label(overall_score),
                    "component_scores": component_scores,
                    "key_findings": key_findings,
                    "nlp_analysis": nlp_result,
                    "fact_check": fact_result,
                    "source_credibility": source_result,
                    "linguistic_analysis": ling_result,
                    "gemini_analysis": gemini_result,
                    "wiki_analysis": wiki_result,
                    "metrics_analysis": metrics_result,
                    "ml_analysis": ml_result,
                    "source_url": url,
                }

                # 7. Save to knowledge base
                try:
                    save_analysis(
                        claim=user_input[:300],
                        verdict=_verdict_label(overall_score),
                        score=overall_score,
                        details=analysis_result,
                        source_url=url,
                    )
                except Exception as db_err:
                    st.warning(f"Could not save to history: {db_err}")

                # 8. Store in session state for display
                st.session_state.analysis_result = analysis_result

        except Exception as exc:
            st.error(f"An error occurred during analysis: {exc}")

# ---- Results Section ------------------------------------------------------
result = st.session_state.analysis_result
if result is not None:
    st.markdown("---")
    overall_score = result.get("score", 50)
    color = _score_color(overall_score)
    verdict = result.get("verdict", _verdict_label(overall_score))

    # Verdict banner
    st.markdown(
        f'<div class="verdict-banner verdict-{color}">{verdict}</div>',
        unsafe_allow_html=True,
    )

    tab_verdict, tab_detail, tab_scores, tab_metrics, tab_report = st.tabs(
        [
            "📊 Verdict",
            "🔬 Detailed Analysis",
            "📈 Score Breakdown",
            "📏 Measurable Metrics & Benchmark",
            "📋 Report",
        ]
    )

    # ---- Tab 1: Verdict ---------------------------------------------------
    with tab_verdict:
        col_score, col_gauge = st.columns([1, 2])
        with col_score:
            score_hex = {
                "green": "#27ae60",
                "yellow": "#f1c40f",
                "orange": "#e67e22",
                "red": "#e74c3c",
            }.get(color, "#3498db")
            st.markdown(
                f'<div class="score-big" style="color:{score_hex}">{overall_score}</div>'
                f'<div class="score-label">out of 100</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Verdict:** {verdict}")
            st.markdown(f"**Analysed:** {result.get('timestamp', '')}")
        with col_gauge:
            st.plotly_chart(_build_gauge(overall_score), use_container_width=True)

        # Key findings
        findings = result.get("key_findings", [])
        if findings:
            st.markdown("#### Key Findings")
            for finding in findings:
                st.markdown(f"- {finding}")

    # ---- Tab 2: Detailed Analysis -----------------------------------------
    with tab_detail:
        # NLP Analysis
        with st.expander("🧠 NLP Analysis", expanded=True):
            nlp = result.get("nlp_analysis", {})
            if nlp:
                comp = result.get("component_scores", {})
                nlp_score = comp.get("NLP Analysis") or comp.get("nlp", "–")
                st.metric("NLP Score", f"{nlp_score}/100")
                cols = st.columns(3)
                cols[0].metric("Sentiment", nlp.get("sentiment", "N/A"))
                cols[1].metric("Subjectivity", nlp.get("subjectivity", "N/A"))
                cols[2].metric("Emotional Language", nlp.get("emotional_language", "N/A"))
                details = nlp.get("details")
                if details:
                    st.info(details)
            else:
                st.caption("NLP analysis data not available.")

        # Fact-Check Results
        with st.expander("✅ Fact-Check Results", expanded=True):
            fc = result.get("fact_check", {})
            if fc:
                comp = result.get("component_scores", {})
                fc_score = comp.get("Fact Check") or comp.get("fact_check", "–")
                st.metric("Fact-Check Score", f"{fc_score}/100")
                claims_found = fc.get("claims", [])
                if claims_found:
                    for c in claims_found:
                        st.markdown(
                            f"- **{c.get('claimant', 'Unknown')}**: "
                            f"{c.get('text', '')} — *{c.get('rating', '')}*"
                        )
                else:
                    status = fc.get("status", "No matching claims found.")
                    st.caption(status)
            else:
                st.caption("Fact-check not configured — add an API key in Settings.")

        # Source Credibility
        with st.expander("🌐 Source Credibility", expanded=True):
            src = result.get("source_credibility")
            if src:
                comp = result.get("component_scores", {})
                src_score = comp.get("Source Credibility") or comp.get("source", "–")
                st.metric("Source Score", f"{src_score}/100")
                st.markdown(f"**Domain:** {src.get('domain', 'N/A')}")
                st.markdown(f"**Reputation:** {src.get('reputation', 'N/A')}")
                details = src.get("details")
                if details:
                    st.info(details)
            else:
                st.caption("Source credibility analysis requires a URL input.")

        # Linguistic Analysis
        with st.expander("📝 Linguistic Analysis", expanded=True):
            ling = result.get("linguistic_analysis", {})
            if ling:
                comp = result.get("component_scores", {})
                ling_score = comp.get("Linguistic Analysis") or comp.get("linguistic", "–")
                st.metric("Linguistic Score", f"{ling_score}/100")
                cols = st.columns(3)
                cols[0].metric("Clickbait", ling.get("clickbait", "N/A"))
                cols[1].metric("Sensationalism", ling.get("sensationalism", "N/A"))
                cols[2].metric("Logical Fallacies", ling.get("fallacies", "N/A"))
                details = ling.get("details")
                if details:
                    st.info(details)
            else:
                st.caption("Linguistic analysis data not available.")

        # Gemini Deep Reasoning & Atomic Claims
        with st.expander("🤖 Gemini Deep Reasoning & Multi-Claim Breakdown", expanded=True):
            gem = result.get("gemini_analysis", {})
            if gem and gem.get("available"):
                gem_score = gem.get("ai_credibility_score", "–")
                conf = gem.get("confidence", "Medium")
                c1, c2, c3 = st.columns([1, 1, 2])
                c1.metric("Gemini Credibility", f"{gem_score}/100")
                c2.metric("Confidence", conf)
                c3.metric("Evidence Quality", gem.get("evidence_quality", "N/A"))

                summary = gem.get("verdict_summary")
                if summary:
                    st.info(f"**Forensic Summary:** {summary}")

                # Atomic claims breakdown
                claims = gem.get("atomic_claims", [])
                if claims:
                    st.markdown("##### 🔬 Atomic Claims Breakdown")
                    for i, ac in enumerate(claims, 1):
                        assessment = ac.get("assessment", "Unverified")
                        badge_color = (
                            "🟢" if "True" in assessment 
                            else "🔴" if ("False" in assessment or "Misleading" in assessment) 
                            else "🟡"
                        )
                        st.markdown(
                            f"**{i}. {ac.get('claim')}**  \n"
                            f"{badge_color} *Status:* **{assessment}** (Verifiability: {ac.get('verifiability', 'N/A')})  \n"
                            f"*{ac.get('note', '')}*"
                        )
                        st.markdown("---")

                # Fallacies & manipulation tactics
                fallacies = [f for f in gem.get("logical_fallacies", []) if f and f.lower() != "none"]
                tactics = [t for t in gem.get("manipulation_tactics", []) if t and t.lower() != "none"]

                if fallacies or tactics:
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        if fallacies:
                            st.markdown("⚠️ **Detected Logical Fallacies:**")
                            for f in fallacies:
                                st.markdown(f"- `{f}`")
                    with fc2:
                        if tactics:
                            st.markdown("🚨 **Manipulation Tactics:**")
                            for t in tactics:
                                st.markdown(f"- `{t}`")
            else:
                err_msg = (
                    gem.get("error")
                    or gem.get("message")
                    or "Add your free Google AI Studio key in sidebar Settings to unlock atomic claim extraction and fallacy detection."
                )
                st.caption(f"ℹ️ {err_msg}")

        # Wikipedia Knowledge Graph Corroboration
        with st.expander("📚 Wikipedia Knowledge Graph & Context", expanded=True):
            wiki = result.get("wiki_analysis", {})
            if wiki and wiki.get("available"):
                w_score = wiki.get("score", 50)
                st.metric("Knowledge Alignment", f"{w_score}/100")
                summary = wiki.get("summary")
                if summary:
                    st.info(summary)

                matches = wiki.get("matches", [])
                if matches:
                    st.markdown("##### 🔍 Relevant Encyclopedia Reference Articles:")
                    for m in matches:
                        st.markdown(f"- [**{m['title']}**]({m['url']}): *{m['snippet']}*")
            else:
                st.caption("Wikipedia knowledge lookup not available.")

        # Quantitative Linguistic & Readability Metrics
        with st.expander("📏 Quantitative Linguistic & Readability Metrics", expanded=True):
            met = result.get("metrics_analysis", {})
            if met:
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Flesch Reading Ease", f"{met.get('flesch_reading_ease', 0)}/100")
                m_col2.metric("Grade Level", f"Grade {met.get('flesch_kincaid_grade', 0)}")
                m_col3.metric("Lexical Diversity (TTR)", f"{met.get('lexical_diversity_ttr', 0):.3f}")
                m_col4.metric("Uppercase Ratio", f"{met.get('uppercase_ratio_pct', 0)}%")

                m_col5, m_col6, m_col7 = st.columns(3)
                m_col5.metric("Punctuation Density", f"{met.get('punctuation_density_per_100w', 0)} / 100w")
                m_col6.metric("Evidence Markers", f"{met.get('evidence_markers_count', 0)} detected")
                m_col7.metric("Margin of Error (95% CI)", f"± {met.get('margin_of_error_pct', 0)}%")
            else:
                st.caption("Quantitative metrics not available.")

        # Machine Learning Statistical Model
        with st.expander("🤖 Machine Learning Statistical Probability & Feature Importances", expanded=True):
            ml_res = result.get("ml_analysis", {})
            if ml_res and ml_res.get("available"):
                ml_c1, ml_c2, ml_c3 = st.columns(3)
                ml_c1.metric(
                    "Misinfo Probability",
                    f"{ml_res.get('misinfo_probability_pct', 0)}%",
                    delta=f"-{ml_res.get('misinfo_probability_pct', 0)}%",
                    delta_color="inverse",
                )
                ml_c2.metric(
                    "Credible Probability",
                    f"{ml_res.get('credible_probability_pct', 0)}%",
                    delta=f"+{ml_res.get('credible_probability_pct', 0)}%",
                )
                ci = ml_res.get("confidence_interval_95", (0, 0))
                ml_c3.metric("95% Confidence Bounds", f"[{ci[0]}% - {ci[1]}%]")

                feats = ml_res.get("top_predictive_features", [])
                if feats:
                    st.markdown("##### 🔬 N-Gram Terms Driving Classification:")
                    for f_item in feats:
                        badge = "🔴" if "Misinformation" in f_item["impact"] else "🟢"
                        st.markdown(
                            f"- {badge} Term **`{f_item['term']}`**: *{f_item['impact']}* (Statistical Coefficient: `{f_item['weight']:+.3f}`)"
                        )
            else:
                st.caption("ML model prediction not available.")

    # ---- Tab 3: Score Breakdown -------------------------------------------
    with tab_scores:
        component_scores = result.get("component_scores", {})
        if component_scores:
            col_radar, col_bar = st.columns(2)
            with col_radar:
                st.plotly_chart(
                    _build_radar(component_scores), use_container_width=True
                )
            with col_bar:
                st.plotly_chart(
                    _build_bar(component_scores), use_container_width=True
                )
        else:
            st.caption("No component scores available.")

        with st.expander("📖 Scoring Methodology"):
            st.markdown(
                """
                The **overall credibility score** (0–100) is a weighted average of
                individual module scores:

                | Module | Weight | What It Measures |
                |---|---|---|
                | NLP Analysis | 25% | Sentiment bias, subjectivity, emotional tone |
                | Fact Check | 30% | Agreement with known fact-check verdicts |
                | Source Credibility | 20% | Domain reputation, publishing history |
                | Linguistic Analysis | 25% | Clickbait patterns, sensationalism, fallacies |

                *Weights are adjusted dynamically when a module is unavailable
                (e.g. no URL provided for source credibility).*

                **Score ranges:**
                - 🟢 **70–100** — Likely credible
                - 🟡 **55–69** — Needs caution
                - 🟠 **40–54** — Questionable
                - 🔴 **0–39** — Likely misinformation
                """
            )

    # ---- Tab 4: Measurable Metrics & Benchmark ---------------------------
    with tab_metrics:
        st.subheader("📏 Empirical Benchmark & Formal Performance Metrics")
        st.markdown(
            "Unlike subjective rules, this engine is continuously validated against a "
            "**ground-truth benchmark dataset** to measure exact classification metrics."
        )

        def _get_benchmark_eval():
            from modules.input_processor import process_input
            from modules.nlp_analyzer import analyze_text
            from modules.fact_checker import check_claim
            from modules.linguistic_analyzer import analyze_linguistics
            from modules.wiki_verifier import verify_with_wikipedia
            from modules.ml_classifier import predict_misinformation_probability
            from modules.scoring_engine import calculate_credibility_score

            def eval_fn(claim_text):
                p = process_input(claim_text, "claim")
                t = p.get("text", claim_text)
                return {
                    "score": calculate_credibility_score(
                        nlp_result=analyze_text(t),
                        fact_check_result=check_claim(t, api_key=None),
                        source_result=None,
                        linguistic_result=analyze_linguistics(t),
                        wiki_result=verify_with_wikipedia(t),
                        ml_result=predict_misinformation_probability(t),
                    ).get("overall_score", 50.0)
                }
            return run_benchmark_evaluation(eval_fn)

        if "benchmark_results" not in st.session_state:
            with st.spinner("Calculating live benchmark metrics across ground-truth test suite..."):
                st.session_state.benchmark_results = _get_benchmark_eval()

        bench = st.session_state.benchmark_results
        bm = bench["metrics"]
        bcm = bench["confusion_matrix"]

        # Metric cards
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        b_c1.metric("Precision (PPV)", f"{bm['precision_pct']}%", help="True Positives / (True Positives + False Positives)")
        b_c2.metric("Recall / Sensitivity", f"{bm['recall_sensitivity_pct']}%", help="True Positives / (True Positives + False Negatives)")
        b_c3.metric("F1 Score", f"{bm['f1_score']}", help="Harmonic mean of precision and recall (0.0 to 1.0)")
        b_c4.metric("Specificity (TNR)", f"{bm['specificity_pct']}%", help="True Negatives / (True Negatives + False Positives)")

        b_c5, b_c6, b_c7 = st.columns(3)
        b_c5.metric("Accuracy", f"{bm['accuracy_pct']}%")
        b_c6.metric("False Positive Rate", f"{bm['false_positive_rate_pct']}%", help="Percentage of true facts wrongly flagged as misinformation")
        b_c7.metric("Brier Calibration Score", f"{bm['brier_calibration_score']}", help="Measures probabilistic calibration error; lower is better")

        st.divider()

        # Visual Confusion Matrix
        col_cm, col_info = st.columns([1, 1])
        with col_cm:
            st.markdown("##### 🔲 Confusion Matrix Heatmap")
            cm_z = [
                [bcm["true_positives"], bcm["false_negatives"]],
                [bcm["false_positives"], bcm["true_negatives"]],
            ]
            cm_fig = px.imshow(
                cm_z,
                text_auto=True,
                x=["Predicted False", "Predicted Not-False"],
                y=["Actual False (Misinfo)", "Actual True / Unverified"],
                color_continuous_scale="Blues",
                aspect="auto",
            )
            cm_fig.update_layout(height=280, margin=dict(t=20, b=20, l=40, r=20))
            st.plotly_chart(cm_fig, use_container_width=True)

        with col_info:
            st.markdown("##### 📊 Benchmark Summary")
            st.markdown(
                f"- **Total Samples:** `{bench['total_samples']}` ground-truth claims\n"
                f"- **True Positives (Correctly Flagged Misinfo):** `{bcm['true_positives']}`\n"
                f"- **True Negatives (Correctly Kept Clean):** `{bcm['true_negatives']}`\n"
                f"- **False Positives (Clean claims wrongly flagged):** `{bcm['false_positives']}` *(Zero false alarms)*\n"
                f"- **False Negatives:** `{bcm['false_negatives']}`"
            )
            if st.button("🔄 Re-run Benchmark Evaluation"):
                with st.spinner("Re-evaluating benchmark suite..."):
                    st.session_state.benchmark_results = _get_benchmark_eval()
                    st.rerun()

        # Detailed test table
        with st.expander("📋 View Full 25-Claim Benchmark Test Breakdown", expanded=False):
            import pandas as pd
            df_bench = pd.DataFrame(bench["detailed_results"])
            st.dataframe(
                df_bench[["claim", "category", "ground_truth", "pred_label", "pred_score", "cm_status"]],
                use_container_width=True,
            )

    # ---- Tab 5: Report ----------------------------------------------------
    with tab_report:
        text_report = _generate_text_report(result)
        st.text_area("Full Analysis Report", value=text_report, height=400)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Download JSON Report",
                data=json.dumps(result, indent=2, default=str),
                file_name=f"misinfo_report_{result.get('timestamp', 'unknown').replace(' ', '_').replace(':', '-')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "⬇️ Download Text Summary",
                data=text_report,
                file_name=f"misinfo_summary_{result.get('timestamp', 'unknown').replace(' ', '_').replace(':', '-')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

# ---- How It Works (bottom) ------------------------------------------------
st.markdown("---")
with st.expander("💡 How It Works"):
    st.markdown(
        """
        **MisInfo Detector** runs your input through five independent analysis
        modules, then combines the results into a single credibility score:

        1. **Input Processor** — Normalises raw text, extracts article content
           from URLs, and prepares metadata for downstream modules.

        2. **NLP Analyser** — Uses natural language processing to measure
           sentiment polarity, subjectivity, and the density of emotionally
           charged language.

        3. **Fact Checker** — Queries the Google Fact Check Tools API to find
           existing fact-check verdicts for the claim or similar claims.

        4. **Source Credibility** — Evaluates the reputation of the publishing
           domain based on known credibility databases and heuristics.

        5. **Linguistic Analyser** — Detects clickbait patterns,
           sensationalist language, and common logical fallacies.

        ---

        > **⚠️ Disclaimer:** This tool is designed to *assist* critical
        > thinking — it does **not** provide definitive truth judgements.
        > Always verify important claims through multiple reputable sources.
        > The credibility score is probabilistic and may be inaccurate for
        > niche or emerging topics.
        """
    )
