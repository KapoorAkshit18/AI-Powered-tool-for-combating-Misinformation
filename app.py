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
from modules.scoring_engine import calculate_credibility_score
from knowledge_base.db import init_db, save_analysis, get_history
from config import Config

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
    st.session_state.fact_check_api_key = ""

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
    with st.expander("⚙️ Settings", expanded=False):
        api_key_input = st.text_input(
            "Google Fact Check API Key",
            value=st.session_state.fact_check_api_key,
            type="password",
            help="Enter your Google Fact Check Tools API key to enable fact-check lookups.",
        )
        if api_key_input != st.session_state.fact_check_api_key:
            st.session_state.fact_check_api_key = api_key_input
            st.success("API key saved for this session.")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    """Return a CSS class suffix based on the credibility score."""
    if score >= 70:
        return "green"
    if score >= 40:
        return "yellow" if score >= 55 else "orange"
    return "red"


def _verdict_label(score: float) -> str:
    """Return a human-readable verdict string."""
    if score >= 70:
        return "✅ Likely Credible"
    if score >= 55:
        return "⚠️ Needs Caution"
    if score >= 40:
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
    for section_key in ("nlp_analysis", "fact_check", "source_credibility", "linguistic_analysis"):
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

                # 6. Aggregate score
                score_result = calculate_credibility_score(
                    nlp_result=nlp_result,
                    fact_check_result=fact_result,
                    source_result=source_result,
                    linguistic_result=ling_result,
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

    tab_verdict, tab_detail, tab_scores, tab_report = st.tabs(
        ["📊 Verdict", "🔬 Detailed Analysis", "📈 Score Breakdown", "📋 Report"]
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

    # ---- Tab 4: Report ----------------------------------------------------
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
