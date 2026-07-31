"""
TAVRA Mission Control Dashboard (Streamlit).

This is an UI/UX and stability upgrade of the original mission_control.py.
The ML pipeline, data assumptions, and decision logic are UNCHANGED —
only the presentation layer, the 3D simulation embedding, and a few
resilience fixes were touched. See "Fixes carried over from the original
build" below for the pre-existing corrections this file still relies on.

Fixes carried over from the original build:
  - No `mission_id` column exists in the data; the sidebar selects an
    `event_batch` (rows grouped by index // BATCH_SIZE) instead.
  - `risk` is log10(Pc), roughly -30..-3 — the ">0.8 / >0.5" decision
    thresholds are applied to the model's predicted *probability*,
    not to the raw `risk` column (see src/scripts/decision_engine.py).
  - Points at data/engineered_data.csv, the file the actual pipeline
    produces (the *_sample.csv file doesn't include risk_label).

New in this revision:
  - Deprecated `st.components.v1.html` usage replaced with a defensive
    helper (`embed_html`) that works across Streamlit's supported import
    paths, so the 3D simulation renders inline instead of erroring out
    or asking the user to open a local HTML file.
  - Deep-space / glassmorphism theme (assets/theme.css), TAVRA branding
    in the header and sidebar, animated metric cards.
  - 3D Simulation tab is full width with an in-scene control panel
    (Pause/Resume, Reset Camera, Orbits, Debris, Labels, Follow, 24h
    Predict) built directly into earth_simulation.html.
  - Dark Plotly theme across all charts.
  - Model Info tab now surfaces the real validation metrics, feature
    importance (SHAP fallback), and pipeline description that already
    existed in outputs/ but weren't shown in the UI.
"""
import base64
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src" / "scripts"))
from decision_engine import decide  # noqa: E402

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = FRONTEND_DIR / "assets"

MODEL_PATH = MODELS_DIR / "lstm_model.keras"
DATA_PATH = DATA_DIR / "engineered_data.csv"
MODEL_COMPARISON_PATH = OUTPUTS_DIR / "model_comparison.csv"
BASELINE_METRICS_PATH = OUTPUTS_DIR / "baseline_validation_metrics.json"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "shap" / "feature_importance.csv"
SIM_HTML_PATH = FRONTEND_DIR / "earth_simulation.html"
LOGO_PATH = ASSETS_DIR / "tavra_logo.png"
THEME_CSS_PATH = ASSETS_DIR / "theme.css"
BATCH_SIZE = 500

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#dfeeff", family="Segoe UI, sans-serif"),
)

st.set_page_config(
    page_title="TAVRA | Space Debris Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_data
def load_css(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


@st.cache_data
def load_image_b64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def embed_html(html: str, height: int = 600, scrolling: bool = False) -> None:
    """Render raw HTML/JS inline inside the Streamlit app.

    Streamlit's HTML-embedding API has moved twice: `st.components.v1.html`
    is what the original build called, but that path can raise
    `module 'streamlit.components' has no attribute 'v1'` depending on the
    installed version, and is itself now deprecated in favor of
    `st.iframe` (which auto-detects raw HTML strings). Try the modern API
    first and fall back defensively so this keeps working on older
    Streamlit installs too — never fall back to telling the user to open
    a file manually.
    """
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
        return
    try:
        import streamlit.components.v1 as components
    except (ImportError, AttributeError):
        from streamlit import components  # type: ignore
    components.html(html, height=height, scrolling=scrolling)


def apply_theme() -> None:
    css = load_css(THEME_CSS_PATH)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header() -> None:
    logo_b64 = load_image_b64(LOGO_PATH)
    logo_html = (
        f'<img class="tv-logo" src="data:image/png;base64,{logo_b64}"/>'
        if logo_b64
        else ""
    )
    st.markdown(
        f"""
        <div class="tv-header">
            {logo_html}
            <div class="tv-title-block">
                <div class="tv-title">TAVRA</div>
                <div class="tv-tagline">Turn Complexity Into Clarity — Space Debris &amp; Satellite Collision Risk Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "", risk_class: str = "") -> str:
    return f"""
    <div class="tv-metric {risk_class}">
        <div class="tv-metric-label">{label}</div>
        <div class="tv-metric-value">{value}</div>
        <div class="tv-metric-sub">{sub}</div>
    </div>
    """


def render_sidebar(logo_b64: str) -> None:
    with st.sidebar:
        if logo_b64:
            st.markdown(
                f'<div class="tv-sidebar-logo"><img src="data:image/png;base64,{logo_b64}"/></div>',
                unsafe_allow_html=True,
            )

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.markdown(
            f"""
            <div class="tv-sb-block">
                <div class="tv-sb-title">Mission Status</div>
                <div class="tv-sb-row"><span class="lbl">Status</span>
                    <span class="val"><span class="tv-status-dot tv-status-ok"></span>OPERATIONAL</span></div>
                <div class="tv-sb-row"><span class="lbl">UTC Time</span><span class="val">{now_utc}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="tv-sb-title" style="margin-top:4px;">Navigation</div>', unsafe_allow_html=True)
        st.caption("Risk Analysis · Predictions · 3D Simulation · Model Info")

        st.markdown('<div class="tv-sb-block">', unsafe_allow_html=True)
        st.markdown('<div class="tv-sb-title">Configuration</div>', unsafe_allow_html=True)


def render_system_status() -> None:
    cpu_txt, gpu_txt = "N/A", "N/A"
    try:
        import psutil  # optional dependency; degrade gracefully if absent

        cpu_txt = f"{psutil.cpu_percent(interval=0.05):.0f}%"
        mem = psutil.virtual_memory()
        gpu_txt = f"{mem.percent:.0f}% MEM"
    except ImportError:
        pass

    gpus = tf.config.list_physical_devices("GPU")
    accel = f"{len(gpus)} GPU(s)" if gpus else "CPU only"

    st.sidebar.markdown(
        f"""
        <div class="tv-sb-block">
            <div class="tv-sb-title">System Status</div>
            <div class="tv-sb-row"><span class="lbl">Compute</span><span class="val">{accel}</span></div>
            <div class="tv-sb-row"><span class="lbl">CPU Load</span><span class="val">{cpu_txt}</span></div>
            <div class="tv-sb-row"><span class="lbl">Memory</span><span class="val">{gpu_txt}</span></div>
            <div class="tv-sb-row"><span class="lbl">Model</span>
                <span class="val"><span class="tv-status-dot tv-status-ok"></span>LOADED</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Cached data / model loaders (unchanged logic from the original build)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["event_batch"] = df.index // BATCH_SIZE
    return df


@st.cache_data
def load_sim_html() -> str:
    return SIM_HTML_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------
apply_theme()
render_header()

if not MODEL_PATH.exists():
    st.error(f"❌ Model not found:\n{MODEL_PATH}")
    st.stop()
if not DATA_PATH.exists():
    st.error(
        f"❌ Dataset not found:\n{DATA_PATH}\n\nRun the data pipeline scripts first "
        f"(preprocess.py -> feature_engineering.py)."
    )
    st.stop()

model = load_model()
df = load_data()
logo_b64 = load_image_b64(LOGO_PATH)

render_sidebar(logo_b64)
selected_batch = st.sidebar.selectbox("Event Batch", sorted(df["event_batch"].unique()))
risk_threshold = st.sidebar.slider("Risk Probability Threshold", 0.0, 1.0, 0.5)
st.sidebar.markdown("</div>", unsafe_allow_html=True)
render_system_status()

batch_data = df[df["event_batch"] == selected_batch]

# ---- Top metric row -------------------------------------------------------
high_risk = int((batch_data["risk"] > -6).sum())
total_events = len(batch_data)
high_risk_pct = (high_risk / total_events * 100) if total_events else 0.0
risk_class = "risk-critical" if high_risk_pct > 15 else ("risk-high" if high_risk_pct > 5 else "risk-low")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card("Total Events", f"{total_events:,}", f"Batch #{selected_batch}"), unsafe_allow_html=True)
with c2:
    st.markdown(
        metric_card("High Risk Events", f"{high_risk:,}", f"Pc > 1e-6 · {high_risk_pct:.1f}% of batch", risk_class),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        metric_card("Avg log₁₀(Risk)", f"{batch_data['risk'].mean():.2f}", "Lower = safer"),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        metric_card("Avg Miss Distance", f"{batch_data['miss_distance'].mean():.2f}", "Scaled units"),
        unsafe_allow_html=True,
    )

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Analysis", "🎯 Predictions", "🛰️ 3D Simulation", "🧠 Model Info"])

# ---- Tab 1: Risk Analysis --------------------------------------------------
with tab1:
    st.markdown('<div class="tv-section-title">Risk Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(
        batch_data, x="risk", nbins=50, title="log₁₀(Collision Probability) Distribution",
        color_discrete_sequence=["#31e7ff"],
    )
    fig.update_layout(**PLOTLY_DARK)
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="tv-section-title">Miss Distance vs Risk</div>', unsafe_allow_html=True)
    fig2 = px.scatter(
        batch_data, x="miss_distance", y="risk", title="Miss Distance vs Risk Score",
        color="risk", color_continuous_scale=["#39e58a", "#ffcf5c", "#ff3b4e"],
    )
    fig2.update_layout(**PLOTLY_DARK)
    st.plotly_chart(fig2, width='stretch')

# ---- Tab 2: Predictions ----------------------------------------------------
with tab2:
    st.markdown('<div class="tv-section-title">Make a Prediction</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        sample_idx = st.slider("Select Sample Index", 0, max(len(batch_data) - 1, 0), 0)
        feature_cols = [c for c in batch_data.columns if c not in ("risk", "risk_label", "event_batch")]
        sample = batch_data.iloc[sample_idx][feature_cols].values.astype("float32")

        reshaped_sample = sample.reshape((1, 1, len(sample)))
        probability = float(model.predict(reshaped_sample, verbose=0)[0][0])
        decision = decide(probability)

        st.markdown(metric_card("Predicted Risk Probability", f"{probability:.4f}"), unsafe_allow_html=True)
        st.write("")
        if decision.risk_level == "CRITICAL":
            st.markdown(
                f'<div class="tv-alert">🔴 {decision.risk_level} — {decision.action}</div>',
                unsafe_allow_html=True,
            )
        elif decision.risk_level == "HIGH":
            st.markdown(
                f'<span class="tv-badge warn">🟡 {decision.risk_level}</span> &nbsp; {decision.action}',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="tv-badge ok">🟢 {decision.risk_level}</span> &nbsp; {decision.action}',
                unsafe_allow_html=True,
            )

    with col2:
        st.write("Sample Features:")
        sample_df = pd.DataFrame({"Feature": feature_cols[:10], "Value": sample[:10]})
        st.dataframe(sample_df, width='stretch')

# ---- Tab 3: 3D Simulation --------------------------------------------------
with tab3:
    st.markdown('<div class="tv-section-title">Mission Control — Live Orbital Simulation</div>', unsafe_allow_html=True)
    st.caption(
        "Earth, atmosphere, cloud layer, Moon, satellites, debris, orbital paths, predicted conjunction "
        "point and risk cone — rendered inline via Three.js. Use the on-screen panel to control the scene."
    )
    embed_html(load_sim_html(), height=760, scrolling=False)

# ---- Tab 4: Model Info -----------------------------------------------------
with tab4:
    st.markdown('<div class="tv-section-title">Pipeline &amp; Model</div>', unsafe_allow_html=True)
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.write("**Model Type:** LSTM (Long Short-Term Memory)")
        st.write(f"**Feature Count:** {len(feature_cols) if 'feature_cols' in dir() else len(df.columns) - 3}")
        st.write("**Output:** Binary classification — P(risk_label = 1), i.e. P(Pc > 1e-6)")
        st.write(f"**Training Rows Available:** {len(df):,}")
    with mcol2:
        st.write("**Pipeline:** preprocess → feature_engineering → train_models → decision_engine")
        st.write("**Sequence Shape:** (batch, 1 timestep, n_features)")
        st.write("**Decision Thresholds:** LOW ≤ 0.5 · HIGH 0.5–0.8 · CRITICAL > 0.8")

    st.markdown('<div class="tv-section-title" style="margin-top:18px;">Validation Metrics</div>', unsafe_allow_html=True)
    if BASELINE_METRICS_PATH.exists():
        import json

        metrics = json.loads(BASELINE_METRICS_PATH.read_text())
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(metric_card("Accuracy", f"{metrics.get('Accuracy', 0):.3f}"), unsafe_allow_html=True)
        with mc2:
            st.markdown(metric_card("F1-Score", f"{metrics.get('F1-Score', 0):.3f}"), unsafe_allow_html=True)
        with mc3:
            st.markdown(metric_card("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.3f}"), unsafe_allow_html=True)
        with mc4:
            st.markdown(
                metric_card(
                    "Test Positive Rate",
                    f"{metrics.get('positive_rate_test', 0) * 100:.1f}%",
                    f"n_train={metrics.get('n_train', 'N/A')} · n_test={metrics.get('n_test', 'N/A')}",
                ),
                unsafe_allow_html=True,
            )
        st.caption(
            "Metrics from the scikit-learn baseline validation run "
            "(outputs/baseline_validation_metrics.json). This is the run that was actually "
            "executed end-to-end; see train_models.py's docstring re: the LSTM comparison run."
        )
    else:
        st.info("Baseline validation metrics not found. Run train_baseline.py to generate them.")

    if MODEL_COMPARISON_PATH.exists():
        st.markdown('<div class="tv-section-title" style="margin-top:18px;">Model Comparison</div>', unsafe_allow_html=True)
        comparison = pd.read_csv(MODEL_COMPARISON_PATH, index_col=0)
        st.dataframe(comparison, width='stretch')

    st.markdown('<div class="tv-section-title" style="margin-top:18px;">Feature Importance</div>', unsafe_allow_html=True)
    if FEATURE_IMPORTANCE_PATH.exists():
        fi = pd.read_csv(FEATURE_IMPORTANCE_PATH, index_col=0)
        fi.columns = ["importance"]
        fi = fi.sort_values("importance", ascending=False).head(15).sort_values("importance")
        fig3 = go.Figure(
            go.Bar(
                x=fi["importance"], y=fi.index, orientation="h",
                marker=dict(color=fi["importance"], colorscale=[[0, "#3aa0ff"], [1, "#ffcf5c"]]),
            )
        )
        fig3.update_layout(title="Top 15 Features (SHAP fallback importance)", **PLOTLY_DARK)
        st.plotly_chart(fig3, width='stretch')
    else:
        st.info("Feature importance data not available.")

st.divider()
st.markdown(
    '<div class="tv-footer">TAVRA MISSION CONTROL · Space Debris Collision Prediction System · '
    'Powered by LSTM + SHAP Explainability</div>',
    unsafe_allow_html=True,
)
