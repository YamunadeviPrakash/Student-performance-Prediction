"""
Student Performance Prediction Dashboard
=========================================
A Streamlit dashboard using Linear Regression to predict student final grades (G3).
Dataset: UCI Student Performance Dataset 
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Dark sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2235;
}
section[data-testid="stSidebar"] * {
    color: #c9d1e0 !important;
}

/* ── Main background ── */
.main .block-container {
    background: #13151f;
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1526 60%, #141930 100%);
    border: 1px solid #2a3158;
    border-radius: 16px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 2.4rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,133,255,0.12) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #e8eaf6;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7b88b0;
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,133,255,0.15);
    color: #7ea3ff;
    border: 1px solid rgba(99,133,255,0.3);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 0.9rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Section headings ── */
.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #c9d4f0;
    font-weight: 600;
    border-left: 3px solid #5272e8;
    padding-left: 0.8rem;
    margin: 2.2rem 0 1.2rem 0;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin-bottom: 1.6rem;
}
.metric-card {
    background: #1a1f35;
    border: 1px solid #252c48;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    color: #7b88b0;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 500;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: #e8eaf6;
}
.metric-accent { color: #7ea3ff; }
.metric-green  { color: #56d99e; }
.metric-yellow { color: #f5c842; }
.metric-red    { color: #ff6b7a; }

/* ── Prediction result box ── */
.pred-box {
    background: linear-gradient(135deg, #1a2340, #1a1f35);
    border: 1px solid #2a3158;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
}
.pred-grade {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    font-weight: 700;
    color: #7ea3ff;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.pred-label {
    font-size: 0.85rem;
    color: #7b88b0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.result-pass {
    background: rgba(86,217,158,0.1);
    border: 1px solid rgba(86,217,158,0.35);
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    display: inline-block;
    color: #56d99e;
    font-weight: 600;
    font-size: 1.1rem;
    margin-top: 0.8rem;
}
.result-fail {
    background: rgba(255,107,122,0.1);
    border: 1px solid rgba(255,107,122,0.35);
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    display: inline-block;
    color: #ff6b7a;
    font-weight: 600;
    font-size: 1.1rem;
    margin-top: 0.8rem;
}

/* ── Chart container ── */
.chart-box {
    background: #1a1f35;
    border: 1px solid #252c48;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
}

/* ── Streamlit overrides ── */
[data-testid="stMarkdownContainer"] p { color: #b0bcd8; }
.stButton>button {
    background: linear-gradient(135deg, #4763d9, #5272e8);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 1rem;
    padding: 0.65rem 2.2rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.88; }
.stSlider label, .stSlider [data-testid="stWidgetLabel"] { color: #b0bcd8 !important; }
div[data-testid="stMetric"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

DARK_BG   = "#1a1f35"
GRID_CLR  = "#252c48"
TEXT_CLR  = "#b0bcd8"
ACCENT    = "#5272e8"
ACCENT2   = "#56d99e"
WARN_CLR  = "#f5c842"
DANGER    = "#ff6b7a"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    DARK_BG,
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
    "grid.color":        GRID_CLR,
    "text.color":        TEXT_CLR,
    "axes.titlecolor":   TEXT_CLR,
    "figure.dpi":        130,
})

@st.cache_data
def load_data():
    """Load and return the UCI Student Performance dataset."""
    df = pd.read_csv("dataset.csv", sep=";")
    return df

@st.cache_resource
def train_model(df):
    """
    Train a Linear Regression model on the selected features.
    Returns: model, X_test, y_test, y_pred, split metrics.
    """
    features = ["studytime", "failures", "absences", "G1", "G2"]
    target   = "G3"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

def to_pass_fail(grades):
    """Convert numeric grades to binary Pass (>=10) / Fail (<10)."""
    return (np.array(grades) >= 10).astype(int)

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️  **dataset.csv** not found. Place the UCI Student Performance CSV "
             "(semicolon-separated) in the same directory as app.py and re-run.")
    st.stop()

model, X_test, y_test, y_pred = train_model(df)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

y_test_cls = to_pass_fail(y_test)
y_pred_cls = to_pass_fail(y_pred)

acc  = accuracy_score (y_test_cls, y_pred_cls)
prec = precision_score(y_test_cls, y_pred_cls, zero_division=0)
rec  = recall_score   (y_test_cls, y_pred_cls, zero_division=0)
f1   = f1_score       (y_test_cls, y_pred_cls, zero_division=0)
conf_cm = confusion_matrix(y_test_cls, y_pred_cls)

with st.sidebar:
    st.markdown("## 📊 Dataset Info")
    st.markdown(f"**Rows:** {len(df):,}")
    st.markdown(f"**Columns:** {df.shape[1]}")
    st.markdown(f"**Training samples:** {int(len(df)*0.8)}")
    st.markdown(f"**Testing samples:** {int(len(df)*0.2)}")
    st.markdown("---")
    st.markdown("**Features used**")
    for f in ["studytime", "failures", "absences", "G1", "G2"]:
        st.markdown(f"• `{f}`")
    st.markdown("**Target:** `G3` (final grade)")
    st.markdown("---")
    st.markdown(f"**R² (Confidence):** `{r2:.3f}`")
    st.markdown(f"**RMSE:** `{rmse:.3f}`")
    st.caption("Pass threshold: G3 ≥ 10")

st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">🎓 ML Dashboard</div>
  <h1 class="hero-title">Student Performance Prediction Dashboard</h1>
  <p class="hero-sub">
    Linear Regression · UCI Dataset · Grades G1 → G2 → G3 · Pass / Fail Classification
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-heading">Exploratory Data Analysis</div>',
            unsafe_allow_html=True)

eda_col1, eda_col2, eda_col3 = st.columns([2, 1.3, 1.3])

with eda_col1:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    corr_cols = ["studytime", "failures", "absences", "G1", "G2", "G3"]
    corr_data = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_data, mask=mask, annot=True, fmt=".2f",
        cmap=cmap, linewidths=0.5, linecolor=GRID_CLR,
        annot_kws={"size": 8}, ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Correlation Heatmap", pad=10, fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with eda_col2:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 4.2))
    ax.scatter(df["G2"], df["G3"],
               color=ACCENT, alpha=0.55, edgecolors="none", s=28)
    m, b = np.polyfit(df["G2"], df["G3"], 1)
    xs = np.linspace(df["G2"].min(), df["G2"].max(), 100)
    ax.plot(xs, m*xs + b, color=ACCENT2, linewidth=1.6, linestyle="--")
    ax.set_xlabel("G2 (Period 2 Grade)", fontsize=9)
    ax.set_ylabel("G3 (Final Grade)", fontsize=9)
    ax.set_title("G2 vs G3", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with eda_col3:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 4.2))
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(df))
    ax.scatter(df["studytime"] + jitter, df["G3"],
               color=WARN_CLR, alpha=0.55, edgecolors="none", s=28)
    m, b = np.polyfit(df["studytime"], df["G3"], 1)
    xs = np.linspace(df["studytime"].min(), df["studytime"].max(), 100)
    ax.plot(xs, m*xs + b, color=ACCENT2, linewidth=1.6, linestyle="--")
    ax.set_xlabel("Study Time", fontsize=9)
    ax.set_ylabel("G3 (Final Grade)", fontsize=9)
    ax.set_title("Studytime vs G3", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-heading">Regression Evaluation Metrics</div>',
            unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">MSE</div>
    <div class="metric-value metric-yellow">{mse:.3f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">RMSE</div>
    <div class="metric-value metric-yellow">{rmse:.3f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">R² Score</div>
    <div class="metric-value metric-accent">{r2:.3f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-heading">Classification Metrics (Pass / Fail)</div>',
            unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Accuracy</div>
    <div class="metric-value metric-green">{acc*100:.1f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Precision</div>
    <div class="metric-value metric-green">{prec*100:.1f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Recall</div>
    <div class="metric-value metric-green">{rec*100:.1f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">F1 Score</div>
    <div class="metric-value metric-green">{f1*100:.1f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Confidence (R²)</div>
    <div class="metric-value metric-accent">{r2*100:.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-heading">Visualization Charts</div>',
            unsafe_allow_html=True)

viz_col1, viz_col2, viz_col3 = st.columns(3)

with viz_col1:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    labels = ["Fail (0)", "Pass (1)"]
    sns.heatmap(
        conf_cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor=GRID_CLR,
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("Actual Label", fontsize=9)
    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with viz_col2:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    metrics_names  = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metrics_values = [acc, prec, rec, f1]
    bar_colors = [ACCENT, ACCENT2, WARN_CLR, DANGER]
    bars = ax.bar(metrics_names, metrics_values, color=bar_colors,
                  width=0.55, zorder=2, edgecolor="none")
    for bar, val in zip(bars, metrics_values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.015,
                f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Classification Scores", fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with viz_col3:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(y_test, y_pred, color=ACCENT, alpha=0.6,
               edgecolors="none", s=30, label="Samples", zorder=3)
    perfect = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(perfect, perfect, color=ACCENT2, linewidth=1.5,
            linestyle="--", label="Perfect fit", zorder=4)
    ax.set_xlabel("Actual G3", fontsize=9)
    ax.set_ylabel("Predicted G3", fontsize=9)
    ax.set_title("Actual vs Predicted", fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, zorder=1)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-heading">Interactive Grade Predictor</div>',
            unsafe_allow_html=True)

pred_left, pred_right = st.columns([1.8, 1])

with pred_left:
    st.markdown("Adjust the sliders below and click **Predict** to estimate a student's final grade.")

    c1, c2 = st.columns(2)
    with c1:
        studytime = st.slider("📚 Study Time (1–4)",  min_value=1,  max_value=4,  value=2)
        failures  = st.slider("❌ Past Failures (0–4)", min_value=0, max_value=4,  value=0)
        absences  = st.slider("🚪 Absences (0–30)",   min_value=0,  max_value=30, value=5)
    with c2:
        g1 = st.slider("📝 G1 — Period 1 Grade (0–20)", min_value=0, max_value=20, value=10)
        g2 = st.slider("📝 G2 — Period 2 Grade (0–20)", min_value=0, max_value=20, value=11)

    predict_btn = st.button("🎯  Predict Final Grade")

with pred_right:
    if predict_btn:
        input_data = np.array([[studytime, failures, absences, g1, g2]])
        predicted_g3 = model.predict(input_data)[0]
        predicted_g3 = float(np.clip(predicted_g3, 0, 20))

        pass_fail = "PASS" if predicted_g3 >= 10 else "FAIL"
        result_class = "result-pass" if predicted_g3 >= 10 else "result-fail"

        st.markdown(f"""
        <div class="pred-box">
          <div class="pred-label">Predicted Final Grade</div>
          <div class="pred-grade">{predicted_g3:.1f}</div>
          <div class="pred-label">out of 20</div>
          <br>
          <div class="{result_class}">{pass_fail}</div>
          <br>
          <div style="font-size:0.8rem;color:#7b88b0;margin-top:0.5rem;">
            Model confidence (R²): {r2*100:.1f}%
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="pred-box" style="opacity:0.55;">
          <div class="pred-label">Predicted Final Grade</div>
          <div class="pred-grade" style="color:#3d4870;">—</div>
          <div class="pred-label">Set sliders and click Predict</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("UCI Student Performance Dataset · Linear Regression · scikit-learn · Streamlit")