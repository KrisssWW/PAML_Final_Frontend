import streamlit as st
import pickle
import numpy as np
from scipy import sparse

# ── Model class definitions (required for pickle to load) ────
class LogisticRegressionNumPy:
    def __init__(self, lr=0.1, n_iter=100, lambda_reg=0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y, verbose=True):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for i in range(self.n_iter):
            z = X.dot(self.weights) + self.bias
            if sparse.issparse(z):
                z = np.asarray(z).flatten()
            y_pred = self.sigmoid(z)
            error = y_pred - y
            dw = (X.T.dot(error) / n_samples) + self.lambda_reg * self.weights
            if sparse.issparse(dw):
                dw = np.asarray(dw).flatten()
            db = np.mean(error)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            eps = 1e-9
            loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
            self.loss_history.append(loss)
        return self

    def predict_proba(self, X):
        z = X.dot(self.weights) + self.bias
        if sparse.issparse(z):
            z = np.asarray(z).flatten()
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.log_priors = {}
        self.log_likelihoods = {}

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.log_priors[c] = np.log(X_c.shape[0] / n_samples)
            word_counts = np.asarray(X_c.sum(axis=0)).flatten() + self.alpha
            total = word_counts.sum()
            self.log_likelihoods[c] = np.log(word_counts / total)
        return self

    def predict_proba(self, X):
        log_probs = []
        for c in self.classes:
            log_p = self.log_priors[c] + X.dot(self.log_likelihoods[c])
            if sparse.issparse(log_p):
                log_p = np.asarray(log_p).flatten()
            log_probs.append(log_p)
        log_probs = np.column_stack(log_probs)
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Crisis Language Detection",
    page_icon="✦",
    layout="wide"
)

# ── Global styles ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: #2a2520;
}

/* Prevent Streamlit from injecting white text anywhere in the main content */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] li {
    color: #2a2520;
}

.stApp {
    background-color: #f5f0e8;
    background-image:
        radial-gradient(ellipse 60% 50% at 80% 10%, rgba(220,190,230,0.35) 0%, transparent 60%),
        radial-gradient(ellipse 50% 45% at 15% 80%, rgba(180,210,240,0.30) 0%, transparent 55%),
        radial-gradient(ellipse 45% 40% at 55% 55%, rgba(255,220,180,0.28) 0%, transparent 50%);
}

/* ── Sidebar: fixed width, no resize ── */
[data-testid="stSidebar"] {
    background-color: #f5f0e8 !important;
    border-right: none !important;
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
[data-testid="stSidebarResizeHandle"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ── Main content: fill remaining width ── */
.block-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1rem !important;
}

[data-testid="stSidebarContent"] {
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}

[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
    color: #3a3530 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #8a7f74 !important;
}

/* ── Headings ── */
h1, h1 *, h1 span, h1 p {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 3.2rem !important;
    font-weight: 400 !important;
    color: #1e1a16 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
    margin-bottom: 0.2rem !important;
}
h2, h2 *, h2 span {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-weight: 400 !important;
    color: #1e1a16 !important;
    font-size: 1.5rem !important;
}
h3, h3 *, h3 span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #8a7f74 !important;
    margin-bottom: 0.75rem !important;
}

/* ── Subtitle / caption text ── */
.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #9a8f82;
    letter-spacing: 0.05em;
    margin-top: -0.5rem;
    margin-bottom: 2.5rem;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,0,0,0.08) !important;
    margin: 2rem 0 !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background-color: rgba(255,255,255,0.65) !important;
    border: 1px solid rgba(0,0,0,0.10) !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    color: #2a2520 !important;
    padding: 1rem 1.1rem !important;
    backdrop-filter: blur(8px);
    transition: border 0.2s ease, box-shadow 0.2s ease;
}
.stTextArea textarea:focus {
    border-color: rgba(160,138,184,0.7) !important;
    box-shadow: 0 0 0 3px rgba(160,138,184,0.20) !important;
}
.stTextArea label {
    font-size: 0.72rem !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: #8a7f74 !important;
    font-weight: 500 !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: #1e1a16 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    cursor: pointer !important;
    transition: background 0.2s ease, transform 0.15s ease !important;
}
.stButton > button[kind="primary"] *,
button[data-testid="baseButton-primary"] * {
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: #3a3530 !important;
    transform: translateY(-1px) !important;
}

/* ── Result cards ── */
.result-card {
    background: rgba(255,255,255,0.60);
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

/* ── Badge dot colors ── */
.dot-high { color: #dc2626 !important; }
.dot-low  { color: #2a7a50 !important; }

/* ── Risk label badges ── */
.badge-high {
    display: inline-block;
    background: rgba(220,80,80,0.10);
    color: #b03030;
    border: 1px solid rgba(180,60,60,0.25);
    border-radius: 100px;
    padding: 0.4rem 1.2rem;
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.badge-low {
    display: inline-block;
    background: rgba(60,150,100,0.10);
    color: #2a7a50;
    border: 1px solid rgba(50,130,80,0.25);
    border-radius: 100px;
    padding: 0.4rem 1.2rem;
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}

/* ── Score display ── */
.score-number {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 3rem;
    color: #1e1a16;
    line-height: 1;
    margin: 0.5rem 0 0.8rem 0;
}

/* ── Custom progress bar ── */
.progress-track {
    height: 6px;
    background: rgba(0,0,0,0.08);
    border-radius: 100px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.progress-fill-high {
    height: 100%;
    background: linear-gradient(90deg, #e8a0a0, #c05050);
    border-radius: 100px;
    transition: width 0.6s ease;
}
.progress-fill-low {
    height: 100%;
    background: linear-gradient(90deg, #a0d0b8, #3a9060);
    border-radius: 100px;
    transition: width 0.6s ease;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(0,0,0,0.07) !important;
}

/* ── Alert / info boxes ── */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid rgba(0,0,0,0.07) !important;
    background: rgba(255,255,255,0.5) !important;
}
.stAlert p, .stAlert div, .stAlert span {
    color: #3a3530 !important;
}

/* ── Selectbox & slider ── */
.stSelectbox > div > div {
    background-color: rgba(255,255,255,0.5) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(0,0,0,0.10) !important;
}

/* ── Slider: muted mauve to match lavender gradient accents ── */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #a08ab8 !important;
    border-color: #a08ab8 !important;
}
div[data-testid="stSlider"] div[role="slider"]:focus,
div[data-testid="stSlider"] div[role="slider"]:active {
    box-shadow: 0 0 0 8px rgba(160, 138, 184, 0.25) !important;
    outline: none !important;
}
div[data-testid="stSlider"] > div > div > div > div:first-child,
[data-baseweb="slider"] [class*="track" i],
[data-baseweb="slider"] [class*="Track"] {
    background: linear-gradient(90deg, #c4aed8, #a08ab8) !important;
}

/* ── Deploy button text ── */
[data-testid="stToolbar"] button,
[data-testid="stHeader"] button,
button[data-testid="baseButton-header"],
button[data-testid="baseButton-headerNoPadding"] {
    color: #ffffff !important;
}
[data-testid="stToolbar"] button *,
[data-testid="stHeader"] button *,
button[data-testid="baseButton-header"] *,
button[data-testid="baseButton-headerNoPadding"] * {
    color: #ffffff !important;
}

/* ── Top header bar ── */
[data-testid="stHeader"] {
    background-color: #1e1a16 !important;
}

/* ── Hide Streamlit default footer & menu ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("logistic_regression.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("naive_bayes.pkl", "rb") as f:
        nb = pickle.load(f)
    feature_names = np.load("feature_names.npy", allow_pickle=True)
    return tfidf, lr, nb, feature_names

tfidf, lr_model, nb_model, feature_names = load_models()


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown(
    '<p style="font-size:0.68rem;letter-spacing:0.14em;text-transform:uppercase;color:#9a8f82;margin-bottom:1.2rem;">Controls</p>',
    unsafe_allow_html=True
)
model_choice = st.sidebar.selectbox(
    "Model",
    ["Logistic Regression", "Naive Bayes"],
    label_visibility="visible"
)
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1, max_value=0.9,
    value=0.5, step=0.05
)
st.sidebar.markdown("<hr style='border-top:1px solid rgba(0,0,0,0.08);margin:1.5rem 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(
    '<p style="font-size:0.78rem;color:#9a8f82;line-height:1.7;">Binary classifier for mental health risk screening. '
    'Lower threshold → higher Recall (fewer missed cases).</p>',
    unsafe_allow_html=True
)


# ── Main UI ───────────────────────────────────────────────────
st.markdown("""
<br>
<h1 style="font-family:'DM Serif Display',Georgia,serif;font-size:3.2rem;font-weight:400;
           color:#1e1a16;letter-spacing:-0.02em;line-height:1.15;margin-bottom:0.2rem;">
    Crisis Language<br>Detection
</h1>
<p class="subtitle">Mental Health Risk Screening &nbsp;·&nbsp; INFO 5368, Cornell University</p>
""", unsafe_allow_html=True)

st.markdown("---")

text_input = st.text_area(
    "Text to analyze",
    placeholder="Paste or type a Reddit post here…",
    height=160,
    label_visibility="visible"
)

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("Analyze Text →", type="primary")


if analyze_btn and text_input.strip():
    X = tfidf.transform([text_input])
    model = lr_model if model_choice == "Logistic Regression" else nb_model

    proba = model.predict_proba(X)
    if hasattr(proba, "ndim") and proba.ndim > 1:
        score = float(proba[0, 1])
    else:
        score = float(proba[0])

    label = "High Risk" if score >= threshold else "Low Risk"
    is_high = label == "High Risk"

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        badge_html = '<span class="badge-high"><span class="dot-high">●</span> High Risk</span>' if is_high else '<span class="badge-low"><span class="dot-low">●</span> Low Risk</span>'
        st.markdown(f"""
        <div class="result-card">
            <p style="font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:#9a8f82;margin-bottom:0.8rem;">Risk Assessment</p>
            {badge_html}
            <p style="font-size:0.78rem;color:#9a8f82;margin-top:1rem;line-height:1.6;">
                {"Linguistic patterns associated with crisis language detected." if is_high else "No significant crisis language patterns detected."}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fill_class = "progress-fill-high" if is_high else "progress-fill-low"
        fill_pct = int(score * 100)
        st.markdown(f"""
        <div class="result-card">
            <p style="font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:#9a8f82;margin-bottom:0rem;">Confidence Score</p>
            <p class="score-number">{score:.3f}</p>
            <div class="progress-track">
                <div class="{fill_class}" style="width:{fill_pct}%;"></div>
            </div>
            <p style="font-size:0.75rem;color:#b0a898;margin-top:0.6rem;">Threshold set at {threshold:.2f} · Model: {model_choice}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    if model_choice == "Logistic Regression":
        st.markdown("### Top Contributing Features")
        vec = X.toarray()[0]
        active_idx = np.where(vec > 0)[0]
        if len(active_idx) > 0:
            contrib = vec[active_idx] * lr_model.weights[active_idx]
            top_idx = np.argsort(np.abs(contrib))[-10:][::-1]
            rows = []
            for i in top_idx:
                feat = feature_names[active_idx[i]]
                w = lr_model.weights[active_idx[i]]
                rows.append({"Feature": feat, "Weight": round(float(w), 4)})
            import pandas as pd
            feat_df = pd.DataFrame(rows)
            st.dataframe(feat_df, use_container_width=True)
        else:
            st.info("No significant features found in input.")

    elif model_choice == "Naive Bayes":
        st.markdown("### Top Contributing Features")
        st.info("Feature weights are available for Logistic Regression. Switch the model in the sidebar to view them.")

elif analyze_btn and not text_input.strip():
    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("Please enter some text before analyzing.")
