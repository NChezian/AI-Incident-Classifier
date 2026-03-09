"""
AI-Powered IT Incident Log Classifier — Streamlit App

Run: streamlit run src/app.py
"""

import json
import joblib
import streamlit as st
import numpy as np

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Incident Classifier",
    page_icon="🔍",
    layout="centered",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 24px;
        color: white;
        margin: 8px 0;
    }
    .result-card h3 {
        margin: 0 0 4px 0;
        font-size: 14px;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-card p {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
    }
    .confidence-bar {
        background: rgba(255,255,255,0.2);
        border-radius: 8px;
        height: 8px;
        margin-top: 8px;
    }
    .confidence-fill {
        background: #00e676;
        border-radius: 8px;
        height: 8px;
        transition: width 0.6s ease;
    }
    .metric-row {
        display: flex;
        gap: 16px;
        margin-bottom: 16px;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Load models (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    cat_model = joblib.load("models/category_model.joblib")
    pri_model = joblib.load("models/priority_model.joblib")
    with open("models/team_map.json") as f:
        team_map = json.load(f)
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    return cat_model, pri_model, team_map, metrics


try:
    cat_model, pri_model, team_map, metrics = load_models()
except FileNotFoundError:
    st.error(
        "⚠️ Models not found. Please run `python src/train_model.py` first."
    )
    st.stop()


# ──────────────────────────────────────────────
# Priority colour helper
# ──────────────────────────────────────────────
PRIORITY_COLORS = {
    "Critical": "#ff1744",
    "High": "#ff9100",
    "Medium": "#ffea00",
    "Low": "#00e676",
}

PRIORITY_EMOJI = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "🟢",
}

CATEGORY_EMOJI = {
    "Network": "🌐",
    "Software": "💻",
    "Hardware": "🖥️",
    "Access": "🔐",
}


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("🔍 AI Incident Log Classifier")
st.caption(
    "Paste an IT incident description below and the model will predict "
    "the **category**, **priority**, and **suggested team**."
)

# ──────────────────────────────────────────────
# Sidebar — model info
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.metric("Category Accuracy", f"{metrics['category_accuracy']:.1%}")
    st.metric("Priority Accuracy", f"{metrics['priority_accuracy']:.1%}")
    st.metric("Training Samples", f"{metrics['training_samples']:,}")
    st.divider()
    st.subheader("Categories")
    for cat in metrics["categories"]:
        st.write(f"{CATEGORY_EMOJI.get(cat, '•')} {cat}")
    st.divider()
    st.subheader("About")
    st.markdown(
        "Built by **Nikhil Chezian** as a portfolio project.\n\n"
        "Stack: Python · Scikit-learn · TF-IDF · Logistic Regression · Streamlit"
    )

# ──────────────────────────────────────────────
# Example tickets for quick demo
# ──────────────────────────────────────────────
EXAMPLES = {
    "🌐 VPN Timeout": "Unable to connect to corporate VPN from remote location. Connection times out after 30 seconds. Multiple retries failed. User is unable to access internal resources.",
    "💻 SAP Runtime Error": "SAP transaction SE38 throwing runtime error DBIF_RSQL_SQL_ERROR during payroll batch processing. Affecting 200+ employees' salary calculations. Needs immediate fix.",
    "🖥️ Server Overheating": "Server CPU temperature reaching 92°C under normal load. Fan RPM dropping below threshold. Risk of thermal shutdown in production environment.",
    "🔐 Account Lockout": "User locked out of Active Directory account after 5 failed login attempts. Self-service password reset not working. User has client meeting in 30 minutes.",
}

st.subheader("Try an Example")
cols = st.columns(len(EXAMPLES))
selected_example = ""
for i, (label, text) in enumerate(EXAMPLES.items()):
    if cols[i].button(label, use_container_width=True):
        selected_example = text

# ──────────────────────────────────────────────
# Input area
# ──────────────────────────────────────────────
st.subheader("📝 Enter Incident Description")
user_input = st.text_area(
    "Paste log entry or incident ticket text:",
    value=selected_example,
    height=150,
    placeholder="e.g. User cannot connect to Wi-Fi on floor 3. Multiple devices affected since 9 AM...",
)

# ──────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────
if st.button("🚀 Classify Incident", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter an incident description first.")
    else:
        with st.spinner("Analysing..."):
            # Category prediction
            cat_pred = cat_model.predict([user_input])[0]
            cat_proba = cat_model.predict_proba([user_input])[0]
            cat_conf = float(np.max(cat_proba))

            # Priority prediction
            pri_pred = pri_model.predict([user_input])[0]
            pri_proba = pri_model.predict_proba([user_input])[0]
            pri_conf = float(np.max(pri_proba))

            # Team suggestion
            suggested_team = team_map.get(cat_pred, "General IT Support")

        # ── Results ──
        st.divider()
        st.subheader("📊 Classification Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            emoji = CATEGORY_EMOJI.get(cat_pred, "📁")
            st.markdown(
                f"""
                <div class="result-card">
                    <h3>Category</h3>
                    <p>{emoji} {cat_pred}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{cat_conf*100:.0f}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Confidence: {cat_conf:.1%}")

        with col2:
            emoji = PRIORITY_EMOJI.get(pri_pred, "⚪")
            color = PRIORITY_COLORS.get(pri_pred, "#888")
            st.markdown(
                f"""
                <div class="result-card" style="background:linear-gradient(135deg, {color}88 0%, {color} 100%);">
                    <h3>Priority</h3>
                    <p>{emoji} {pri_pred}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{pri_conf*100:.0f}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Confidence: {pri_conf:.1%}")

        with col3:
            st.markdown(
                f"""
                <div class="result-card" style="background:linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                    <h3>Suggested Team</h3>
                    <p>👥 {suggested_team}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Detailed probabilities ──
        with st.expander("🔬 Detailed Probabilities"):
            st.write("**Category Probabilities:**")
            cat_classes = cat_model.classes_
            for cls, prob in sorted(
                zip(cat_classes, cat_proba), key=lambda x: -x[1]
            ):
                st.progress(prob, text=f"{cls}: {prob:.1%}")

            st.write("**Priority Probabilities:**")
            pri_classes = pri_model.classes_
            for cls, prob in sorted(
                zip(pri_classes, pri_proba), key=lambda x: -x[1]
            ):
                st.progress(prob, text=f"{cls}: {prob:.1%}")
