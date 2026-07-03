import os
import re
import string
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChildSafe AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0f1e;
    color: #e8eaf2;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1530 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * { color: #cdd6f4 !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
h1 { color: #00e5ff !important; letter-spacing: -1px; }
h2 { color: #90caf9 !important; }
h3 { color: #b0bec5 !important; }

.risk-card {
    background: linear-gradient(135deg, #0d1b35, #112040);
    border-radius: 16px;
    padding: 24px 28px;
    border: 1px solid #1e3a5f;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0,229,255,0.06);
}
.badge-safe       { background:#00c853; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px; }
.badge-suspicious { background:#ff9800; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px; }
.badge-dangerous  { background:#f44336; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px; }

.action-box {
    background: rgba(244,67,54,0.12);
    border: 1px solid rgba(244,67,54,0.4);
    border-radius: 12px;
    padding: 12px 18px;
    margin: 6px 0;
    font-size: 14px;
}
.action-safe {
    background: rgba(0,200,83,0.10);
    border: 1px solid rgba(0,200,83,0.4);
    border-radius: 12px;
    padding: 12px 18px;
    margin: 6px 0;
    font-size: 14px;
}
.helpline-card {
    background: linear-gradient(135deg, #0d1b35, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 16px 18px;
    margin: 6px 0;
    text-align: center;
}
.helpline-number {
    font-size: 28px;
    font-weight: 800;
    color: #00e5ff;
    font-family: 'Syne', sans-serif;
}
.helpline-title {
    font-size: 12px;
    color: #90caf9;
    margin-top: 4px;
}
.helpline-link {
    font-size: 13px;
    color: #64b5f6;
    margin-top: 6px;
    word-break: break-all;
}
.kw {
    background:#ff1744; color:#fff; padding:1px 6px; border-radius:4px;
    font-weight:600; font-size:13px; margin:0 2px;
}
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    color: #90caf9 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom: 2px solid #00e5ff !important;
}
div[data-testid="metric-container"] {
    background: #0d1b35;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 12px;
}
textarea, .stTextInput input {
    background: #0d1b35 !important;
    color: #e8eaf2 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}
.stButton>button {
    background: linear-gradient(135deg, #0077b6, #00b4d8);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    padding: 10px 24px;
    transition: all 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #00b4d8, #00e5ff);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,229,255,0.25);
}
hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RISK KEYWORD ENGINE
# ─────────────────────────────────────────────────────────────────────────────
RISK_DICT = {
    "grooming": {
        "weight": 3,
        "keywords": [
            "meet secretly", "don't tell your parents", "keep this secret",
            "secret chat", "come alone", "meet me alone", "special friend",
            "our little secret", "don't tell anyone", "private conversation",
            "you're so mature", "you understand me like no one else",
            "never met anyone like you", "you're not like other kids",
            "meet me tomorrow", "meet up alone", "just between us",
            "no one needs to know", "delete this message",
        ],
    },
    "image_exploitation": {
        "weight": 4,
        "keywords": [
            "send photo", "send me a picture", "send pic", "private picture",
            "send video", "video call alone", "show me", "send nudes",
            "send selfie", "your photo", "share photo", "take picture",
            "send me photos", "send your picture", "show your",
        ],
    },
    "phishing_financial": {
        "weight": 3,
        "keywords": [
            "click this link", "click here", "otp", "password",
            "money transfer", "send money", "bank details", "verify account",
            "gift for you", "free gift", "you won", "claim your prize",
            "enter your details", "confirm your account", "your pin",
            "enter otp", "share your password", "login details",
        ],
    },
    "threats_blackmail": {
        "weight": 5,
        "keywords": [
            "i will tell everyone", "i will share your photos",
            "i will expose you", "do as i say", "you'll regret this",
            "i know where you live", "i will hurt you",
            "blackmail", "sextortion", "share your video",
            "i will post your pictures", "i will ruin you",
            "you have no choice", "do what i say or",
        ],
    },
    "isolation": {
        "weight": 2,
        "keywords": [
            "don't tell your friends", "stay away from them",
            "they don't understand you", "only i care about you",
            "block your friends", "leave your family",
            "they're bad for you", "trust only me",
            "your parents won't understand", "don't listen to them",
        ],
    },
}

def detect_risks(text: str):
    text_lower = text.lower()
    matched = []
    for cat, data in RISK_DICT.items():
        for kw in data["keywords"]:
            if kw in text_lower:
                matched.append((kw, cat, data["weight"]))
    total_score = sum(w for _, _, w in matched)
    if total_score == 0:
        risk_level = "Safe"
    elif total_score <= 4:
        risk_level = "Suspicious"
    else:
        risk_level = "Dangerous"
    highlighted = text
    for kw, _, _ in matched:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        highlighted = pattern.sub(f"<span class='kw'>{kw.upper()}</span>", highlighted)
    return matched, total_score, risk_level, highlighted

def get_recommendations(risk_level: str) -> list:
    base = {
        "Safe": [
            "✅ No immediate action required.",
            "👀 Continue monitoring conversations periodically.",
            "💬 Encourage open communication with the child.",
        ],
        "Suspicious": [
            "⚠️ Review the full conversation history.",
            "🔒 Restrict contact with this sender.",
            "👨‍👩‍👧 Inform Parents / Guardian immediately.",
            "📝 Save a copy of this conversation as evidence.",
            "🧑‍🏫 Contact a Trusted Adult or School Counsellor.",
            "🚔 Report at cybercrime.gov.in",
        ],
        "Dangerous": [
            "🚫 Block this User immediately.",
            "📢 Report this User to the platform.",
            "👨‍👩‍👧 Inform Parents / Guardian RIGHT NOW.",
            "📁 Save all Evidence (screenshots, chat logs).",
            "🚔 Call Cyber Crime Helpline: 1930",
            "👶 Call Childline: 1098",
            "🌐 Report online: cybercrime.gov.in",
        ],
    }
    return base.get(risk_level, [])

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_dir = "models/saved"
    models = {}
    if not os.path.exists(model_dir):
        return models
    for fname in os.listdir(model_dir):
        if fname.endswith(".pkl"):
            name = fname.replace(".pkl", "")
            try:
                obj = joblib.load(os.path.join(model_dir, fname))
                models[name] = obj
            except Exception:
                pass
    return models

def ml_predict(text: str, models: dict):
    results = []
    label_map = {0: "Safe", 1: "Suspicious", 2: "Dangerous"}
    for name, obj in models.items():
        try:
            pipeline = obj["pipeline"] if isinstance(obj, dict) else obj
            proba = pipeline.predict_proba([text])[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(np.max(proba)) * 100
            results.append({
                "model": name,
                "prediction": label_map.get(pred_idx, str(pred_idx)),
                "confidence": round(confidence, 1),
            })
        except Exception:
            results.append({"model": name, "prediction": "Error", "confidence": 0.0})
    return results

# ─────────────────────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_ocr(image: Image.Image) -> str:
    try:
        import os
        import pytesseract

        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]

        found = False
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                found = True
                break

        if not found:
            return "[OCR Error: Tesseract not installed]"

        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()

    except ImportError:
        return "[pytesseract not installed]"

    except Exception as e:
        return f"[OCR Error: {e}]"

# ─────────────────────────────────────────────────────────────────────────────
# PHONE NUMBER CHECKER
# ─────────────────────────────────────────────────────────────────────────────
def check_phone_number(phone_number: str, api_key: str) -> dict:
    url = "https://apilayer.net/api/validate"
    params = {
        "access_key": api_key,
        "number": phone_number.strip(),
        "country_code": "IN",
        "format": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()
        if not data.get("valid") and "error" in data:
            return {"error": data["error"].get("info", "API error")}
        return {
            "valid":     data.get("valid", False),
            "number":    data.get("international_format", phone_number),
            "country":   data.get("country_name", "Unknown"),
            "carrier":   data.get("carrier", "Unknown"),
            "line_type": data.get("line_type", "Unknown"),
            "location":  data.get("location", "Unknown"),
        }
    except requests.exceptions.Timeout:
        return {"error": "Request timed out."}
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
def risk_gauge(score: int, max_score: int = 20) -> go.Figure:
    colour = "#00c853" if score == 0 else ("#ff9800" if score <= 4 else "#f44336")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Risk Score", "font": {"color": "#e8eaf2", "size": 16}},
        gauge={
            "axis": {"range": [0, max_score], "tickcolor": "#e8eaf2"},
            "bar": {"color": colour},
            "bgcolor": "#0d1b35",
            "bordercolor": "#1e3a5f",
            "steps": [
                {"range": [0, 4],          "color": "rgba(0,200,83,0.15)"},
                {"range": [4, 10],         "color": "rgba(255,152,0,0.15)"},
                {"range": [10, max_score], "color": "rgba(244,67,54,0.15)"},
            ],
            "threshold": {
                "line": {"color": colour, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"font": {"color": colour, "size": 36}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eaf2",
        height=240,
        margin=dict(t=40, b=10, l=30, r=30),
    )
    return fig

def confidence_bar(ml_results: list) -> go.Figure:
    if not ml_results:
        return None
    colour_map = {"Safe": "#00c853", "Suspicious": "#ff9800", "Dangerous": "#f44336"}
    colours = [colour_map.get(r["prediction"], "#90caf9") for r in ml_results]
    fig = go.Figure(go.Bar(
        x=[r["model"] for r in ml_results],
        y=[r["confidence"] for r in ml_results],
        marker_color=colours,
        text=[f"{r['prediction']}<br>{r['confidence']}%" for r in ml_results],
        textposition="outside",
    ))
    fig.update_layout(
        title="ML Model Confidence (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eaf2",
        yaxis=dict(range=[0, 115], gridcolor="#1e3a5f"),
        xaxis=dict(tickangle=-30),
        height=320,
        margin=dict(t=50, b=80),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SHARED ANALYSIS RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def render_analysis(text: str, models: dict):
    if not text.strip():
        st.warning("No text to analyse.")
        return

    matched, score, risk_level, highlighted = detect_risks(text)
    ml_results = ml_predict(text, models)
    recommendations = get_recommendations(risk_level)

    st.subheader("📊 Risk Analysis Results")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            risk_gauge(score),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with col2:
        badge_cls = {
            "Safe": "badge-safe",
            "Suspicious": "badge-suspicious",
            "Dangerous": "badge-dangerous"
        }[risk_level]

        st.markdown("### Risk Level")
        st.markdown(
            f"<span class='{badge_cls}'>{risk_level.upper()}</span>",
            unsafe_allow_html=True
        )

        st.markdown(f"### Risk Score: {score}")

        if matched:
            st.metric("Keywords Detected", len(matched))

    st.markdown("---")

    if ml_results:
        st.subheader("🤖 Model Confidence")

        st.plotly_chart(
            confidence_bar(ml_results),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown("---")

    st.subheader("📝 Analysed Message")

    st.markdown(
        f"""
        <div class='risk-card'
        style='font-size:15px;line-height:1.7'>
        {highlighted}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🔑 Detected Keywords")

        if matched:
            for kw, cat, wt in matched:
                st.markdown(
                    f"""
                    <div class='action-box'>
                    ⚠️ <b>{kw}</b><br>
                    Category: {cat}<br>
                    Weight: {wt}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("No suspicious keywords detected.")

    with col4:
        st.subheader("🛡️ Recommended Actions")

        action_cls = (
            "action-safe"
            if risk_level == "Safe"
            else "action-box"
        )

        for rec in recommendations:
            st.markdown(
                f"<div class='{action_cls}'>{rec}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.subheader("📋 ML Predictions")

    if ml_results:
        df_ml = pd.DataFrame(ml_results)

        df_ml.columns = [
            "Model",
            "Prediction",
            "Confidence (%)"
        ]

        st.dataframe(
            df_ml,
            use_container_width=True,
            hide_index=True
        )# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ ChildSafe AI")
    st.markdown("**Cyber Threat Detection**\nfor Children & Guardians")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    selected_model_key = st.selectbox(
        "Primary ML Model",
        ["Best Available (Auto)", "LR_TF-IDF", "RF_TF-IDF", "LR_Count", "RF_Count"],
    )

    numverify_key = "43e76592d663b916d59600dbb9972941"

    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    st.markdown("""
1. **Screenshot Tab** – Upload chat screenshot; OCR reads the text.
2. **Text Tab** – Paste any suspicious message directly.
3. **Phone Tab** – Verify a suspicious phone number.
4. **Models Tab** – Compare all 6 trained classifiers.
    """)

    st.markdown("---")
    st.markdown("### 🇮🇳 Indian Helplines")
    st.markdown("""
- 🚔 Cyber Crime: **1930**
- 👶 Childline: **1098**
- 👮 Women Helpline: **1091**
- 🚨 Police: **100**
- 🌐 [cybercrime.gov.in](https://cybercrime.gov.in)
- 🏛️ [ncpcr.gov.in](https://ncpcr.gov.in)
- 🔒 [cert-in.org.in](https://cert-in.org.in)
    """)
    st.markdown("---")
    st.caption("© 2025 ChildSafe AI · Final Year Cybersecurity Project")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 10px 0 4px 0;'>
  <h1 style='font-size:2.6rem; margin-bottom:0;'>🛡️ ChildSafe AI</h1>
  <p style='color:#90caf9; font-size:1rem; margin-top:4px;'>
    Cyber Threat Detection System for Children's Online Safety
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
models = load_models()
if models:
    st.sidebar.success(f"✅ {len(models)} model(s) loaded")
else:
    st.sidebar.warning("⚠️ No models found.\nRun python models/train_evaluate.py first.")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📷  Screenshot Analyser",
    "✍️   Text Analyser",
    "📞  Phone Number Checker",
    "📊  Model Comparison",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – SCREENSHOT ANALYSER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📷 Upload a Suspicious Chat Screenshot")
    st.markdown("Upload a PNG / JPG image. OCR will extract the text automatically.")

    col_up, col_prev = st.columns([1, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

    ocr_text = ""
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        with col_prev:
            st.image(image, caption="Uploaded Screenshot", width="stretch")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔍 Extract Text (OCR)", use_container_width=True):
                with st.spinner("Running OCR…"):
                    ocr_text = extract_text_ocr(image)
                st.session_state["ocr_text_tab1"] = ocr_text

        if "ocr_text_tab1" in st.session_state:
            ocr_text = st.session_state["ocr_text_tab1"]

        if ocr_text:
            st.markdown("#### 📄 Extracted Text")
            edited_text = st.text_area(
                "Review / edit extracted text before analysis:",
                value=ocr_text,
                height=140,
                key="ocr_edit",
            )
            with col_b:
                if st.button("🛡️ Analyse Message", use_container_width=True, key="analyse_ocr"):
                    st.markdown("---")
                    render_analysis(edited_text, models)
    else:
        st.info("👆 Upload a screenshot to get started.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – TEXT ANALYSER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### ✍️ Paste or Type a Suspicious Message")

    example_messages = {
        "Select an example…": "",
        "⚠️ Grooming attempt": "Hey! Don't tell your parents about our chats. You're so mature for your age. Let's meet secretly tomorrow.",
        "🔴 Image request": "Send me your photo. Don't tell anyone. This is just between us.",
        "🔴 Blackmail / Sextortion": "I will share your photos with everyone if you don't do as I say. You'll regret this.",
        "🟡 Phishing / OTP fraud": "Congratulations! You've won a prize. Click this link and enter your OTP to claim it.",
        "🟡 Financial fraud": "Send money to this account or enter your bank details to receive your gift.",
        "✅ Safe message": "Hey! Are you coming to school tomorrow? We have a project due.",
    }

    selected_example = st.selectbox("Load an example message:", list(example_messages.keys()))
    example_val = example_messages[selected_example]

    user_text = st.text_area(
        "Message to analyse:",
        value=example_val,
        height=160,
        placeholder="Type or paste the suspicious message here…",
        key="text_input",
    )

    if st.button("🛡️ Analyse Message", use_container_width=False, key="analyse_text"):
        if user_text.strip():
            st.markdown("---")
            render_analysis(user_text, models)
        else:
            st.warning("Please enter a message to analyse.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – PHONE NUMBER CHECKER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📞 Suspicious Phone Number Checker")
    st.markdown(
        "Validate an unknown or suspicious phone number using **NumVerify API**. "
        "Get a free API key (100 checks/month) at [numverify.com](https://numverify.com) "
        "and paste it in the sidebar."
    )

    phone_col, btn_col = st.columns([3, 1])
    with phone_col:
        phone_input = st.text_input(
            "Phone Number",
            placeholder="+91 9876543210",
        )
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        check_btn = st.button("🔍 Check Number", use_container_width=True)

    if check_btn:
        if not phone_input.strip():
            st.warning("Please enter a phone number.")
        elif not numverify_key:
            st.error(
                "⚠️ NumVerify API key not set.\n\n"
                "1. Go to [numverify.com](https://numverify.com)\n"
                "2. Sign up free\n"
                "3. Copy your API key\n"
                "4. Paste it in the **sidebar** under NumVerify API Key"
            )
        else:
            with st.spinner("Validating number…"):
                result = check_phone_number(phone_input, numverify_key)

            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Valid Number", "✅ Yes" if result["valid"] else "❌ No")
                c2.metric("Country",   result["country"])
                c3.metric("Line Type", result["line_type"])
                c4.metric("Carrier",   result["carrier"])

                st.markdown(
                    f"<div class='risk-card'>📍 <b>Location:</b> {result['location']}&nbsp;&nbsp;"
                    f"🌐 <b>International Format:</b> {result['number']}</div>",
                    unsafe_allow_html=True,
                )

                line = result.get("line_type", "").lower()
                if not result["valid"]:
                    st.markdown(
                        "<div class='action-box'>🚫 <b>INVALID number</b> — could be spoofed or fake. "
                        "Report to cybercrime.gov.in</div>",
                        unsafe_allow_html=True,
                    )
                elif line in ["voip", "virtual"]:
                    st.markdown(
                        "<div class='action-box'>⚠️ <b>VOIP / Virtual number</b> — commonly used to hide "
                        "identity. Treat with caution. Report if suspicious.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='action-safe'>✅ Standard registered number. "
                        "Still report to authorities if message content is suspicious.</div>",
                        unsafe_allow_html=True,
                    )

    # ── Indian Government Helplines ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🇮🇳 Indian Government Cyber Safety Helplines")

    helplines = [
        {"emoji": "🚔", "title": "Cyber Crime Helpline",     "number": "1930",  "link": None},
        {"emoji": "👶", "title": "Childline India",           "number": "1098",  "link": None},
        {"emoji": "👮", "title": "Women Helpline",            "number": "1091",  "link": None},
        {"emoji": "🚨", "title": "Police Emergency",          "number": "100",   "link": None},
        {"emoji": "🚑", "title": "National Emergency",        "number": "112",   "link": None},
        {"emoji": "🌐", "title": "Cyber Crime Portal",        "number": None,    "link": "https://cybercrime.gov.in"},
        {"emoji": "🏛️", "title": "NCPCR Child Rights",       "number": None,    "link": "https://ncpcr.gov.in"},
        {"emoji": "🔒", "title": "CERT-In (Cyber Security)",  "number": None,    "link": "https://cert-in.org.in"},
    ]

    cols = st.columns(4)
    for i, h in enumerate(helplines):
        with cols[i % 4]:
            if h["number"]:
                st.markdown(
                    f"<div class='helpline-card'>"
                    f"<div style='font-size:24px'>{h['emoji']}</div>"
                    f"<div class='helpline-number'>{h['number']}</div>"
                    f"<div class='helpline-title'>{h['title']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='helpline-card'>"
                    f"<div style='font-size:24px'>{h['emoji']}</div>"
                    f"<div class='helpline-link'><a href='{h['link']}' target='_blank'>{h['link'].replace('https://','')}</a></div>"
                    f"<div class='helpline-title'>{h['title']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "📌 **How to report cyber crime in India:**  \n"
        "1. Call **1930** (National Cyber Crime Helpline)  \n"
        "2. File online complaint at **cybercrime.gov.in**  \n"
        "3. Visit your nearest **Cyber Crime Police Station**  \n"
        "4. For child-related crimes contact **NCPCR**: ncpcr.gov.in"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 ML Model Performance Comparison")

    results_path = "outputs/model_comparison.csv"
    fallback_data = {
        "Model":     ["LR_TF-IDF", "LR_Count",  "RF_TF-IDF",
                      "RF_Count",  "DT_TF-IDF", "DT_Count"],
        "Accuracy":  [82.46, 81.63, 80.92, 80.75, 76.42, 76.51],
        "Precision": [81.99, 81.88, 80.37, 80.46, 80.21, 80.41],
        "Recall":    [82.46, 81.63, 80.92, 80.75, 76.42, 76.51],
        "F1 Score":  [82.15, 81.75, 80.52, 80.55, 73.43, 73.43],
    }

    if os.path.exists(results_path):
        try:
            df_results = pd.read_csv(results_path)
        except Exception:
            df_results = pd.DataFrame(fallback_data)
    else:
        df_results = pd.DataFrame(fallback_data)

    st.dataframe(
        df_results.style.background_gradient(cmap="Blues", subset=["Accuracy", "F1 Score"]),
        use_container_width=True,
        hide_index=True,
    )

    fig_cmp = go.Figure()
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colours = ["#00b4d8", "#0077b6", "#00e5ff", "#90caf9"]

    for metric, colour in zip(metrics, colours):
        if metric in df_results.columns:
            fig_cmp.add_trace(go.Bar(
                name=metric,
                x=df_results["Model"],
                y=df_results[metric],
                marker_color=colour,
            ))

    fig_cmp.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eaf2",
        yaxis=dict(range=[60, 100], gridcolor="#1e3a5f", title="Score (%)"),
        xaxis=dict(tickangle=-20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=400,
        margin=dict(t=20, b=80)
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🏆 Deployment Recommendation")
    st.markdown("""
<div class='risk-card'>
<b>Recommended Model for Deployment: LR_TF-IDF (Logistic Regression + TF-IDF)</b><br><br>
Based on evaluation on the real Kaggle cyberbullying dataset (47,291 tweets):
<ul>
  <li>🥇 <b>Highest Accuracy: 82.46%</b> and best F1 Score: 82.15%</li>
  <li>⚡ <b>Fastest prediction</b> — suitable for real-time chat monitoring</li>
  <li>🔍 <b>TF-IDF</b> captures word importance better than raw counts</li>
  <li>📊 <b>Best balance</b> of Precision and Recall across all 3 classes</li>
  <li>🧠 <b>Interpretable</b> — coefficients show which words drive predictions</li>
</ul>
<b>Note:</b> Decision Tree models underperform on the Safe class due to class imbalance (only 16.5% safe samples). 
For production, consider oversampling the Safe class using SMOTE to improve Safe detection.
</div>
""", unsafe_allow_html=True)