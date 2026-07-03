"""
ChildSafe AI – Risk Keyword Engine
===================================
Scans messages for grooming, exploitation, phishing,
blackmail, and isolation patterns.
Returns risk score, level, and highlighted HTML.
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# RISK KEYWORD DICTIONARY
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

LABEL_NAMES = {0: "Safe", 1: "Suspicious", 2: "Dangerous"}


# ─────────────────────────────────────────────────────────────────────────────
# CORE DETECTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_risks(text: str):
    """
    Scan text for risk keywords.

    Returns:
        matched_keywords : list of (keyword, category, weight)
        total_score      : int
        risk_level       : str  — Safe | Suspicious | Dangerous
        highlighted_html : str  — text with <span class='kw'>…</span>
    """
    text_lower = text.lower()
    matched = []

    for category, data in RISK_DICT.items():
        for kw in data["keywords"]:
            if kw in text_lower:
                matched.append((kw, category, data["weight"]))

    total_score = sum(w for _, _, w in matched)

    if total_score == 0:
        risk_level = "Safe"
    elif total_score <= 4:
        risk_level = "Suspicious"
    else:
        risk_level = "Dangerous"

    # Build highlighted HTML — wrap matched keywords in <span class='kw'>
    highlighted = text
    for kw, _, _ in matched:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        highlighted = pattern.sub(
            f"<span class='kw'>{kw.upper()}</span>", highlighted
        )

    return matched, total_score, risk_level, highlighted


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(risk_level: str) -> list:
    actions = {
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
        ],
        "Dangerous": [
            "🚫 Block this User immediately.",
            "📢 Report this User to the platform.",
            "👨‍👩‍👧 Inform Parents / Guardian RIGHT NOW.",
            "📁 Save all Evidence (screenshots, chat logs).",
            "🚔 Contact Cyber Crime Helpline: 1930 (India)",
            "🆘 Contact NCMEC CyberTipline if outside India.",
        ],
    }
    return actions.get(risk_level, [])


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "Hey! Are you coming to school tomorrow? We have a project due.",
        "Can we talk privately? Don't tell your friends about this.",
        "Don't tell your parents. Send me your photo and meet me alone tomorrow. I will share your photos if you don't.",
    ]

    for msg in tests:
        matched, score, level, _ = detect_risks(msg)
        print(f"\nMessage  : {msg[:70]}...")
        print(f"Score    : {score}")
        print(f"Level    : {level}")
        print(f"Keywords : {[kw for kw, _, _ in matched]}")