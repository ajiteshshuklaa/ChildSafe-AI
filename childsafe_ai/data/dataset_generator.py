"""
ChildSafe AI – Dataset Loader & Preprocessor
=============================================
Loads the Kaggle Cyberbullying Classification Dataset and prepares it
for training. Maps 6 original labels → 3 classes:
    0 = Safe
    1 = Suspicious
    2 = Dangerous

Place cyberbullying_tweets.csv inside the data/ folder before running.
"""

import os
import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Optional: download NLTK stopwords if not already present ──────────────
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))
except ImportError:
    STOPWORDS = set()
    print("[WARNING] nltk not installed. Stopword removal skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAPPING
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "not_cyberbullying": 0,   # Safe
    "age":               1,   # Suspicious
    "gender":            1,   # Suspicious
    "religion":          1,   # Suspicious
    "ethnicity":         2,   # Dangerous
    "other_cyberbullying": 2, # Dangerous
}

LABEL_NAMES = {0: "Safe", 1: "Suspicious", 2: "Dangerous"}


# ─────────────────────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Clean and normalise a single text string.
    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove mentions (@user) and hashtags (#tag)
        4. Remove punctuation
        5. Remove digits
        6. Remove extra whitespace
        7. Remove stopwords
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove Twitter mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # 4. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 5. Remove digits
    text = re.sub(r"\d+", "", text)

    # 6. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Remove stopwords (keep negations like "not", "no", "never" — important for threat detection)
    keep_negations = {"not", "no", "never", "dont", "dont", "wont", "cant", "shouldnt"}
    if STOPWORDS:
        words = text.split()
        words = [w for w in words if w not in STOPWORDS or w in keep_negations]
        text = " ".join(words)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(csv_path: str = "data/cyberbullying_tweets.csv") -> pd.DataFrame:
    """
    Load and preprocess the Kaggle cyberbullying dataset.

    Args:
        csv_path: path to cyberbullying_tweets.csv

    Returns:
        Cleaned DataFrame with columns: text, label, label_name, clean_text
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {csv_path}\n"
            "Please download 'cyberbullying_tweets.csv' from:\n"
            "  kaggle.com/datasets/andrewmvd/cyberbullying-classification\n"
            "and place it in the data/ folder.\n"
        )

    print(f"[INFO] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # ── Validate expected columns ──────────────────────────────────────────
    expected_cols = {"tweet_text", "cyberbullying_type"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"[ERROR] Expected columns {expected_cols} but found: {list(df.columns)}\n"
            "Make sure you downloaded the correct dataset."
        )

    # ── Rename columns ─────────────────────────────────────────────────────
    df = df.rename(columns={
        "tweet_text":        "text",
        "cyberbullying_type": "original_label",
    })

    # ── Map labels to 0/1/2 ───────────────────────────────────────────────
    df["label"] = df["original_label"].map(LABEL_MAP)

    # Drop rows with unmapped labels
    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    after = len(df)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with unknown labels.")

    # ── Add human-readable label name ─────────────────────────────────────
    df["label_name"] = df["label"].map(LABEL_NAMES)

    # ── Preprocess text ───────────────────────────────────────────────────
    print("[INFO] Preprocessing text (this may take 30–60 seconds)...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Drop empty texts after cleaning
    df = df[df["clean_text"].str.strip().str.len() > 0]

    # ── Shuffle ───────────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[INFO] Dataset loaded: {len(df):,} rows")
    print(f"[INFO] Class distribution:")
    for label_id, name in LABEL_NAMES.items():
        count = (df["label"] == label_id).sum()
        pct = count / len(df) * 100
        print(f"         {name:12s} ({label_id}): {count:,} ({pct:.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split the dataset into 80% train / 20% test.

    Args:
        df        : preprocessed DataFrame from load_dataset()
        test_size : fraction for test set (default 0.20)

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df["clean_text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,          # keeps class balance in both splits
    )

    print(f"\n[INFO] Train/Test split (80/20):")
    print(f"         Training samples : {len(X_train):,}")
    print(f"         Testing  samples : {len(X_test):,}")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify the dataset loads correctly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  ChildSafe AI — Dataset Verification")
    print("=" * 55)

    df = load_dataset("data/cyberbullying_tweets.csv")

    print("\n[INFO] Sample rows:")
    print(df[["text", "label_name"]].head(5).to_string(index=False))

    X_train, X_test, y_train, y_test = get_train_test_split(df)

    print("\n[INFO] Example cleaned text:")
    for i in range(3):
        print(f"\n  Original : {df['text'].iloc[i][:80]}...")
        print(f"  Cleaned  : {df['clean_text'].iloc[i][:80]}...")
        print(f"  Label    : {df['label_name'].iloc[i]}")

    print("\n[SUCCESS] Dataset is ready for model training!")
    print("          Next step: python models/train_evaluate.py")