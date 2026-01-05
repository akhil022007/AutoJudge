import joblib
import sys
import os
from scipy.sparse import hstack

# -----------------------------
# Fix import path 
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from features.features import extract_features


# -----------------------------
# Load trained classifier
# -----------------------------
tfidf_cls = joblib.load("models/tfidf_cls.pkl")
clf_rf = joblib.load("models/clf_rf.pkl")


def predict_difficulty(text: str) -> str:
    """
    Predict difficulty class: easy / medium / hard
    """

    # TF-IDF features
    X_text = tfidf_cls.transform([text])

    # Numeric features
    X_num = extract_features(text)

    # Combine
    X = hstack([X_text, X_num])

    return clf_rf.predict(X)[0]
