import joblib
import sys
import os
from scipy.sparse import hstack

# --------------------------------------------------
# Fix project root for imports
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from features.features import extract_features
from app.predict_rf import predict_difficulty


# --------------------------------------------------
# Load models
# --------------------------------------------------
tfidf_score = joblib.load("models/tfidf_score.pkl")
reg_score = joblib.load("models/reg_score.pkl")
calibration = joblib.load("models/score_calibration.pkl")


# --------------------------------------------------
# Target semantic score ranges
# --------------------------------------------------
CLASS_RANGES = {
    "easy":   (1.0, 3.5),
    "medium": (3.5, 7.0),
    "hard":   (7.0, 10.0)
}


# --------------------------------------------------
# Class-aware calibrated alignment
# --------------------------------------------------
def align_score(score: float, predicted_class: str) -> float:
    """
    Align raw regression score using learned
    class-conditional calibration.
    """

    predicted_class = predicted_class.lower()

    # Fallback safety
    if predicted_class not in calibration:
        return float(min(max(score, 1.0), 10.0))

    raw_min = calibration[predicted_class]["raw_min"]
    raw_max = calibration[predicted_class]["raw_max"]

    # Avoid division by zero
    if raw_max <= raw_min:
        return float(min(max(score, 1.0), 10.0))

    # Normalize within learned bounds
    t = (score - raw_min) / (raw_max - raw_min)
    t = min(max(t, 0.0), 1.0)

    lo, hi = CLASS_RANGES[predicted_class]
    return float(lo + t * (hi - lo))


# --------------------------------------------------
# Final prediction function
# --------------------------------------------------
def predict_score(text: str) -> float:
    """
    Predict difficulty score using regression
    + learned class-aware calibration.
    """

    # Feature extraction
    X_text = tfidf_score.transform([text])
    X_num = extract_features(text)
    X = hstack([X_text, X_num])

    # Raw regression
    raw_score = float(reg_score.predict(X)[0])

    # Class prediction
    predicted_class = predict_difficulty(text)

    # Calibrated score
    final_score = align_score(raw_score, predicted_class)

    return final_score
