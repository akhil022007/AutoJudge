import os
import sys
import numpy as np
import joblib

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from training.data import load_dataset
from features.features import extract_features
from app.predict_rf import predict_difficulty


# ==================================================
# Project Path Setup
# ==================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==================================================
# Model Loading
# ==================================================
tfidf_score = joblib.load("models/tfidf_score.pkl")
reg_score = joblib.load("models/reg_score.pkl")


# ==================================================
# Dataset Loading
# ==================================================
df = load_dataset("data/problems_data.jsonl")

texts = df["text"].values
y_true = df["problem_score"].values


# ==================================================
# Calibration / Evaluation Split
# ==================================================
df_calib, df_eval = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

texts_calib = df_calib["text"].values
y_calib = df_calib["problem_score"].values

texts_eval = df_eval["text"].values
y_eval = df_eval["problem_score"].values


# ==================================================
# Cache Class Predictions
# ==================================================
calib_classes = [predict_difficulty(text) for text in texts_calib]
eval_classes = [predict_difficulty(text) for text in texts_eval]


# ==================================================
# Raw Regression Predictions (Calibration Set)
# ==================================================
raw_calib_preds = []

for text in texts_calib:
    X_text = tfidf_score.transform([text])
    X_num = extract_features(text)
    X = hstack([X_text, X_num])
    raw_calib_preds.append(float(reg_score.predict(X)[0]))

raw_calib_preds = np.array(raw_calib_preds)


# ==================================================
# Learn Calibration Ranges
# ==================================================
class_scores = {"easy": [], "medium": [], "hard": []}

for score, cls in zip(raw_calib_preds, calib_classes):
    if cls in class_scores:
        class_scores[cls].append(score)

calibration = {}

for cls, scores in class_scores.items():
    scores = np.array(scores)
    calibration[cls] = {
        "raw_min": float(np.percentile(scores, 5)),
        "raw_max": float(np.percentile(scores, 95))
    }


# ==================================================
# Raw Regression Predictions (Evaluation Set)
# ==================================================
raw_eval_preds = []

for text in texts_eval:
    X_text = tfidf_score.transform([text])
    X_num = extract_features(text)
    X = hstack([X_text, X_num])
    raw_eval_preds.append(float(reg_score.predict(X)[0]))

raw_eval_preds = np.array(raw_eval_preds)


# ==================================================
# Metrics Before Calibration
# ==================================================
mae_raw = mean_absolute_error(y_eval, raw_eval_preds)
rmse_raw = np.sqrt(mean_squared_error(y_eval, raw_eval_preds))


# ==================================================
# Apply Class-Aware Calibration
# ==================================================
CLASS_RANGES = {
    "easy": (1.0, 3.5),
    "medium": (3.5, 7.0),
    "hard": (7.0, 10.0)
}

adjusted_preds = []

for raw_score, cls in zip(raw_eval_preds, eval_classes):
    if cls not in calibration:
        adjusted_preds.append(raw_score)
        continue

    raw_min = calibration[cls]["raw_min"]
    raw_max = calibration[cls]["raw_max"]

    t = (raw_score - raw_min) / (raw_max - raw_min)
    t = min(max(t, 0.0), 1.0)

    lo, hi = CLASS_RANGES[cls]
    adjusted_preds.append(lo + t * (hi - lo))

adjusted_preds = np.array(adjusted_preds)


# ==================================================
# Metrics After Calibration
# ==================================================
mae_adj = mean_absolute_error(y_eval, adjusted_preds)
rmse_adj = np.sqrt(mean_squared_error(y_eval, adjusted_preds))


# ==================================================
# Results
# ==================================================
print("Regression performance (evaluation set):")
print(f"  Raw model       -> MAE: {mae_raw:.4f}, RMSE: {rmse_raw:.4f}")
print(f"  Calibrated model-> MAE: {mae_adj:.4f}, RMSE: {rmse_adj:.4f}")

print("\nLearned calibration ranges:")
for cls, vals in calibration.items():
    print(f"  {cls.capitalize():6}: raw_min={vals['raw_min']:.2f}, raw_max={vals['raw_max']:.2f}")


# ==================================================
# Save Calibration Model
# ==================================================
os.makedirs("models", exist_ok=True)
joblib.dump(calibration, "models/score_calibration.pkl")
