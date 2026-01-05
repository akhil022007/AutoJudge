import numpy as np
import joblib
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from training.data import load_dataset
from features.features import extract_features


# =====================================================
# DATA LOADING
# =====================================================
# Load and preprocess the dataset from JSONL file
df = load_dataset("data/problems_data.jsonl")

# Target variable for regression (problem difficulty score)
y = df["problem_score"].values


# =====================================================
# TRAIN–TEST SPLIT
# =====================================================
# Split dataset into training and testing sets
train_df, test_df, y_train, y_test = train_test_split(
    df,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# TF-IDF FEATURE EXTRACTION (TEXT FEATURES)
# =====================================================
# Convert textual problem descriptions into TF-IDF vectors
tfidf_score = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_text = tfidf_score.fit_transform(train_df["text"])
X_test_text = tfidf_score.transform(test_df["text"])


# =====================================================
# NUMERIC FEATURE EXTRACTION
# =====================================================
# Extract handcrafted numeric features from text
X_train_num = np.vstack(train_df["text"].apply(extract_features))
X_test_num = np.vstack(test_df["text"].apply(extract_features))


# =====================================================
# FEATURE COMBINATION
# =====================================================
# Combine sparse TF-IDF features with numeric features
X_train = hstack([X_train_text, X_train_num])
X_test = hstack([X_test_text, X_test_num])


# =====================================================
# REGRESSION MODEL
# =====================================================
# Gradient Boosting Regressor for difficulty score prediction
reg = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Train the regression model
reg.fit(X_train, y_train)


# =====================================================
# MODEL EVALUATION
# =====================================================
y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("Regression Model Evaluation Results")
print("-----------------------------------")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# =====================================================
# MODEL SAVING
# =====================================================
# Save trained models for later inference
joblib.dump(tfidf_score, "models/tfidf_score.pkl")
joblib.dump(reg, "models/reg_score.pkl")

print("\nRegression model trained and saved successfully.")