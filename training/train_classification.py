import numpy as np
import joblib
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from training.data import load_dataset
from features.features import extract_features


# =====================================================
# DATA LOADING
# =====================================================
# Load and preprocess dataset from JSONL file
df = load_dataset("data/problems_data.jsonl")

# Target labels for classification
y = df["problem_class"]


# =====================================================
# TRAIN–TEST SPLIT
# =====================================================
# Stratified split to preserve class distribution
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_train = train_df["problem_class"]
y_test = test_df["problem_class"]


# =====================================================
# TF-IDF FEATURE EXTRACTION (TEXT FEATURES)
# =====================================================
# Convert problem text into TF-IDF feature vectors
tfidf_cls = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_text = tfidf_cls.fit_transform(train_df["text"])
X_test_text = tfidf_cls.transform(test_df["text"])


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
X_train = hstack([X_train_text, X_train_num]).tocsr()
X_test = hstack([X_test_text, X_test_num]).tocsr()


# =====================================================
# CLASSIFICATION MODEL
# =====================================================
# Random Forest Classifier for difficulty class prediction
clf = RandomForestClassifier(
    n_estimators=400,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight={
        "easy": 3.2,
        "medium": 2.8,
        "hard": 1.4
    },
    n_jobs=-1,
    random_state=42
)

# Train the classifier
clf.fit(X_train, y_train)


# =====================================================
# MODEL EVALUATION
# =====================================================
# Predict difficulty classes for test data
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Classification Model Evaluation Results")
print("---------------------------------------")
print(f"Accuracy: {accuracy:.4f}\n")

# Confusion Matrix
labels = ["easy", "medium", "hard"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("Confusion Matrix")
print("Rows: Actual labels | Columns: Predicted labels")
print("Labels:", labels)
print(cm)
print()

# Detailed classification metrics
print("Classification Report")
print(classification_report(y_test, y_pred))


# =====================================================
# MODEL SAVING
# =====================================================
# Save trained models for deployment
joblib.dump(tfidf_cls, "models/tfidf_cls.pkl")
joblib.dump(clf, "models/clf_rf.pkl")

print("Classification model trained and saved successfully.")
