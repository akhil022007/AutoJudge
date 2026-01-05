# AutoJudge: Predicting Programming Problem Difficulty

## Project Overview
AutoJudge is a machine learning–based system that predicts the difficulty of programming problems using only their textual descriptions. The system classifies problems as Easy, Medium, or Hard and also assigns a numerical difficulty score without solving the problem. A simple web interface is provided to test the model.

---

## Dataset Used
- **Name:** TaskComplexityEval-24  
- **Source:** Public GitHub repository  
- **Link:** https://github.com/AREEG94FAHAD/TaskComplexityEval-24  

---

## Approach and Models Used
- Text descriptions are converted into numerical form using TF-IDF.
- **Classification Model:** Random Forest Classifier (Easy / Medium / Hard)
- **Regression Model:** Gradient Boosting Regressor (difficulty score)
- A class-aware calibration step is applied to improve the final score representation.

---

## Evaluation Metrics
- **Classification Accuracy:** 51.64%
- **Mean Absolute Error (MAE):** 1.6994
- **Root Mean Squared Error (RMSE):** 2.0372

---

## Project Structure

autojudge/
│
├── app/
│ ├── predict_rf.py
│ ├── predict_score.py
│ └── run.py
│
├── data/
│ └── problems_data.jsonl
│
├── features/
│ └── features.py
│
├── training/
│ ├── data.py
│ ├── train_classification.py
│ ├── train_regression.py
│ └── train_score_calibration.py
│
├── models/
│ ├── clf_rf.pkl
│ ├── reg_score.pkl
│ ├── score_calibration.pkl
│ ├── tfidf_cls.pkl
│ └── tfidf_score.pkl
│
├── requirements.txt
└── README.md


---

## Steps to Run the Project Locally

### 1. Clone the Repository
```bash
git clone https://github.com/akhil022007/AutoJudge.git
cd AutoJudge

2. Install Dependencies

Linux / macOS

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Windows

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

3. Run the Application

streamlit run app/run.py

Web Interface

Users can paste a programming problem description into the web interface and click Predict to receive:

    Predicted difficulty class

    Predicted difficulty score



Demo Video Link:


Report Link:


Name: Ragala Akhil
Enrollment Number: 24114072


