AutoJudge: Predicting Programming Problem Difficulty
Project Overview
AutoJudge is a machine learning–based system that predicts the difficulty of programming problems using only their textual descriptions.
The system predicts:
Problem Class: Easy / Medium / Hard
Problem Score: Numerical difficulty score (1–10)

The model does not solve the problem and does not use execution statistics.

It relies purely on the problem text and is inspired by online coding platforms such as Codeforces, Kattis, and CodeChef.
A simple web interface allows users to paste a problem statement and instantly receive predictions.
Dataset Used
Dataset: TaskComplexityEval-24
Format: JSONL
Source:
https://github.com/AREEG94FAHAD/TaskComplexityEval-24/blob/main/problems_data.jsonl

Fields Used
problem_description
input_description
output_description
problem_class
problem_score

Approach
Combine problem description, input description, and output description into a single text
Clean and normalize text
Extract text-based, structural, keyword, and constraint-aware features
Predict difficulty class using a classification model
Predict difficulty score using a regression model
Calibrate the predicted score based on the predicted class
Display results through a web interface

Model
Classification Model: Random Forest Classifier
Regression Model: Gradient Boosting Regressor
Score Calibration: Class-aware calibration to align scores with difficulty classes

How the System Works (Diagram)

Problem Text
     |
Text Preprocessing
     |
Feature Extraction
     |
Difficulty Class Prediction
     |
Difficulty Score Prediction
     |
Score Calibration
     |
Final Output



Running the Project
Step 1: Download the Project


git clone https://github.com/<your-username>/AutoJudge.git
cd AutoJudge



Step 2: Create Virtual Environment
Linux / macOS


python3 -m venv venv



Windows

python -m venv venv



Step 3: Activate Virtual Environment
Linux / macOS



source venv/bin/activate



Windows



venv\Scripts\activate



Step 4: Install Dependencies


pip install -r requirements.txt



Step 5: Run the Application

streamlit run app.py



Evaluation
Classification
Accuracy: ~52%

Regression Performance
Before Score Calibration
Metric                   Value
MAE                          1.6994
RMSE                       2.0372
After Score Calibration
Metric                 Value
MAE                   1.5308
RMSE                 1.9278
Why Score Calibration is Needed
The regression model predicts difficulty scores in a compressed numerical range, even for problems of very different difficulty levels.

This causes overlap between Easy, Medium, and Hard problems.
To address this:
The predicted difficulty class is used to guide score scaling
Raw scores are mapped to class-specific ranges
The relative order of scores is preserved while improving interpretability

This calibration step improves both semantic correctness and numerical accuracy, as reflected in the reduced MAE and RMSE.
Web Interface Explanation
Users paste the problem text into input fields
Click Predict
The system displays:
Predicted difficulty class
Predicted difficulty score


The same preprocessing and feature extraction pipeline used during training is reused during prediction.
Demo Video
Demo Video Link:





Project Report
 Report Link:
https://docs.google.com/document/d/1Y5Gi0a7k08wvvdf_HSAq2I6YLZQjm7oaoU95-tR2cMY/edit?usp=sharing
About Me
Ragala Akhil

Enrollment Number: 24114072

Institute: IIT Roorkee
