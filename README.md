# AutoJudge: Predicting Programming Problem Difficulty

## Project Overview 
Project Overview
AutoJudge is a machine learning–based system that predicts the difficulty of programming problems using only their textual descriptions.

The system predicts:

Problem Class: Easy / Medium / Hard.

Problem Score: Numerical difficulty score (1–10)

The model does not solve the problem and does not use execution statistics.

It relies purely on the problem text and is inspired by online coding platforms such as Codeforces, Kattis, and CodeChef.

A simple web interface allows users to paste a problem statement and instantly receive predictions.

## Dataset Used
Dataset: TaskComplexityEval-24

Format: JSONL

Source:
https://github.com/AREEG94FAHAD/TaskComplexityEval-24/blob/main/problems_data.jsonl

### Fields Used
problem_description

input_description

output_description

problem_class

problem_score

## Approach
Combine problem description, input description, and output description into a single text

Clean and normalize text

Extract text-based, structural, keyword, and constraint-aware features

Predict difficulty class using a classification model
Predict difficulty score using a regression model

Calibrate the predicted score based on the predicted class

Display results through a web interface


## Model
Classification Model: Random Forest Classifier

Regression Model: Gradient Boosting Regressor

Score Calibration: Class-aware calibration to align scores with difficulty classes

## How the System Works (Diagram)

```bash
----------------------------
│        User Input       │
│ ─────────────────────── │
│  Problem Description    │
│  Input Description      │ 
│  Output Description     │ 
----------------------------
              │
              |
---------------------------
│   Text Preprocessing    │
│ ─────────────────────── │
│  Lowercasing            │
│  Extra space removal    │
│  Duplicate removal      │
│  Label normalization    │
----------------------------
              │
              |
----------------------------
│   Text Combination       │
│ ───────────────────────  │
│ problem + input + output │
│ =single text field      │
----------------------------
              │
              ▼
--------------------------
│   Feature Extraction    │
│ ─────────────────────── │
│  TF-IDF (text features) │
│  Length-based features  │
│  Math & symbol counts   │
│  Constraint indicators  │
│  Keyword frequencies    │
│  Output-type features   │
--------------------------
              │
      ------------------
      │                │
      ▼                ▼
┌────────────────┐  ┌──────────────────────┐
│ Classification │  │     Regression       │
│ Model          │  │     Model            │
│ ────────────── │  │ ───────────────────  │
│ Random Forest  │  │ Gradient Boosting    │
│ Classifier     │  │ Regressor            │
│ (Easy/Med/Hard)│  │ (Raw Score)          │
└───────┬────────┘  └──────────┬───────────┘
        │                       │
        ▼                       ▼
--------------------------------------------
│        Class-Aware Score Calibration     │
│ ───────────────────────────────────────  │
│ • Easy   → 1.0 – 3.5                     │
│ • Medium → 3.5 – 7.0                     │
│ • Hard   → 7.0 – 10.0                    │
└-------------------------------------------
              │
              ▼
----------------------------
│        Final Output      │
│ ───────────────────────  │
│  Difficulty Class       │
│  Calibrated Score       │
----------------------------
              │
              ▼
----------------------------
│   Streamlit Web UI      │
│ ─────────────────────── │
│  Clean display          │
│  Instant prediction     │
---------------------------

```


## Running the Project


### 1.Download the Project

```bash
git clone https://github.com/akhil022007/AutoJudge.git
cd AutoJudge
```
### 2.Create Virtual Environment

Linux / macOS

```bash
python3 -m venv venv
```
Windows
```bash
python -m venv venv
```
### 3.Activate Virtual Environment
Linux / macOS

```bash
source venv/bin/activate
```
Windows
```bash
venv\Scripts\activate
```
### 4.Install Dependencies
```bash
pip install -r requirements.txt
```
### 5.Run the Application
```bash
streamlit run app/run.py
```

## Evaluation
### Classification
Accuracy: ~52%

Regression Performance

### Before Score Calibration
```bash
Metric                   Value
MAE                      1.6994
RMSE                     2.0372
```
### After Score Calibration
```bash
Metric                 Value
MAE                    1.5308
RMSE                   1.9278
```
## Why Score Calibration is Needed
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
## Demo Video Link:

## Report link
https://docs.google.com/document/d/1Y5Gi0a7k08wvvdf_HSAq2I6YLZQjm7oaoU95-tR2cMY/edit?usp=sharing

## About me
   Name :   Ragala Akhil

Enrollment Number: 24114072

Institute: IIT Roorkee

Branch - Computer Science and Engineering


