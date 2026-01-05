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


## Steps to Run the Project Locally

