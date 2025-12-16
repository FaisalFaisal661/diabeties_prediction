# ⚕️ Diabetes Risk Prediction using Logistic Regression

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diabetiesprediction-dcpsdbg4ztpjmiqq76zchx.streamlit.app/)

This project demonstrates a full-stack Machine Learning pipeline...

# ⚕️ Diabetes Risk Prediction using Logistic Regression

This project demonstrates a full-stack Machine Learning pipeline, from Exploratory Data Analysis (EDA) and model training in a Jupyter Notebook to a professional, interactive web deployment using Streamlit. The goal is to predict the risk of diabetes based on diagnostic measurements from the Pima Indians Diabetes Dataset.

## 🎯 Project Goal

The primary objective is to build a robust **Logistic Regression** classifier to determine whether a patient is at high or low risk of developing diabetes, and to deploy this model as an accessible web application.

## 🚀 Key Features

* **End-to-End ML Pipeline:** Covers data cleaning, feature scaling, modeling, and evaluation.
* **Production-Ready Deployment:** An interactive **Streamlit** application that accepts user input and provides instant, interpretable predictions.
* **Visual Interpretation:** Uses Plotly to generate a **Radar Chart** comparing patient metrics against average healthy and diabetic profiles, enhancing model explainability.
* **Robust Preprocessing:** Handles missing values (erroneous zeros) in key features using median imputation, ensuring reliable predictions.

## 📁 Project Structure

The repository contains the following files:
  ├── diabetes.csv   # The raw dataset
  ├── model.pkl   # Trained Logistic Regression model (Serialized)
  ├── scaler.pkl   # Fitted StandardScaler object (Serialized)
  ├── Diabetes_Analysis.ipynb   # Jupyter Notebook with EDA, training, and evaluation 
  ├── app.py   # Streamlit application code (The web app) └── README.md # This file

## DataSet link
The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
[Link Text](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## 🛠️ Installation and Setup

Follow these steps to set up the project locally.

### 1. Clone the repository.
    ```bash
    git clone <https://github.com/FaisalFaisal661/diabeties_prediction>
    cd diabetes-risk-predictor  
### 2.Create and activate a virtual environment

# Create environment
    python -m venv venv

# Activate environment (Linux/macOS)
    source venv/bin/activate

# Activate environment (Windows)
    .\venv\Scripts\activate

### 3. Install dependencies

    pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly

## 🌐 How to Run the Streamlit Application
The application is run via the app.py file.

Ensure your model files (model.pkl and scaler.pkl) are in the root directory.

Run the Streamlit command from your terminal:

Bash

    streamlit run app.py

The application will open automatically in your browser (usually at http://localhost:8501).

## ✨ Application Demo (Streamlit)
The application features a professional, wide-layout interface with a sidebar for controls and two main visual components:

### 1. Risk Probability Gauge
Provides a clear, instantaneous reading of the predicted diabetes risk.

### 2. Comparative Radar Chart
This interactive chart visualizes the patient's eight health metrics (Blue) against the population averages for both Healthy (Green) and Diabetic (Red) individuals. This allows users to immediately identify which factors are pushing their risk profile toward the diabetic average.

Built with: Python, Jupyter Notebook, Pandas, Scikit-learn, Streamlit, Plotly.
