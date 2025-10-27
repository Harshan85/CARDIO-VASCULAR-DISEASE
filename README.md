❤️ Cardiovascular Disease Prediction using Machine Learning

This project applies machine learning models to predict the likelihood of cardiovascular disease (CVD) using the Cardiovascular Disease dataset.
It demonstrates a full pipeline — from data preprocessing and exploratory analysis to model comparison and performance evaluation using multiple classifiers.

🧠 Project Overview

Cardiovascular diseases are among the leading causes of death worldwide.
By analyzing health and lifestyle indicators such as age, blood pressure, BMI, cholesterol, and glucose, we can build predictive models that help in early detection and prevention.

This project compares multiple ML algorithms to find the best-performing model for predicting whether a patient has cardiovascular disease.

⚙️ Features

🩺 Data Preprocessing: Cleans invalid values, fixes blood pressure inconsistencies, and computes BMI

📊 Exploratory Data Analysis: Visualizes correlations and feature distributions

🧩 Feature Scaling: Standardizes numerical data using StandardScaler

🤖 Model Training: Trains and evaluates multiple models including:

Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

🧾 Model Comparison: Reports accuracy and ROC-AUC for all models

📈 Visualizations: Includes correlation heatmap, boxplots, ROC curves, and confusion matrices

🧰 Technologies Used
Category	Libraries
Data Handling	pandas, numpy
Visualization	matplotlib, seaborn
Machine Learning	scikit-learn
Models Used	SVC, KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
📂 Dataset Information

File: cardio_train.csv (or similar)
Delimiter: ; (semicolon-separated)

Key Columns:
Column	Description
age	Age in days
height, weight	Physical measurements
ap_hi, ap_lo	Systolic and diastolic blood pressure
cholesterol, gluc	Cholesterol and glucose levels (1: normal, 2: above normal, 3: well above normal)
smoke, alco, active	Lifestyle indicators
cardio	Target variable (1 = has disease, 0 = no disease)
🧮 Project Workflow
1️⃣ Data Preprocessing

Converts age from days → years

Removes outliers and unrealistic values

Corrects swapped blood pressure readings (ap_lo > ap_hi)

Creates a new BMI feature

Drops irrelevant columns (id, original age)

2️⃣ Exploratory Data Analysis

Correlation Matrix of all features

Boxplots comparing each feature by target (cardio)

Visual insight into the effect of health indicators

3️⃣ Model Training and Evaluation

Train/test split (80%/20%) with stratification

Standard scaling applied to numeric features

Trains five models:

Logistic Regression

Decision Tree

Random Forest

KNN

SVM

4️⃣ Model Comparison

Each model is evaluated using:

Accuracy

ROC-AUC Score

Classification Report

Results are summarized in a comparison table.

5️⃣ Visualization

Correlation Heatmap

Boxplots for key features

Confusion Matrix (normalized)

ROC Curve for the best-performing model

📈 Example Output

Model Comparison Table:

Model	Accuracy	ROC_AUC
Random Forest	0.736	0.802
Logistic Regression	0.724	0.793
Decision Tree	0.715	0.770
KNN	0.699	0.751
SVM	0.708	0.789

Example Output (Console):

=== Random Forest ===
Accuracy: 0.7362, ROC AUC: 0.8021
              precision    recall  f1-score   support
           0     0.75      0.70      0.72      6878
           1     0.73      0.78      0.75      7122
    accuracy                         0.74     14000
   macro avg     0.74      0.74      0.74     14000
weighted avg     0.74      0.74      0.74     14000

🪄 How to Run
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/cardio-disease-prediction.git
cd cardio-disease-prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Place Dataset

Place your dataset file (cardio_train.csv) in the same folder as the script.
If the file is named differently, update the path in:

df = pd.read_csv("/content/cardio_train (1).csv", sep=';')

4️⃣ Run the Script
python cardio_disease_prediction.py

🧩 Output Files (Optional)
File	Description
best_model.pkl	You can save your best model for later use
plots/	Generated visualizations (heatmaps, ROC, confusion matrices)
🔮 Future Enhancements

Hyperparameter tuning with GridSearchCV

Feature importance ranking

Use of XGBoost / LightGBM for better performance

Deployment using Streamlit for an interactive web app

Integration with real-time health input APIs
