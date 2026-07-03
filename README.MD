# Titanic Survival Prediction

## 📌 Project Overview
This project predicts whether a passenger would survive the Titanic disaster using Machine Learning. The model is trained on the Titanic dataset using Logistic Regression and deployed with Streamlit.

## 🚀 Features
- Data Cleaning & Preprocessing
- Feature Encoding
- Logistic Regression Model
- Survival Prediction
- Prediction Probability
- Interactive Streamlit Web App

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## 📂 Project Structure

```
Titanic-Survival-Prediction/
│── App.py
│── Titanic_Model.ipynb
│── Titanic_dataset.ipynb
│── cleaned_titanic.csv
│── Titanic.csv
│── titanic_model.pkl
│── requirements.txt
│── README.md
```

## 📊 Input Features
- Passenger Class (Pclass)
- Sex
- Age
- Siblings/Spouses (SibSp)
- Parents/Children (Parch)
- Fare
- Embarked

## 🎯 Model
- Logistic Regression

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run App.py
```

## 📈 Output
The application predicts:
- Passenger Survived ✅
- Passenger Did Not Survive ❌
- Survival Probability (%)

## 👩‍💻 Author
Akriti Mishra
BCA (Machine Learning & Data Science)
Deen Dayal Upadhyaya Gorakhpur University