import streamlit as st
import joblib
import pandas as pd

# Load Model
model = joblib.load("titanic_model.pkl")

st.title("Titanic Survival Prediction")

st.write("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ["Male", "Female"])

age = st.number_input("Age", min_value=0, max_value=100, value=25)

sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)

parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)

fare = st.number_input("Fare", min_value=0.0, value=50.0)

embarked = st.selectbox("Embarked", ["C", "Q", "S"])


# Encoding
sex = 1 if sex == "Male" else 0

if embarked == "C":
    embarked = 0
elif embarked == "Q":
    embarked = 1
else:
    embarked = 2

data = pd.DataFrame(
    [[pclass, sex, age, sibsp, parch, fare, embarked]],
    columns=[
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
)

if st.button("Predict"):

    prediction = model.predict(data)

    probability = model.predict_proba(data)

    if prediction[0] == 1:
        st.success("Passenger Survived ")
    else:
        st.error("Passenger Did Not Survive ")

    survival_prob = probability[0][1] * 100

    st.write("### Survival Probability")
    st.progress(int(survival_prob))
    st.write(f"**{survival_prob:.2f}%** chance of survival")
