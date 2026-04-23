import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load('random_forest_model.pkl')

# Page Configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.title("🚢 Titanic Survival Predictor")
st.markdown("### Predicting passenger survival using **Ensemble Learning (Random Forest)**")
st.divider()

# Input Section
st.subheader("Passenger Information")
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=2, help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
    sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
    age = st.slider("Age", 0, 80, 25)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=6, value=0)
    fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=512.0, value=32.0)
    embarked = st.selectbox("Port of Embarkation", options=["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"], index=2)

# Encoding inputs
sex_val = 1 if sex == "Male" else 0
embarked_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
embarked_val = embarked_map[embarked]

# Model expects columns: ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
input_data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]], 
                          columns=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])

# Prediction
if st.button("Predict Survival Result"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    st.divider()
    
    if prediction[0] == 1:
        st.success(f"### Result: Survived! 🎉")
        st.markdown(f"**Confidence Level:** {probability:.2%}")
    else:
        st.error(f"### Result: Did Not Survive 😔")
        st.markdown(f"**Confidence Level:** {(1-probability):.2%}")

# Footer
st.divider()
st.markdown("<p style='text-align: center; color: grey;'>Developed using Random Forest Ensemble Model</p>", unsafe_allow_html=True)
