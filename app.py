import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.title("🏦 Bank Customer Churn Predictor")
st.write("Enter customer details to predict if they will leave the bank.")

# 1. Load the Keras Model (.h5)
@st.cache_resource
def load_keras_model():
    # Make sure this name matches the file in your GitHub repo!
    model = tf.keras.models.load_model('best_model.h5')
    return model

model = load_keras_model()

# 2. Create Input Fields for EDA Variables
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)

with col2:
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_card = st.checkbox("Has Credit Card?")
    is_active = st.checkbox("Is Active Member?")
    salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# 3. Prediction Logic
if st.button("Predict Churn"):
    # Prepare input data (Matching your assignment preprocessing)
    # Note: You would normally need the original Scaler object here for accuracy
    input_data = np.array([[credit_score, age, tenure, balance, num_products, 
                            int(has_card), int(is_active), salary]])
    
    # Placeholder for scaling - In a real app, you'd load a saved scaler.pkl
    prediction = model.predict(input_data)
    probability = prediction[0][0]

    if probability > 0.5:
        st.error(f"High Risk: {probability:.2%} chance of churning.")
    else:
        st.success(f"Low Risk: {probability:.2%} chance of staying.")
