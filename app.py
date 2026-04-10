import streamlit as st
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# --- PAGE CONFIG ---
st.set_page_config(page_title="Deep Learning Built By Sarveyasha Sodhiya", layout="wide")

# --- LOAD MODELS (Cached for performance) ---
@st.cache_resource
def load_churn_model():
    # Ensure you have saved your model using model.save('models/churn_model.h5')
    return load_model('models/churn_model.h5')

@st.cache_resource
def load_accident_model():
    # Ensure architecture matches your AccidentCNN or ResNet50 setup
    # This example assumes the ResNet50 fine-tuned version
    import torchvision
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 2)
    )
    model.load_state_dict(torch.load('models/accident_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Assignment Modules")
app_mode = st.sidebar.selectbox("Choose the project", ["Home", "Bank Churn Prediction", "Accident Detection"])

# --- MODULE 1: HOME ---
if app_mode == "Home":
    st.title("Deep Learning Model")
    st.write("Welcome! Use the sidebar to navigate between the Churn ANN and the Accident Detection CNN.")

# --- MODULE 2: CHURN PREDICTION ---
elif app_mode == "Bank Churn Prediction":
    st.header("Bank Customer Churn Prediction")
    
    # Input Fields
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 600)
        age = st.number_input("Age", 18, 100, 30)
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
    with col2:
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

    if st.button("Predict Churn"):
        # Logic for preprocessing (Encoding/Scaling) must match your training
        # This is a simplified placeholder for the logic
        st.info("Processing inputs and running ANN inference...")
        # prediction = model.predict(processed_input)
        st.success("Result: Customer is likely to Stay (Example)")

# --- MODULE 3: ACCIDENT DETECTION ---
elif app_mode == "Accident Detection":
    st.header("CCTV Accident Detection")
    uploaded_file = st.file_uploader("Upload a CCTV frame...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocessing for PyTorch
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_t = transform(image).unsqueeze(0)
        
        if st.button("Analyze Frame"):
            model = load_accident_model()
            with torch.no_grad():
                output = model(img_t)
                _, pred = torch.max(output, 1)
                labels = ["Accident", "No Accident"]
                result = labels[pred.item()]
                
            if result == "Accident":
                st.error(f"Alert: {result} Detected!")
            else:
                st.success(f"Status: {result}")
