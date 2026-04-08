import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

st.title("🚀 My AI Model Deployment")
st.write("Upload an image to see the prediction.")

@st.cache_resource
def load_my_model():
    # 1. Try to load the file directly (Safe for most Colab saves)
    try:
        # We load with weights_only=False here because many Colab models 
        # are saved as full objects rather than just state_dicts
        model = torch.load('my_awesome_model.pth', map_location=torch.device('cpu'))
        # If it's just the 'brain' weights, we need to put it in a skeleton
        if isinstance(model, dict):
            # If this fails, change 'resnet18' to the one you used in Colab
            skeleton = models.resnet18() 
            skeleton.load_state_dict(model, strict=False)
            model = skeleton
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
    model.eval()
    return model

model = load_my_model()

# 2. Upload and Predict Logic
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
    
    st.success("Analysis Complete!")
    st.write("Raw output scores:", output)
