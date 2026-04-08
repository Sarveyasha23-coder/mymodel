import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

st.title("🚀 My AI Model Deployment")
st.write("Upload an image to see the model's prediction.")

# 1. Setup the "Skeleton" 
# IMPORTANT: If you used ResNet50 or another model in Colab, 
# change 'resnet18' to that name below.
@st.cache_resource
def load_my_model():
    model = models.resnet18() 
    
    # 2. Load the "Brain" (Weights)
    # We use map_location='cpu' because Streamlit servers don't always have GPUs
    state_dict = torch.load('my_awesome_model.pth', map_location=torch.device('cpu'))
    
    # strict=False helps ignore minor naming mismatches
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model = load_my_model()

# 3. Image Upload logic
uploaded_file = st.file_bar("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Prepare the image for the model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # 4. Make Prediction
    with torch.no_grad():
        output = model(input_batch)
    
    st.write("### Prediction Complete!")
    st.write(output) # This shows the raw scores; you can refine this later
