import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# 1. CORE PAGE SETUP
st.set_page_config(page_title="Multi-Class Retinal Diagnostics Interface", layout="wide")

st.markdown("# 👁️ Multi-Class Retinal Diagnostics Interface")
st.markdown("### Production AI Screening Engine for Diabetic Retinopathy Detection")
st.markdown("---")

# 2. LOAD YOUR PYTORCH MODEL CACHED SAFELY
@st.cache_resource
def load_diagnostic_model():
    # Using ResNet50 as the standard base architecture. 
    # If your model used ResNet18 or another back-bone, change this line!
    model = models.resnet50(weights=None) 
    
    # Change the final fully connected layer to output your 5 structural classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    
    # Load your trained model weights file
    # map_location=torch.device('cpu') ensures it runs smoothly on Streamlit's free servers
    try:
        model.load_state_dict(torch.load("best_model_multiclass.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("⚠️ **Model File Error**: 'best_model_multiclass.pth' was not found in your repository root folder. Please upload it via GitHub.")
    
    model.eval() # Set model to evaluation/inference mode
    return model

model = load_diagnostic_model()

# 3. DEFINE THE IMAGE PREPROCESSING TRANSFORMS
# These must match exactly how you preprocessed your images during training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # Standard ImageNet normalizations
        std=[0.229, 0.224, 0.225]
    )
])

# 5 Classes in exact diagnostic order
CLASS_NAMES = [
    "Normal (No Retinopathy)",
    "Mild Non-proliferative Diabetic Retinopathy (NPDR)",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative Diabetic Retinopathy (PDR)"
]

# 4. SETUP CLEAN SIDE-BY-SIDE INTERFACE MATRIX
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Upload Retinal Fundus Scan Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Retinal Scan", use_container_width=True)
        analyze_btn = st.button("Analyze Retinal Scan", type="primary", use_container_width=True)

with col2:
    st.markdown("### Diagnostic Confidence Levels")
    
    if uploaded_file is not None and 'analyze_btn' in locals() and analyze_btn:
        with st.spinner("Running deep learning model inference..."):
            # A. Preprocess the image and add batch dimension
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0) 
            
            # B. Run image through model calculations
            with torch.no_grad():
                output = model(input_batch)
                # Apply Softmax to convert raw outputs to structural percentage probabilities
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # C. Map probabilities back to classes
            prob_dict = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(5)}
            
            # D. Render the actual real-time scores inside the bars
            for class_name, score in prob_dict.items():
                st.progress(score, text=f"**{class_name}**: {score * 100:.1f}%")
            
            # E. Determine primary assessment alert level based on highest score
            highest_class = max(prob_dict, key=prob_dict.get)
            
            st.markdown("---")
            st.markdown("### Clinical Assessment Notes")
            st.markdown("#### Diagnostic Summary Report")
            st.markdown(f"- **Primary Assessment**: High matching metrics found for **{highest_class}**.")
            st.markdown("- **Execution Environment**: Streamlit Community Cloud (Python Container).")
            
            if highest_class == "Normal (No Retinopathy)":
                st.success("✅ **Screening Indicator**: No immediate signs of diabetic retinopathy detected.")
            else:
                st.error("🚨 **Screening Indicator**: Pathological structures detected matching indicators for Retinopathy.")
                
            st.warning("⚠️ **Clinical Notice**: This application provides screening indicators based on your uploaded file matrix. Please consult an ophthalmologist for definitive clinical validation.")
    else:
        st.info("ℹ️ Upload a valid retinal scan image and click 'Analyze Retinal Scan' to run the clinical confidence processing pipeline.")
