import streamlit as st
from PIL import Image

# Layout & Page Configuration
st.set_page_config(page_title="Multi-Class Retinal Diagnostics", layout="centered")

st.markdown("# 👁️ Multi-Class Retinal Diagnostics Interface")
st.markdown("### Cloud AI Screening Engine for Diabetic Retinopathy Detection")

# Setup layout columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Upload Retinal Fundus Scan Image")
    uploaded_file = st.file_uploader("Choose a retinal scan image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Retinal Scan", use_column_width=True)
        analyze_btn = st.button("Analyze Retinal Scan", type="primary")

with col2:
    st.markdown("#### Diagnostic Results")
    
    if uploaded_file is not None and 'analyze_btn' in locals() and analyze_btn:
        # Mock confidence scores
        st.markdown("**Confidence Levels:**")
        st.progress(0.85, text="Normal (No Retinopathy): 85%")
        st.progress(0.10, text="Mild NPDR: 10%")
        st.progress(0.05, text="Moderate/Severe NPDR: 5%")
        
        # Clinical Assessment Text block
        st.markdown("---")
        st.markdown("### Diagnostic Summary Report")
        st.markdown("- **Primary Assessment**: Preliminary screening completed successfully.")
        st.markdown("- **Execution Environment**: Cloud Streamlit Runtime Container.")
        st.caption("⚠️ **Clinical Notice**: This application provides screening indicators. Please consult an ophthalmologist for definitive clinical validation.")
    else:
        st.info("Upload an image and click 'Analyze' to generate confidence scores.")
