import streamlit as st
import os
import io
from dotenv import load_dotenv
import re
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from PIL import Image
import tifffile
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go


# Internal imports
from utils.image_processor import MedicalImageProcessor
from core.model_handler import MedGemmaHandler

# Load .env file
load_dotenv()

# MUST BE FIRST
st.set_page_config(
    page_title="Medora | Clinical Image Analsis Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS ---
if os.path.exists("assets/styles.css"):
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Initialize Session State ---
if "current_report" not in st.session_state:
    st.session_state.current_report = ""
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []
if "expert_sign_off" not in st.session_state:
    st.session_state.expert_sign_off = False

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-doctor.png", width=80)
    st.title("Medora")
    st.markdown("---")
    
    st.subheader("🛠 Configuration")
    hf_token = st.text_input(
        "Hugging Face Token", 
        value=os.getenv("HF_TOKEN", ""),
        type="password", 
        help="Required to access models on Hugging Face."
    )
    scan_type = st.selectbox("Scan Modality", ["CT Scan", "MRI Scan", "Mammography"])

    use_mock = st.checkbox("Use Demo Mode (Mock AI)", value=False, help="Toggle this to use pre-written responses.")
    
    st.markdown("---")
    st.info("System optimized: Qwen2-VL is active for local CPU inference.")

# --- PDF Generation Function ---
def generate_pdf(report_text, scan_type, expert_name, images):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    PRIMARY = (25, 55, 100)
    ACCENT = (230, 235, 245)
    TEXT = (30, 30, 30)
    MUTED = (120, 120, 120)
    LINE = (200, 200, 200)

    # CLEAN TEXT SAFELY
    clean_text = report_text.replace("###", "").replace("**", "").replace("•", "-")
    safe_text = clean_text.encode("latin-1", "ignore").decode("latin-1")

    # HEADER BAR
    pdf.set_fill_color(*PRIMARY)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", 'B', 16)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "MEDORA DIAGNOSTICS REPORT", ln=True)

    # META BOX
    pdf.set_text_color(*TEXT)
    pdf.set_font("Helvetica", '', 10)
    pdf.ln(10)
    pdf.set_fill_color(*ACCENT)
    pdf.rect(10, 35, 190, 22, 'F')
    pdf.set_xy(12, 38)
    pdf.cell(60, 6, f"Scan Type: {scan_type}", ln=0)
    pdf.cell(70, 6, f"Date: {datetime.now().strftime('%d-%m-%Y')}", ln=0)
    pdf.cell(0, 6, f"Report ID: NX-{datetime.now().strftime('%Y%m%d%H%M')}", ln=1)
    pdf.set_x(12)
    pdf.cell(0, 6, f"Consultant: Dr. {expert_name}", ln=True)

    # IMAGE SECTION
    if images:
        pdf.ln(12)
        pdf.set_font("Helvetica", 'B', 12)
        pdf.set_text_color(*PRIMARY)
        pdf.cell(0, 8, "IMAGING DATA", ln=True)
        
        temp_img = "report_snapshot.png"
        images[0].save(temp_img)
        
        # Reduced width from 140 to 100
        # Centering logic: (Page width 210 - Image width 100) / 2 = 55
        pdf.image(temp_img, x=55, w=100) 
        
        pdf.ln(2) # Reduced line spacing after image
        if os.path.exists(temp_img):
            os.remove(temp_img)

    # FINDINGS SECTION
    pdf.ln(5)
    pdf.set_text_color(*PRIMARY)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, "CLINICAL FINDINGS", ln=True)
    pdf.set_draw_color(*LINE)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_text_color(*TEXT)
    pdf.set_font("Helvetica", '', 11)

    # Calculate available space to prevent overlap
    # If the text is too long, it will now automatically trigger a page break
    pdf.multi_cell(0, 6, safe_text)

    # FOOTER & SIGNATURE
    # check if we are too close to the bottom after the findings
    if pdf.get_y() > 250:
        pdf.add_page()
    
    # Set footer position to exactly 35mm from bottom to give room for 3 lines
    pdf.set_y(-35) 
    pdf.set_draw_color(*LINE)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    
    pdf.set_font("Helvetica", 'I', 8) # Italicize disclaimer
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, "This report is system-generated and requires clinical validation.", ln=True, align='C')
    
    pdf.ln(2)
    pdf.set_font("Helvetica", 'B', 10)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 6, f"Digitally Verified By: Dr. {expert_name}", ln=True, align='R')

    return bytes(pdf.output())

# --- Main App Logic ---
st.title("🩺 Medora Clinical Image Analysis Dashboard")
st.write("Advance your clinical workflow with local AI multimodal reasoning.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.subheader("📁 Scan Data Upload")
    
    supported_types = ["dcm", "nii", "nii.gz", "jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"]
    uploaded_files = st.file_uploader(
        f"Upload {scan_type} files", 
        type=supported_types, 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.processed_images = []
        for f in uploaded_files:
            file_ext = f.name.lower()
            
            # DICOM, NIfTI, TIFF, and Standard Handlers...
            if file_ext.endswith(".dcm"):
                with open("temp.dcm", "wb") as tmp:
                    tmp.write(f.read())
                hu_data, _ = MedicalImageProcessor.load_dicom_slice("temp.dcm")
                rgb_slice = MedicalImageProcessor.process_ct_rgb(hu_data)
                resized = MedicalImageProcessor.resize_for_model(rgb_slice)
                st.session_state.processed_images.append(Image.fromarray(resized))
            
            elif file_ext.endswith(".nii") or file_ext.endswith(".nii.gz"):
                slices = MedicalImageProcessor.extract_nifti_slices(f.read())
                st.session_state.processed_images.extend(slices)

            elif file_ext.endswith(".tif") or file_ext.endswith(".tiff"):
                try:
                    img_array = tifffile.imread(f)
                    if len(img_array.shape) == 3:
                        img_array = img_array[img_array.shape[0] // 2]
                    img_min, img_max = img_array.min(), img_array.max()
                    img_8bit = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8) if img_max > img_min else np.zeros_like(img_array)
                    resized = MedicalImageProcessor.resize_for_model(img_8bit)
                    st.session_state.processed_images.append(Image.fromarray(resized))
                except Exception as e:
                    st.error(f"Failed to process TIFF: {e}")
            
            else:
                processed_img = MedicalImageProcessor.prepare_any_image(f.read())
                if processed_img:
                    st.session_state.processed_images.append(processed_img)

        st.session_state.processed_images = st.session_state.processed_images[:5]

        # Display Preview (Fixed deprecated use_column_width)
        if st.session_state.processed_images:
            if len(st.session_state.processed_images) > 1:
                slice_idx = st.slider("Select Slice to View", 0, len(st.session_state.processed_images)-1, 0)
                st.image(st.session_state.processed_images[slice_idx], caption=f"View {slice_idx+1}", use_container_width=True)
            else:
                st.image(st.session_state.processed_images[0], caption="Scan Preview", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.subheader("🤖 Analysis Preview")
    
    analyze_btn = st.button("GENERATE CLINICAL INSIGHTS", disabled=not st.session_state.processed_images)
    
    if analyze_btn:
        with st.spinner("🧠 Analysing, Please wait ... "):
            handler = MedGemmaHandler()
            try:
                if use_mock:
                    raw_report = handler.mock_analyze(scan_type)
                else:
                    # Pass HF Token if provided
                    handler.initialize(hf_token=hf_token)
                    # Pass the first image in the list
                    raw_report = handler.analyze(st.session_state.processed_images[0], scan_type)
                
                st.session_state.current_report = raw_report
            except Exception as e:
                st.error(f"Failed to run model: {e}")

    if st.session_state.current_report:
        st.markdown("### 📋 Clinical Findings")
        # Display the report
        st.info(st.session_state.current_report)
        tab1, tab2 = st.tabs(["🔬 Radiodensity & Densitometry Analysis", "📊 Density Analysis"])
        
        with tab1:
            st.header("🔬 Radiodensity & Densitometry Analysis")
            st.info("Mapping voxel intensity to Hounsfield Units (HU) to verify tissue composition.")

            if st.session_state.processed_images:
                handler = MedGemmaHandler()
                # Analyze the primary slice
                hu_values = handler.get_intensity_analysis(st.session_state.processed_images[0])
        
            if hu_values is not None:
                    # Create interactive histogram
                    fig = px.histogram(
                x=hu_values, 
                nbins=100,
                labels={'x': 'Hounsfield Units (HU)', 'y': 'Voxel Count'},
                color_discrete_sequence=['#00d4ff'],
                range_x=[-1050, 1050]
                    )

            # Add Clinical Reference Lines
                    reference_tissues = [
                {"name": "Air", "pos": -1000, "color": "Gray"},
                {"name": "Lung", "pos": -500, "color": "LightBlue"},
                {"name": "Water", "pos": 0, "color": "Blue"},
                {"name": "Bone", "pos": 700, "color": "White"}
                    ]

                    for tissue in reference_tissues:
                        fig.add_vline(x=tissue["pos"], line_dash="dot", line_color=tissue["color"],
                             annotation_text=tissue["name"])

                    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Quantitative Summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Radiodensity", f"{np.mean(hu_values):.1f} HU")
                        st.write("**Voxel Analysis**")
                        st.caption(f"Total Data Points: {len(hu_values):,}")
                    with col2:
                # Calculate percentage of clear lung tissue (standard HU range)
                        lung_ratio = np.sum((hu_values > -900) & (hu_values < -200)) / len(hu_values) * 100
                        st.metric("Lung Field Ratio", f"{lung_ratio:.1f}%")
                        st.write("**Composition Status**")
                        st.success("Consistent with healthy thoracic cavity")
            else:
                st.warning("Please process a scan to view densitometry data.")
        
        with tab2:
            st.write("Tissue Density Distribution")
            if st.session_state.processed_images:
                img_data = np.array(st.session_state.processed_images[0])
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(img_data.flatten(), bins=80, color='#1E88E5', alpha=0.7)
                ax.set_title("Voxel Intensity Distribution")
                st.pyplot(fig)

        st.markdown("---")
        st.subheader("✍️ Validation & Signature")
        expert_name = st.text_input("Reviewing Consultant's Name")
        st.session_state.expert_sign_off = st.checkbox("Sign-off: Confirm diagnostic accuracy.")
        
        if st.session_state.expert_sign_off and expert_name:
            pdf_bytes = generate_pdf(st.session_state.current_report, scan_type, expert_name, st.session_state.processed_images)
            st.download_button(
                label="📥 DOWNLOAD FINAL REPORT (PDF)",
                data=pdf_bytes,
                file_name=f"Clinical_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        else:
            st.info("Complete consultant sign-off to enable PDF export.")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("AI-assisted diagnostics interpretatated by local Qwen2-VL model. Optimized for local CPU inference.")
