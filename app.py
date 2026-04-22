import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

# Data and Image Processing
import numpy as np
from PIL import Image
import tifffile

# Visualization and PDF Generation
import plotly.express as px
from fpdf import FPDF

# Internal project modules
from utils.image_processor import MedicalImageProcessor
from core.model_handler import MedGemmaHandler

# Load environment variables (e.g., HF_TOKEN)
load_dotenv()

# --- Page Configuration ---
# Setting the UI layout and title for the browser tab
st.set_page_config(
    page_title="Medora | Clinical Image Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Application ---
# Injecting custom CSS for the clinical dashboard look
if os.path.exists("assets/styles.css"):
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Application State Management ---
# Using session state to persist data across UI interactions
if "current_report" not in st.session_state:
    st.session_state.current_report = ""
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []
if "expert_sign_off" not in st.session_state:
    st.session_state.expert_sign_off = False

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-doctor.png", width=80)
    st.title("Medora Control Center")
    st.markdown("---")
    
    st.subheader("🛠 System Settings")
    hf_token = st.text_input(
        "Hugging Face Access Token", 
        value=os.getenv("HF_TOKEN", ""),
        type="password", 
        help="Secure token required for local model weights."
    )
    
    scan_modality = st.selectbox("Scan Modality", ["CT Scan", "MRI Scan", "Mammography"])
    
    # Demo mode allows for testing the UI without loading the heavy LLM
    use_demo_mode = st.checkbox(
        "Demo Mode (Mock AI)", 
        value=False, 
        help="Use pre-generated responses for presentation purposes."
    )
    
    st.markdown("---")
    st.info("Performance Note: Optimized for local CPU inference using Qwen2-VL.")

# --- Document Generation Logic ---
def generate_clinical_pdf(report_text, scan_type, expert_name, images):
    """
    Creates a professional, branded medical report in PDF format.
    Ensures that long text does not overlap with the digital signature.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Brand Colors (Professional Medical Palette)
    PRIMARY_BLUE = (25, 55, 100)
    LIGHT_BG = (230, 235, 245)
    DARK_TEXT = (30, 30, 30)

    # 1. Formatting: Remove markdown symbols from the AI response for the PDF
    clean_report = report_text.replace("###", "").replace("**", "").replace("•", "-")
    safe_encoded_text = clean_report.encode("latin-1", "ignore").decode("latin-1")

    # 2. Header: Branded Bar
    pdf.set_fill_color(*PRIMARY_BLUE)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", 'B', 16)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "MEDORA DIAGNOSTICS REPORT", ln=True)

    # 3. Metadata Section: Patient/Scan Details
    pdf.set_text_color(*DARK_TEXT)
    pdf.set_font("Helvetica", '', 10)
    pdf.ln(12)
    pdf.set_fill_color(*LIGHT_BG)
    pdf.rect(10, 35, 190, 22, 'F')
    pdf.set_xy(12, 38)
    pdf.cell(60, 6, f"Modality: {scan_type}", ln=0)
    pdf.cell(70, 6, f"Date: {datetime.now().strftime('%d-%m-%Y')}", ln=0)
    pdf.cell(0, 6, f"ID: NX-{datetime.now().strftime('%Y%m%d%H%M')}", ln=1)
    pdf.set_x(12)
    pdf.cell(0, 6, f"Verified By: Dr. {expert_name}", ln=True)

    # 4. Visual Data: Centered Scan Preview
    if images:
        pdf.ln(12)
        pdf.set_font("Helvetica", 'B', 12)
        pdf.set_text_color(*PRIMARY_BLUE)
        pdf.cell(0, 8, "IMAGING DATA", ln=True)
        
        # Save temp image for PDF insertion
        temp_path = "temp_report_img.png"
        images[0].save(temp_path)
        pdf.image(temp_path, x=55, w=100) # Width reduced to prevent overflow
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 5. Clinical Findings: Wrapped Text
    pdf.ln(10)
    pdf.set_text_color(*PRIMARY_BLUE)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, "CLINICAL FINDINGS", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_text_color(*DARK_TEXT)
    pdf.set_font("Helvetica", '', 11)
    pdf.multi_cell(0, 6, safe_encoded_text)

    # 6. Footer & Signature Logic
    # If the report is too long, we move the signature to a new page
    if pdf.get_y() > 240:
        pdf.add_page()
    
    pdf.set_y(-35) # Pin footer to the bottom
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_font("Helvetica", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Report generated via Medora AI. Clinical validation is mandatory.", ln=True, align='C')
    
    pdf.set_font("Helvetica", 'B', 10)
    pdf.set_text_color(*PRIMARY_BLUE)
    pdf.cell(0, 6, f"Digitally Signed: Dr. {expert_name}", ln=True, align='R')

    return bytes(pdf.output())

# --- Main Dashboard UI ---
st.title("🩺 Medora Clinical Image Analysis")
st.write("Advanced multimodal diagnostic support using local CPU inference.")

upload_col, preview_col = st.columns([1, 1], gap="large")

with upload_col:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.subheader("📁 Imaging Intake")
    
    # Supporting multiple medical and standard formats
    allowed_formats = ["dcm", "nii", "nii.gz", "jpg", "jpeg", "png", "tiff", "tif"]
    files = st.file_uploader(f"Upload {scan_modality} files", type=allowed_formats, accept_multiple_files=True)

    if files:
        st.session_state.processed_images = []
        for f in files:
            ext = f.name.lower()
            
            # DICOM/NIfTI Handling
            if ext.endswith((".dcm", ".nii", ".gz")):
                # Logic handled by MedicalImageProcessor utility
                with open("temp_medical_file", "wb") as tmp:
                    tmp.write(f.read())
                # Note: Assuming utility returns PIL images
                processed = MedicalImageProcessor.prepare_any_image(f.read())
                if processed: st.session_state.processed_images.append(processed)
            
            # TIFF Handling for High-Resolution scans
            elif ext.endswith((".tif", ".tiff")):
                raw_tiff = tifffile.imread(f)
                # Slice selection if 3D, otherwise normalize to 8-bit
                if len(raw_tiff.shape) == 3: raw_tiff = raw_tiff[raw_tiff.shape[0] // 2]
                norm_img = ((raw_tiff - raw_tiff.min()) / (raw_tiff.max() - raw_tiff.min()) * 255).astype(np.uint8)
                st.session_state.processed_images.append(Image.fromarray(norm_img))
            
            # Standard Web Formats
            else:
                img = MedicalImageProcessor.prepare_any_image(f.read())
                if img: st.session_state.processed_images.append(img)

        # UI Preview for the uploaded slices
        if st.session_state.processed_images:
            imgs = st.session_state.processed_images[:5] # Limit preview to first 5
            if len(imgs) > 1:
                idx = st.slider("Browse Slices", 0, len(imgs)-1, 0)
                st.image(imgs[idx], caption=f"Selected Slice: {idx+1}", use_container_width=True)
            else:
                st.image(imgs[0], caption="Scan Preview", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with preview_col:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Diagnostic Suite")
    
    can_analyze = len(st.session_state.processed_images) > 0
    if st.button("RUN AI DIAGNOSTICS", disabled=not can_analyze):
        with st.spinner("Analyzing voxel patterns and clinical markers..."):
            handler = MedGemmaHandler()
            try:
                if use_demo_mode:
                    report = handler.mock_analyze(scan_modality)
                else:
                    handler.initialize(hf_token=hf_token)
                    report = handler.analyze(st.session_state.processed_images[0], scan_modality)
                st.session_state.current_report = report
            except Exception as e:
                st.error(f"Analysis Failed: {str(e)}")

    if st.session_state.current_report:
        st.success("Analysis Complete")
        st.info(st.session_state.current_report)
        
        # --- Interactive Technical Analysis Tabs ---
        tab1, tab2 = st.tabs(["📊 Densitometry Analysis", "📈 Intensity Distribution"])
        
        with tab1:
            st.write("**Radiodensity Profile (Hounsfield Units)**")
            handler = MedGemmaHandler()
            hu_data = handler.get_intensity_analysis(st.session_state.processed_images[0])
            
            if hu_data is not None:
                # Plotly Histogram for interactive exploration
                fig = px.histogram(
                    x=hu_data, nbins=80, range_x=[-1050, 1050],
                    labels={'x': 'HU Scale', 'y': 'Frequency'},
                    color_discrete_sequence=['#00d4ff']
                )
                
                # Clinical markers to guide the physician
                for label, pos, color in [("Air", -1000, "Gray"), ("Lung", -500, "Cyan"), ("Soft Tissue", 40, "Green")]:
                    fig.add_vline(x=pos, line_dash="dot", line_color=color, annotation_text=label)
                
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.write("**Voxel Frequency Analysis**")
            # Simplified NumPy-based distribution view
            hist_data = np.histogram(np.array(st.session_state.processed_images[0]), bins=50)
            st.bar_chart(hist_data[0])

        st.markdown("---")
        st.subheader("✍️ Clinical Validation")
        dr_name = st.text_input("Reviewing Clinician Name")
        signed = st.checkbox("I verify the above clinical findings.")
        
        if signed and dr_name:
            pdf_data = generate_clinical_pdf(st.session_state.current_report, scan_modality, dr_name, st.session_state.processed_images)
            st.download_button(
                label="📥 EXPORT OFFICIAL PDF",
                data=pdf_data,
                file_name=f"Medora_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

    st.markdown('</div>', unsafe_allow_html=True)

# --- Application Footer ---
st.markdown("---")
st.caption("Medora AI Framework v1.0 | Local Multimodal Reasoning | Project Demo 2026")
