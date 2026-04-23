import streamlit as st
import os
import base64
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from PIL import Image
import tifffile
from fpdf import FPDF
import plotly.express as px

from utils.image_processor import MedicalImageProcessor
from core.model_handler import MedoraHandler

load_dotenv()

st.set_page_config(
    page_title="Medora | Clinical Imaging",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

def file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_path   = "assets/background.jpeg"
logo_path = "assets/logo.png"
 
bg_b64   = file_to_b64(bg_path)   if os.path.exists(bg_path)   else None
logo_b64 = file_to_b64(logo_path) if os.path.exists(logo_path) else None
 
# ── Load CSS ───────────────────────────────────────────────────────────────
css = ""
if os.path.exists("assets/styles.css"):
    with open("assets/styles.css") as f:
        css = f.read()
 
# # Inject background dynamically
# if bg_b64:
#     css += f"""
# .stApp {{
#     background-image: url("data:image/jpeg;base64,{bg_b64}") !important;
#     background-size: cover !important;
#     background-position: center !important;
#     background-attachment: fixed !important;
# }}
# """
 
# st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# if os.path.exists("assets/styles.css"):
#     with open("assets/styles.css") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.session_state.setdefault("current_report", "")
st.session_state.setdefault("processed_images", [])
st.session_state.setdefault("expert_sign_off", False)


def generate_pdf(report_text, scan_type, expert_name, images):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    PRIMARY = (15, 40, 80)
    ACCENT  = (235, 240, 250)
    TEXT    = (30, 30, 30)
    MUTED   = (110, 110, 110)
    LINE    = (210, 215, 225)
    clean = report_text.replace("###", "").replace("**", "").replace("•", "-")
    safe  = clean.encode("latin-1", "ignore").decode("latin-1")
    pdf.set_fill_color(*PRIMARY)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "MEDORA DIAGNOSTICS REPORT", ln=True)
    pdf.set_text_color(*TEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(10)
    pdf.set_fill_color(*ACCENT)
    pdf.rect(10, 35, 190, 22, "F")
    pdf.set_xy(12, 38)
    pdf.cell(60, 6, f"Scan Type: {scan_type}", ln=0)
    pdf.cell(70, 6, f"Date: {datetime.now().strftime('%d-%m-%Y')}", ln=0)
    pdf.cell(0,  6, f"Report ID: NX-{datetime.now().strftime('%Y%m%d%H%M')}", ln=1)
    pdf.set_x(12)
    pdf.cell(0, 6, f"Consultant: Dr. {expert_name}", ln=True)
    if images:
        pdf.ln(12)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*PRIMARY)
        pdf.cell(0, 8, "IMAGING DATA", ln=True)
        tmp = "report_snapshot.png"
        images[0].save(tmp)
        pdf.image(tmp, x=55, w=100)
        pdf.ln(2)
        if os.path.exists(tmp):
            os.remove(tmp)
    pdf.ln(5)
    pdf.set_text_color(*PRIMARY)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "CLINICAL FINDINGS", ln=True)
    pdf.set_draw_color(*LINE)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_text_color(*TEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, safe)
    if pdf.get_y() > 250:
        pdf.add_page()
    pdf.set_y(-35)
    pdf.set_draw_color(*LINE)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, "This report is system-generated and requires clinical validation.", ln=True, align="C")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 6, f"Digitally Verified By: Dr. {expert_name}", ln=True, align="R")
    return bytes(pdf.output())


with st.sidebar:
    if logo_b64:
        st.markdown(
            f"""
            <div class="sidebar-brand" style="margin-bottom:0;">
                <img src="data:image/png;base64,{logo_b64}" class="brand-logo" width="20%" style="margin-bottom:8%; margin-right:2%" alt="Medora logo"/>
                <span style="font-size:40px; font-weight:600; text-align:center;" class="brand-name" >Medora</span>
            </div>
            <p class="brand-tagline" style="font-size:20px; margin-left: 5%; margin-top: 0;">Agentic AI for Clinic Imaging</p>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("## 🩺 Medora")
        st.caption("Clinical Imaging Intelligence")

    st.divider()
    st.markdown("**Configuration**")
    hf_token = st.text_input(
        "Hugging Face Token",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        help="Required to access models on Hugging Face.",
    )
    scan_type = st.selectbox("Scan Modality", ["CT Scan", "MRI Scan", "Mammography"])
    use_mock = st.toggle("Demo Mode", value=False, help="Use pre-written mock responses instead of live AI inference.")
    st.divider()
    if use_mock:
        st.info("🟡 Demo mode active")
    else:
        st.success("🟢 Qwen2-VL · CPU inference")


st.markdown("# Clinical Image Analysis")
st.caption("AI-assisted diagnostics · Local inference · Expert sign-off workflow")
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("#### 📁 Scan Upload")
    supported = ["dcm", "nii", "nii.gz", "jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"]
    uploaded_files = st.file_uploader(
        f"Upload {scan_type} files",
        type=supported,
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.session_state.processed_images = []
        for f in uploaded_files:
            name = f.name.lower()
            if name.endswith(".dcm"):
                with open("temp.dcm", "wb") as tmp:
                    tmp.write(f.read())
                hu_data, _ = MedicalImageProcessor.load_dicom_slice("temp.dcm")
                rgb_slice  = MedicalImageProcessor.process_ct_rgb(hu_data)
                resized    = MedicalImageProcessor.resize_for_model(rgb_slice)
                st.session_state.processed_images.append(Image.fromarray(resized))
                if os.path.exists("temp.dcm"):
                    os.remove("temp.dcm")
            elif name.endswith(".nii") or name.endswith(".nii.gz"):
                slices = MedicalImageProcessor.extract_nifti_slices(f.read())
                st.session_state.processed_images.extend(slices)
            elif name.endswith(".tif") or name.endswith(".tiff"):
                try:
                    arr = tifffile.imread(f)
                    if len(arr.shape) == 3:
                        arr = arr[arr.shape[0] // 2]
                    mn, mx = arr.min(), arr.max()
                    arr_8 = ((arr - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(arr)
                    resized = MedicalImageProcessor.resize_for_model(arr_8)
                    st.session_state.processed_images.append(Image.fromarray(resized))
                except Exception as e:
                    st.error(f"Failed to process TIFF: {e}")
            else:
                processed = MedicalImageProcessor.prepare_any_image(f.read())
                if processed:
                    st.session_state.processed_images.append(processed)
        st.session_state.processed_images = st.session_state.processed_images[:5]

    if st.session_state.processed_images:
        imgs = st.session_state.processed_images
        if len(imgs) > 1:
            idx = st.slider("Slice viewer", 0, len(imgs) - 1, 0)
            st.image(imgs[idx], caption=f"Slice {idx + 1} of {len(imgs)}", use_container_width=True)
        else:
            st.image(imgs[0], caption="Scan preview", use_container_width=True)
        st.caption(f"✓ {len(imgs)} slice(s) loaded · {scan_type}")

with col_right:
    st.markdown("#### 🤖 AI Analysis")
    analyze_btn = st.button(
        "Generate Clinical Insights →",
        disabled=not st.session_state.processed_images,
        use_container_width=True,
    )
    if analyze_btn:
        with st.spinner("Analysing scan — please wait…"):
            handler = MedoraHandler()
            try:
                if use_mock:
                    raw_report = handler.mock_analyze(scan_type)
                else:
                    handler.initialize(hf_token=hf_token)
                    raw_report = handler.analyze(st.session_state.processed_images[0], scan_type)
                st.session_state.current_report = raw_report
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    if st.session_state.current_report:
        st.markdown("##### 📋 Findings")
        st.info(st.session_state.current_report)

        tab_density, tab_histogram = st.tabs(["Radiodensity Analysis", "Intensity Distribution"])

        with tab_density:
            if st.session_state.processed_images:
                handler   = MedoraHandler()
                hu_values = handler.get_intensity_analysis(st.session_state.processed_images[0])
                if hu_values is not None:
                    fig = px.histogram(
                        x=hu_values, nbins=100,
                        labels={"x": "Hounsfield Units (HU)", "y": "Voxel Count"},
                        color_discrete_sequence=["#3b82f6"],
                        range_x=[-1050, 1050],
                    )
                    for tissue in [
                        {"name": "Air",   "pos": -1000, "color": "#94a3b8"},
                        {"name": "Lung",  "pos": -500,  "color": "#7dd3fc"},
                        {"name": "Water", "pos": 0,     "color": "#60a5fa"},
                        {"name": "Bone",  "pos": 700,   "color": "#e2e8f0"},
                    ]:
                        fig.add_vline(x=tissue["pos"], line_dash="dot", line_color=tissue["color"],
                                      annotation_text=tissue["name"], annotation_font_size=11)
                    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                                      paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    m1, m2 = st.columns(2)
                    m1.metric("Mean Radiodensity", f"{np.mean(hu_values):.1f} HU")
                    lung_pct = np.sum((hu_values > -900) & (hu_values < -200)) / len(hu_values) * 100
                    m2.metric("Lung Field Ratio", f"{lung_pct:.1f}%")

        with tab_histogram:
            if st.session_state.processed_images:
                img_data = np.array(st.session_state.processed_images[0])
                fig2, ax = plt.subplots(figsize=(9, 3))
                fig2.patch.set_alpha(0)
                ax.set_facecolor("#0f172a")
                ax.hist(img_data.flatten(), bins=80, color="#3b82f6", alpha=0.8, edgecolor="none")
                ax.set_title("Voxel Intensity Distribution", color="#94a3b8", fontsize=12, pad=10)
                ax.tick_params(colors="#64748b")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1e293b")
                st.pyplot(fig2)

        st.divider()
        st.markdown("##### ✍️ Consultant Sign-off")
        expert_name = st.text_input("Reviewing consultant's name", placeholder="e.g. Dr. XYZ")
        st.session_state.expert_sign_off = st.checkbox("I confirm the diagnostic accuracy of this report.")
        if st.session_state.expert_sign_off and expert_name:
            pdf_bytes = generate_pdf(
                st.session_state.current_report, scan_type, expert_name, st.session_state.processed_images
            )
            st.download_button(
                label="Download Report (PDF) ↓",
                data=pdf_bytes,
                file_name=f"Medora_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption("Complete sign-off above to enable PDF export.")

st.divider()
st.caption("AI-assisted diagnostics via local Qwen2-VL inference. All outputs require clinical validation.")
