# 🩺 Medora - Clinical Image Analysis Dashboard

Medora is a local AI-powered clinical imaging dashboard built with Streamlit. It enables radiologists and clinicians to upload medical scans, generate AI-assisted diagnostic reports, and export signed PDF reports - all running on local CPU inference with no cloud dependency.

<img width="1919" height="878" alt="image" src="https://github.com/user-attachments/assets/afc39eac-98ce-46e1-b4c5-a10539e977a4" />

---

## ✨ Features

- **Multi-format scan ingestion** - supports DICOM (`.dcm`), NIfTI (`.nii`, `.nii.gz`), TIFF, and standard image formats (JPG, PNG, BMP, WebP)
- **AI-generated clinical insights** - powered by the Qwen2-VL vision-language model via Hugging Face Transformers, optimized for local CPU inference
- **Scan modality selection** - CT Scan, MRI Scan, and Mammography
- **Radiodensity & densitometry analysis** - interactive Hounsfield Unit (HU) histogram with tissue reference lines (air, lung, water, bone)
- **Voxel intensity distribution** - density chart for visual tissue composition review
- **Demo mode** - pre-written mock responses for quick exploration without a model download
- **Consultant sign-off & PDF export** - generate a professionally formatted diagnostic report PDF with imaging snapshot, clinical findings, and digital consultant signature

---

## 🗂 Project Structure

```
Medora/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── assets/
│   └── styles.css          # Custom UI styling
├── core/
│   └── model_handler.py    # Qwen2-VL model loading and inference (MedGemmaHandler)
├── utils/
│   └── image_processor.py  # DICOM/NIfTI/TIFF processing utilities (MedicalImageProcessor)
└── testdata/               # Sample scans for testing
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A [Hugging Face account](https://huggingface.co) and access token (required to download the model)

### Installation

```bash
git clone https://github.com/vanshfr/Medora.git
cd Medora
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

Alternatively, you can paste the token directly in the sidebar at runtime.

### Running the App

```bash
streamlit run app.py
```

---

## 🧪 Demo Mode

Don't want to download the full model? Toggle **"Use Demo Mode (Mock AI)"** in the sidebar to use pre-written responses and explore the full UI and PDF export workflow instantly.

---

## 📦 Dependencies

| Category | Packages |
|---|---|
| Web framework | `streamlit`, `python-dotenv` |
| AI / ML | `torch`, `transformers`, `accelerate`, `qwen-vl-utils` |
| Medical imaging | `pydicom`, `nibabel`, `tifffile`, `opencv-python-headless` |
| Image processing | `Pillow`, `numpy` |
| Visualization | `plotly`, `matplotlib` |
| PDF generation | `fpdf2` |

---

## ⚠️ Disclaimer

Medora is a research and workflow-assistance tool. All AI-generated outputs require review and sign-off by a qualified clinician before any clinical use. This software is **not a certified medical device** and should not be used as the sole basis for diagnosis or treatment decisions.

---

## 📄 License

This project currently has no license specified. Please contact the repository owner before using or distributing.
