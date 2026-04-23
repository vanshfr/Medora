import os
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


class MedoraHandler:
    """
    Singleton handler for Qwen2-VL medical image analysis.
    Keeps the model in memory after first initialisation so
    subsequent calls don't reload weights.
    """

    _instance = None

    # ── Singleton ──────────────────────────────────────────────────────────
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    # ── Initialisation ─────────────────────────────────────────────────────
    def initialize(self, hf_token: str | None = None) -> None:
        """Load processor and model weights (runs once per session)."""
        if self.initialized:
            return

        self.model_id = "Qwen/Qwen2-VL-2B-Instruct"

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        print("Initialising Qwen2-VL (CPU fast-inference mode)…")

        # Pixel caps kept low for reasonable CPU latency
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        self.initialized = True
        print("✅ Model ready.")

    # ── Intensity / HU analysis ────────────────────────────────────────────
    def get_intensity_analysis(self, image: Image.Image) -> np.ndarray | None:
        """
        Convert a PIL image to a flat array of pseudo-Hounsfield Units.
        Grayscale 0–255 is linearly mapped to CT HU range –1000 → +1000.
        """
        try:
            img_array = np.array(image.convert("L"), dtype=np.float32)
            hu_data   = img_array * (2000.0 / 255.0) - 1000.0
            return hu_data.flatten()
        except Exception as e:
            print(f"[Densitometry] Error: {e}")
            return None

    # ── Live inference ─────────────────────────────────────────────────────
    def analyze(self, image: Image.Image, scan_type: str = "Medical Scan") -> str:
        """Run Qwen2-VL on a single image and return a formatted clinical report."""
        if not self.initialized:
            return "⚠️ Model not initialised. Please provide a Hugging Face token."

        try:
            image = image.convert("RGB").resize((224, 224))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": (
                                f"Analyze this {scan_type}. "
                                "Provide a brief clinical impression and "
                                "provide a definitive concluding sentence within 100 words. Ensure the response is complete."
                            ),
                        },
                    ],
                }
            ]

            text   = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )

            output_ids = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False
            )

            # Strip the prompt tokens to keep only the generated portion
            generated = [
                out[len(inp):]
                for out, inp in zip(output_ids, inputs.input_ids)
            ]
            raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0]

            # Remove any leading "assistant" marker injected by the template
            findings = raw.lower().split("assistant")[-1].strip()

            return self._format_report(findings, scan_type)

        except Exception as e:
            return f"⚠️ Analysis error: {e}"

    # ── Demo / mock response ───────────────────────────────────────────────
    def mock_analyze(self, scan_type: str = "CT") -> str:
        """Return a pre-written report for demo purposes."""
        findings = (
            "No significant abnormalities detected in the imaged region. "
            "Tissue density is within the normal reference range for this modality."
        )
        return self._format_report(findings, scan_type, demo=True)

    # ── Internal helpers ───────────────────────────────────────────────────
    @staticmethod
    def _format_report(findings: str, scan_type: str, demo: bool = False) -> str:
        """Wrap raw findings text in a consistent Markdown report template."""
        mode_note = "Running in **Demo Mode** — mock response." if demo else \
                    "Tissue density assessed via local CPU inference."
        return (
            f"### Analysis Report — {scan_type}\n\n"
            f"**Findings:**\n"
            f"- {findings}\n\n"
            f"**Note:** {mode_note}"
        )
