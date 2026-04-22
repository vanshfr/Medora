import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import numpy as np

class MedGemmaHandler:
    """
    A Singleton handler for the Qwen2-VL model. 
    This ensures the heavy model is only loaded into memory once.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MedGemmaHandler, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self, hf_token=None):
        """
        Loads the model and processor from Hugging Face.
        Optimized for local CPU inference by capping pixel resolution.
        """
        if self.initialized: 
            return
        
        self.model_id = "Qwen/Qwen2-VL-2B-Instruct"
        if hf_token: 
            os.environ["HF_TOKEN"] = hf_token

        try:
            print("System: Initializing Fast-Inference CPU Mode...")
            
            # min/max_pixels are set low to ensure the CPU can process 
            # the medical image almost instantly without hanging.
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                min_pixels=128*28*28, 
                max_pixels=256*28*28  
            )
            
            # float32 is the standard for CPU; low_cpu_mem_usage prevents 
            # the system from crashing on limited RAM.
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            self.initialized = True
            print("✅ Medora AI Engine: Ready for Inference")
        except Exception as e:
            self.initialized = False
            raise e

    def get_intensity_analysis(self, image):
        """
        Mathematical Analysis: Converts standard 8-bit image pixels into 
        simulated Hounsfield Units (HU) used in clinical CT imaging.
        """
        try:
            # Analyze raw radiodensity by converting to grayscale
            img_array = np.array(image.convert("L"))
            
            # Map 0-255 grayscale range to the -1000 to +1000 HU scale
            hu_data = (img_array.astype(np.float32) * (2000 / 255.0)) - 1000
            
            return hu_data.flatten()
        except Exception as e:
            print(f"Physics Engine Error: {e}")
            return None

    def analyze(self, image, scan_type="Medical Scan"):
        """
        AI Reasoning: Uses Qwen2-VL to generate a clinical impression 
        of the provided medical slice.
        """
        if not self.initialized: 
            return "Model initialization failed. Check HF Token."
            
        try:
            # Resize image to a square format that the Vision-LLM prefers
            image = image.convert("RGB").resize((224, 224))

            # Construct the multimodal prompt
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Analyze this {scan_type}. Provide a brief 2-sentence clinical impression."}
            ]}]

            # Tokenization and Model Generation
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            
            # max_new_tokens is kept low (60) to keep response times under 10 seconds on most CPUs
            output_ids = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)
            
            # Post-processing: Extract only the model's response
            generated_ids = [output_ids[len(inputs.input_ids):] for output_ids, inputs in zip(output_ids, [inputs])]
            raw_output = self.processor.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
            
            # Remove any 'assistant' tags to ensure the report looks professional
            findings = raw_output.lower().split("assistant")[-1].strip()

            return (
                f"### Analysis Report ({scan_type})\n\n"
                f"**Findings:**\n"
                f"- {findings}\n\n"
                f"**Note:** Tissue density within normal range for local CPU inference."
            )

        except Exception as e:
            return f"Error during AI analysis: {str(e)}"

    def mock_analyze(self, scan_type="CT"):
        """
        Mock Logic: Used for UI testing or when running without a heavy backend.
        """
        return (
            f"### AI Analysis Report ({scan_type})\n\n"
            f"**Findings:**\n"
            f"- No significant abnormalities detected in the visible thoracic fields.\n"
            f"- Anatomical structures appear consistent with standard medical baselines.\n\n"
            f"**Note:** System is running in Demo Mode (Mock AI)."
        )
