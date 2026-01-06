<<<<<<< HEAD
﻿import open_clip
import torch
from PIL import Image

class MedicalEmbedder:
    def __init__(self):
        # Load BiomedCLIP model (Microsoft)
        # This model is excellent for retrieval as it understands both medical images and text
=======
﻿# File: backend/embedding.py

import open_clip
import torch
from PIL import Image
import numpy as np

class MedicalEmbedder:
    def __init__(self):
        # Load BiomedCLIP model
>>>>>>> b37cd91 (Upload medical retrieval project)
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        print(f"Loading model: {model_name}...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
<<<<<<< HEAD
        # Load model and preprocessing transforms
=======
>>>>>>> b37cd91 (Upload medical retrieval project)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print("Model loaded successfully!")

<<<<<<< HEAD
    def encode_image(self, image_path):
        """Converts an image into a vector embedding."""
        image = Image.open(image_path).convert("RGB")
=======
    def encode_image(self, image_input):
        """
        Converts an image into a vector embedding.
        Input: 
            - image_input: Có thể là đường dẫn file (str) HOẶC đối tượng PIL Image
        """
        # --- NÂNG CẤP: Hỗ trợ cả Path và PIL Image ---
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB") # Xử lý ảnh crop từ RAM
            
>>>>>>> b37cd91 (Upload medical retrieval project)
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
<<<<<<< HEAD
            # Normalize vector for accurate cosine similarity calculation
=======
>>>>>>> b37cd91 (Upload medical retrieval project)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()[0].tolist()

    def encode_text(self, text):
<<<<<<< HEAD
        """Converts text (query) into a vector embedding - useful for text-based search later."""
=======
        """Converts text query into vector."""
>>>>>>> b37cd91 (Upload medical retrieval project)
        texts = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()[0].tolist()