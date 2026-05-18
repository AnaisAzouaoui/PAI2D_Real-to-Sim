import numpy as np
import torch
from PIL import Image

_processor = None
_model = None

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def get_model():
    global _processor, _model
    if _model is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print(f"[depth_estimator] chargement {MODEL_ID}...")
        _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device)
        _model.eval()
        print(f"[depth_estimator] modele pret sur {device}")
    return _processor, _model


def estimate_depth(img_pil):
    """Retourne une depth map metrique float32 (metres), meme taille que img_pil."""
    processor, model = get_model()
    device = next(model.parameters()).device
    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth.squeeze().cpu().numpy().astype(np.float32)
    if depth.shape != (img_pil.height, img_pil.width):
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize((img_pil.width, img_pil.height), Image.BILINEAR)
        depth = np.array(depth_img, dtype=np.float32)
    return depth
