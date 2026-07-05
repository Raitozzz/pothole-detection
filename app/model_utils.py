"""Pemuatan model dan logika prediksi untuk deteksi pothole."""
from functools import lru_cache
from pathlib import Path

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).parent.parent / "models" / "exp4_mobilenetv3.pth"
CONFIDENCE_THRESHOLD = 0.70

# Urutan kelas mengikuti class_to_idx saat training: indeks 0 = normal, 1 = pothole.
CLASSES = ["Normal", "Pothole"]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@lru_cache(maxsize=1)
def load_model():
    """Muat MobileNetV3 hasil fine-tuning untuk klasifikasi normal vs pothole."""
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict(pil_image: Image.Image) -> dict:
    """Prediksi satu gambar: kelas, tingkat keyakinan, status, dan probabilitas."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    model = load_model()
    tensor = _transform(pil_image).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(tensor), dim=1)[0]
    prob_normal  = probs[0].item()
    prob_pothole = probs[1].item()
    pred_idx     = probs.argmax().item()
    confidence   = probs[pred_idx].item()
    prediction   = CLASSES[pred_idx]
    status = "Terverifikasi" if confidence >= CONFIDENCE_THRESHOLD else "Perlu Tinjauan"

    return {
        "prediction":   prediction,
        "confidence":   confidence,
        "status":       status,
        "prob_normal":  prob_normal,
        "prob_pothole": prob_pothole,
    }
