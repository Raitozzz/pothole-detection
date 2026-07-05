"""REST API deteksi pothole berbasis FastAPI.

Menjalankan: `uvicorn api.server:app --reload`
Dokumentasi interaktif tersedia di `/docs` setelah server berjalan.
"""
import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
from model_utils import predict  # noqa: E402

app = FastAPI(
    title="Pothole Detection API",
    description="Deteksi kondisi jalan (normal atau berlubang) dari sebuah foto.",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Cek apakah service hidup."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Terima satu foto jalan, kembalikan prediksi kelas dan tingkat keyakinan."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Gambar tidak dapat dibaca.")

    result = predict(image)
    return {"filename": file.filename, **result}
