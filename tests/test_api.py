"""Test end-to-end REST API: request masuk → model → response JSON."""
import io

from fastapi.testclient import TestClient
from PIL import Image

from api.server import app

client = TestClient(app)


def _jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(100, 100, 100)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_endpoint_returns_prediction():
    r = client.post("/predict", files={"file": ("test.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in ("Normal", "Pothole")
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["filename"] == "test.jpg"


def test_predict_rejects_non_image():
    r = client.post(
        "/predict",
        files={"file": ("note.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert r.status_code == 400
