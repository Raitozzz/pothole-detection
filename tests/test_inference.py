"""Test pipeline prediksi model pada satu gambar."""
from PIL import Image
from model_utils import predict, CLASSES


def test_predict_returns_valid_schema():
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    result = predict(img)

    assert result["prediction"] in CLASSES
    for key in ("confidence", "prob_normal", "prob_pothole", "status"):
        assert key in result

    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(result["prob_normal"] + result["prob_pothole"] - 1.0) < 1e-4
    assert result["status"] in ("Terverifikasi", "Perlu Tinjauan")


def test_predict_handles_non_rgb():
    # Gambar grayscale harus otomatis dikonversi tanpa error.
    img = Image.new("L", (224, 224), color=100)
    result = predict(img)
    assert result["prediction"] in CLASSES
