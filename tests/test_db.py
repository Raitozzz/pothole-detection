"""Test penyimpanan laporan ke database (roundtrip save/load/clear)."""
import db


def test_save_load_clear_roundtrip():
    db.clear_reports()
    assert db.get_all_reports() == []

    rows = [{
        "filename": "x.jpg", "prediction": "Pothole", "confidence": 0.88,
        "status": "Terverifikasi", "prob_normal": 0.12, "prob_pothole": 0.88,
        "lat": -6.2, "lon": 106.8, "cluster_id": 1, "hash": "abc123",
    }]
    n = db.save_reports(rows)
    assert n == 1

    out = db.get_all_reports()
    assert len(out) == 1
    assert out[0]["filename"] == "x.jpg"
    assert out[0]["prediction"] == "Pothole"
    assert out[0]["cluster_id"] == 1

    db.clear_reports()
    assert db.get_all_reports() == []
