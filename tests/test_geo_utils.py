"""Test utilitas geospasial: jarak, clustering, dan tabel prioritas."""
from geo_utils import haversine, assign_cluster, compute_priority_table


def test_haversine_zero():
    assert haversine(0, 0, 0, 0) == 0


def test_haversine_one_degree_latitude():
    # 1 derajat lintang ~ 111 km
    d = haversine(0, 0, 1, 0)
    assert 110_000 < d < 112_000


def test_assign_cluster_near_and_far():
    rows = []
    c1 = assign_cluster(-6.2000, 106.8000, rows)
    assert c1 == 1
    rows.append({"lat": -6.2000, "lon": 106.8000, "cluster_id": c1})

    # ~11 meter → cluster sama
    c2 = assign_cluster(-6.2000, 106.8001, rows)
    assert c2 == 1

    # ~20 km → cluster baru
    c3 = assign_cluster(-6.4000, 107.0000, rows)
    assert c3 == 2


def test_compute_priority_table_ranking():
    reports = [
        {"filename": "a.jpg", "prediction": "Pothole", "confidence": 0.90,
         "lat": -6.2, "lon": 106.8, "cluster_id": 1},
        {"filename": "b.jpg", "prediction": "Pothole", "confidence": 0.80,
         "lat": -6.2, "lon": 106.8, "cluster_id": 1},
        {"filename": "c.jpg", "prediction": "Normal", "confidence": 0.95,
         "lat": -6.5, "lon": 107.0, "cluster_id": 2},
    ]
    df = compute_priority_table(reports)
    assert not df.empty
    top = df.iloc[0]
    assert top["cluster_id"] == 1
    assert top["count_pothole"] == 2
    assert top["priority_level"] in ("TINGGI", "SEDANG", "RENDAH")
