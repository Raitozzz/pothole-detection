"""Ekstraksi koordinat GPS, perhitungan jarak, dan pengelompokan lokasi."""
import math
from PIL import Image
import pandas as pd


def extract_exif_coords(pil_image: Image.Image):
    """Ambil koordinat (lat, lon) dari metadata EXIF foto, atau None bila tidak ada."""
    try:
        import piexif
        exif_bytes = pil_image.info.get("exif", b"")
        if not exif_bytes:
            return None
        exif = piexif.load(exif_bytes)
        gps = exif.get("GPS", {})
        if not gps:
            return None

        def _to_deg(vals, ref):
            """Konversi koordinat format derajat-menit-detik (DMS) ke desimal."""
            d = vals[0][0] / vals[0][1]
            m = vals[1][0] / vals[1][1]
            s = vals[2][0] / vals[2][1]
            val = d + m / 60 + s / 3600
            if ref in (b"S", b"W"):
                val = -val
            return val

        lat = _to_deg(gps[piexif.GPSIFD.GPSLatitude],
                      gps[piexif.GPSIFD.GPSLatitudeRef])
        lon = _to_deg(gps[piexif.GPSIFD.GPSLongitude],
                      gps[piexif.GPSIFD.GPSLongitudeRef])
        return (lat, lon)
    except Exception:
        return None


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Jarak antara dua titik koordinat GPS dalam meter (rumus Haversine)."""
    R = 6_371_000  # Radius rata-rata bumi dalam meter
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def assign_cluster(lat: float, lon: float, existing_rows: list, radius: float = 25) -> int:
    """Tentukan cluster untuk sebuah titik koordinat.

    Bila ada laporan lain dalam jangkauan `radius` meter, titik ini masuk ke
    cluster terdekat yang sudah ada. Bila tidak, dibuat cluster baru (id max + 1).
    """
    best_id = None
    best_dist = float("inf")
    for row in existing_rows:
        if row.get("lat") is None or row.get("lon") is None:
            continue
        d = haversine(lat, lon, row["lat"], row["lon"])
        if d < radius and d < best_dist:
            best_dist = d
            best_id = row["cluster_id"]
    if best_id is not None:
        return best_id
    existing_ids = [r["cluster_id"] for r in existing_rows if r.get("cluster_id") is not None]
    return (max(existing_ids) + 1) if existing_ids else 1


def compute_priority_table(reports: list) -> pd.DataFrame:
    """Rangkum laporan per cluster menjadi tabel prioritas perbaikan.

    Hanya laporan berkoordinat valid yang diproses. Cluster diurutkan dari jumlah
    pothole terbanyak, lalu diberi level prioritas TINGGI, SEDANG, atau RENDAH.
    """
    rows_with_coords = [r for r in reports if r.get("lat") is not None and r.get("cluster_id")]
    if not rows_with_coords:
        return pd.DataFrame()

    df = pd.DataFrame(rows_with_coords)
    grouped = df.groupby("cluster_id").agg(
        center_lat=("lat", "mean"),
        center_lon=("lon", "mean"),
        total_laporan=("filename", "count"),
        count_pothole=("prediction", lambda x: (x == "Pothole").sum()),
        count_normal=("prediction", lambda x: (x == "Normal").sum()),
        avg_confidence=("confidence", "mean"),
    ).reset_index()

    grouped = grouped.sort_values(
        ["count_pothole", "avg_confidence"], ascending=[False, False]
    ).reset_index(drop=True)

    grouped["rank"] = grouped.index + 1
    grouped["priority_level"] = grouped["count_pothole"].apply(
        lambda c: "TINGGI" if c >= 5 else ("SEDANG" if c >= 2 else "RENDAH")
    )
    grouped["avg_confidence"] = (grouped["avg_confidence"] * 100).round(1)

    cols = ["rank", "cluster_id", "center_lat", "center_lon",
            "total_laporan", "count_pothole", "count_normal",
            "avg_confidence", "priority_level"]
    return grouped[cols]
