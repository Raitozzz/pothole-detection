"""Penyimpanan laporan deteksi ke SQLite via SQLAlchemy.

Lokasi database bisa diatur lewat env var POTHOLE_DB (dipakai saat testing).
Default: data/reports.db (folder data/ diabaikan git).
"""
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

_default = Path(__file__).parent.parent / "data" / "reports.db"
DB_PATH = Path(os.environ.get("POTHOLE_DB") or _default)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False}
)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class Report(Base):
    __tablename__ = "reports"

    id           = Column(Integer, primary_key=True)
    filename     = Column(String)
    prediction   = Column(String)
    confidence   = Column(Float)
    status       = Column(String)
    prob_normal  = Column(Float)
    prob_pothole = Column(Float)
    lat          = Column(Float, nullable=True)
    lon          = Column(Float, nullable=True)
    cluster_id   = Column(Integer, nullable=True)
    hash         = Column(String, nullable=True)
    timestamp    = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Buat tabel bila belum ada."""
    Base.metadata.create_all(engine)


def save_reports(rows: list) -> int:
    """Simpan sekumpulan laporan (list of dict). Kembalikan jumlah tersimpan."""
    init_db()
    session = SessionLocal()
    try:
        for r in rows:
            session.add(Report(
                filename=r.get("filename"),
                prediction=r.get("prediction"),
                confidence=r.get("confidence"),
                status=r.get("status"),
                prob_normal=r.get("prob_normal"),
                prob_pothole=r.get("prob_pothole"),
                lat=r.get("lat"),
                lon=r.get("lon"),
                cluster_id=r.get("cluster_id"),
                hash=r.get("hash"),
            ))
        session.commit()
        return len(rows)
    finally:
        session.close()


def get_all_reports() -> list:
    """Ambil semua laporan tersimpan sebagai list of dict."""
    init_db()
    session = SessionLocal()
    try:
        rows = session.query(Report).order_by(Report.id).all()
        return [{
            "filename":     r.filename,
            "prediction":   r.prediction,
            "confidence":   r.confidence,
            "status":       r.status,
            "prob_normal":  r.prob_normal,
            "prob_pothole": r.prob_pothole,
            "lat":          r.lat,
            "lon":          r.lon,
            "cluster_id":   r.cluster_id,
            "hash":         r.hash,
            "timestamp":    r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else None,
        } for r in rows]
    finally:
        session.close()


def clear_reports():
    """Hapus semua laporan tersimpan."""
    init_db()
    session = SessionLocal()
    try:
        session.query(Report).delete()
        session.commit()
    finally:
        session.close()
