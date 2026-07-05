# Dataset & Project Goals — Pothole Detection System

## Business Question

> **"Dapatkah sistem  mengklasifikasikan kondisi jalan dari sebuah foto sebagai normal atau berlubang (pothole) dengan akurasi ≥ 05% pada test set?"**

---

## Problem Statement

Kerusakan jalan berupa lubang (pothole) merupakan masalah infrastruktur yang berdampak langsung pada keselamatan berkendara dan biaya operasional kendaraan. Inspeksi jalan secara manual membutuhkan waktu dan tenaga yang besar. Sistem klasifikasi berbasis AI dapat mengotomasi proses deteksi ini dari foto survei jalan.

---

## Dataset

| Atribut | Detail |
|---|---|
| Sumber | Self-collected (foto jalan raya) |
| Format | JPG |
| Total gambar | 681 |
| Kelas | `normal` (352), `potholes` (329) |
| Imbalance ratio | 1.07:1 (sangat ringan) |
| Dimensi | Bervariasi — di-resize ke 224×224 saat preprocessing |
| Color mode | RGB |

### Struktur Folder
```
data/
├── raw/images/
│   ├── normal/       # 352 gambar
│   └── potholes/     # 329 gambar
└── split/
    ├── train/        # 70%
    ├── val/          # 15%
    └── test/         # 15%
```

---

## Goals & Success Metrics

| Metric                | Target                |
| -----------------------| -----------------------|
| Accuracy (test set)   | ≥ 80%                 |
| F1-Score macro        | ≥ 0.80                |
| Recall kelas potholes | ≥ 80%                 |
| Inference time        | < 10 detik per gambar |

**Prioritas:** Recall kelas potholes diutamakan karena *false negative* (jalan berlubang diklasifikasi sebagai normal) jauh lebih berbahaya dibandingkan *false positive*.

---

## Model Path

**Path B — Transfer Learning**

Dipilih karena:
- Dataset 681 gambar terlalu kecil untuk training CNN dari awal
- Pretrained model (EfficientNet-B0, dilatih pada ImageNet) sudah memahami fitur visual dasar (tepi, tekstur, warna)
- Fine-tuning jauh lebih efisien secara komputasi

**Baseline:** Majority class classifier (~51.7% accuracy) — model wajib melampaui ini.

---

## Preprocessing Plan

| Tahap | Detail |
|---|---|
| Split | Stratified 70/15/15 (train/val/test) |
| Resize | 224 × 224 px |
| Augmentasi train | RandomCrop, HorizontalFlip, VerticalFlip, ColorJitter, Rotation(±15°) |
| Val / Test | Resize only |
| Normalisasi | ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Class weight | Diterapkan pada loss function untuk menangani imbalance ringan |

---

## Deployment Target

| Komponen | Teknologi |
|---|---|
| API | FastAPI + Uvicorn |
| UI | Streamlit (upload single & bulk foto) |
| Database | SQLite (logging prediksi) |
| Container | Docker + docker-compose |
| CI/CD | GitHub Actions |

---

## Stakeholder Impact

Sistem ini dapat digunakan oleh:
- **Dinas Pekerjaan Umum** — deteksi otomatis dari foto survei drone/kamera
- **Perusahaan logistik** — penilaian kondisi rute pengiriman
- **Masyarakat umum** — pelaporan kerusakan jalan via aplikasi
