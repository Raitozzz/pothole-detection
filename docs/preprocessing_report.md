# Preprocessing Report — Pothole Detection Dataset

## Tujuan
Membersihkan data, membagi dataset, dan mendefinisikan transformasi gambar yang siap digunakan untuk training model.

---

## Alur Preprocessing

```
Load semua file JPG dari data/raw/images/
    ↓
1. Data Cleaning
   - Cek file corrupt (PIL verify)
   - Cek dimensi terlalu kecil (< 32px)
   - Catat file bermasalah → pisahkan dari data valid
    ↓
2. Cek Non-RGB
   - Filter gambar bukan RGB
   - Tidak dihapus, di-convert otomatis saat load
    ↓
3. Stratified Split 70 / 15 / 15
   - Train : 70%
   - Val   : 15%
   - Test  : 15%
   - Proporsi kelas dijaga sama di setiap split
    ↓
4. Resize 224×224 + convert RGB → simpan ke data/processed/{split}/{class}/
    ↓
5. Simpan manifest.csv ke data/split/
    ↓
6. Definisi Feature Transformation
   - Train  : augmentasi real-time + normalize (tanpa Resize, gambar sudah 224×224)
   - Val/Test: normalize only (gambar sudah 224×224 dari disk)
```

---

## Hasil

### Data Cleaning
| Check | Hasil |
|---|---|
| Total file di-scan | 681 |
| File valid | 681 |
| File corrupt / bermasalah | 0 |
| Gambar non-RGB | 0 |

### Split Result
| Split | Total | Normal | Potholes |
|---|---|---|---|
| Train | 477 | 246 | 231 |
| Val | 102 | 53 | 49 |
| Test | 102 | 53 | 49 |

> Angka di atas dapat sedikit berbeda tergantung pembulatan stratified split.

### Struktur Output
```
data/
├── processed/
│   ├── train/
│   │   ├── normal/      (246 gambar, 224x224 RGB)
│   │   └── potholes/    (231 gambar, 224x224 RGB)
│   ├── val/
│   │   ├── normal/      (53 gambar, 224x224 RGB)
│   │   └── potholes/    (49 gambar, 224x224 RGB)
│   └── test/
│       ├── normal/      (53 gambar, 224x224 RGB)
│       └── potholes/    (49 gambar, 224x224 RGB)
└── split/
    └── manifest.csv
```

---

## Feature Transformation

### Disimpan ke Disk (semua split)
| Langkah | Detail |
|---|---|
| Convert RGB | Image.open().convert("RGB") |
| Resize | 224 × 224 px (LANCZOS) |
| Simpan | JPEG quality=95 ke data/processed/ |

### Train (real-time saat training)
| Langkah | Detail |
|---|---|
| RandomHorizontalFlip | p = 0.5 |
| RandomVerticalFlip | p = 0.2 (rendah — orientasi jalan bermakna) |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.2 |
| RandomRotation | ±15° |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

### Val / Test (real-time saat evaluasi)
| Langkah | Detail |
|---|---|
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

> Normalisasi menggunakan statistik ImageNet karena model yang dipakai (EfficientNet-B0) di-pretrain pada ImageNet.

> **Augmentasi hanya aktif saat training** — tidak disimpan ke disk, dijalankan real-time di memori setiap epoch.

---

## Keputusan & Alasan

| Keputusan | Alasan |
|---|---|
| Tidak oversampling | Imbalance ratio hanya 1.07:1, di bawah threshold 1.5:1 |
| Class weight pada loss | Cara ringan menangani imbalance tanpa menambah data |
| VerticalFlip p=0.2 | Foto jalan punya orientasi atas-bawah yang bermakna |
| ColorJitter | Foto diambil dalam berbagai kondisi cahaya |
| ImageNet normalisasi | Model pretrained EfficientNet-B0 menggunakan statistik ini |

---

## Output File
| File | Keterangan |
|---|---|
| `data/processed/` | Gambar 224×224 RGB tersimpan per split dan kelas |
| `data/split/manifest.csv` | Daftar semua file: split, label, filename |
| `preproc_split_distribution.png` | Bar chart distribusi kelas per split |
| `preproc_augmentation.png` | Visualisasi efek augmentasi train vs val/test |
