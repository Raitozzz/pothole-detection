# EDA Report — Pothole Detection Dataset

## Tujuan
Memvalidasi kualitas dan memahami karakteristik dataset sebelum masuk ke tahap modeling.

---

## Alur EDA

```
Load dataset
    ↓
1. Pemilihan Dataset & Explore
    ↓
2. Data Understanding & Penentuan Goals
    ↓
3. Class Distribution (bar + pie chart)
    ↓
4. Sample Grid Visualisasi
    ↓
5. Dimension Audit (width, height, scatter)
    ↓
6. Color Channel Verification
    ↓
7. Aspect Ratio & Top 10 Resolusi (histogram + horizontal bar chart)
    ↓
8. Brightness Analysis (histogram + boxplot)
    ↓
9. Average Image per Kelas (visual fingerprint)
    ↓
10. Insight Summary
```

---

## Hasil

### Dataset
| Atribut | Nilai |
|---|---|
| Total gambar | 681 |
| Kelas normal | 352 (51.7%) |
| Kelas potholes | 329 (48.3%) |
| Imbalance ratio | 1.07:1 |

### Kualitas Data
| Check | Hasil |
|---|---|
| File corrupt | 0 |
| Color mode | Semua RGB |
| Dimensi | Bervariasi (tidak seragam) |
| Ukuran terlalu kecil | 0 |

### Distribusi Dimensi & Resolusi
- Width dan height gambar bervariasi signifikan antar file
- Tidak ada resolusi yang dominan → resize ke 224×224 
- Top 10 resolusi divisualisasikan dengan horizontal bar chart untuk melihat distribusi ukuran gambar yang paling umum

### Brightness
- Gambar `potholes` cenderung memiliki brightness lebih rendah dibanding `normal` karena lubang menciptakan bayangan gelap
- Implikasi: **ColorJitter** pada augmentasi penting agar model tidak hanya bergantung pada kecerahan

### Average Image (Visual Fingerprint)
- Rata-rata piksel kelas `normal` menunjukkan tekstur jalan yang lebih seragam dan terang
- Rata-rata piksel kelas `potholes` menunjukkan area gelap di tengah frame

---

## Insight Utama

1. **Dataset bersih** — tidak ada file corrupt atau gambar non-RGB yang perlu dihapus
2. **Class imbalance sangat ringan** (1.07:1) — tidak perlu oversampling, cukup `class_weight` pada loss function
3. **Dimensi tidak seragam** — resize ke 224×224 wajib sebelum masuk model
4. **Perbedaan brightness antar kelas** terdeteksi — ColorJitter pada augmentasi akan membantu model tidak overfit pada kondisi cahaya tertentu

---

## Output File
| File | Keterangan |
|---|---|
| `eda_class_distribution.png` | Bar + pie chart distribusi kelas |
| `eda_sample_grid.png` | Grid 5×2 sample gambar per kelas |
| `eda_dimensions.png` | Histogram width, height, scatter plot |
| `eda_aspect_ratio.png` | Distribusi aspect ratio + horizontal bar chart top 10 resolusi |
| `eda_brightness.png` | Histogram + boxplot brightness per kelas |
| `eda_average_image.png` | Visual fingerprint rata-rata per kelas |
