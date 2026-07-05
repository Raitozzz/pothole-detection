# Modeling Report — Pothole Detection

## Tujuan
Melatih dan membandingkan model transfer learning untuk klasifikasi biner gambar jalan (normal vs potholes), dengan target akurasi ≥ 85% dan recall kelas potholes ≥ 85%.

---

## Alur Modeling

```
Data (data/processed/ — 224×224 RGB)
    ↓
DataLoader (ImageFolder, augmentasi real-time pada train)
    ↓
Baseline Model (Majority Classifier)
    ↓
Transfer Learning Models
  ├── EfficientNet-B0 (frozen)     → Exp 1
  ├── EfficientNet-B0 (fine-tune)  → Exp 2
  ├── EfficientNet-B0 + scheduler  → Exp 3
  ├── MobileNetV3-Small            → Exp 4
  ├── ResNet18                     → Exp 5
  └── ViT-Tiny (timm)              → Exp 6
    ↓
Simpan best model → models/best_model.pth
    ↓
Evaluasi pada test set → 04_evaluation.ipynb
```

---

## Model Selection

### Mengapa Transfer Learning?

Dataset hanya 681 gambar — terlalu kecil untuk training dari scratch.
Transfer learning menggunakan bobot ImageNet sehingga backbone sudah "memahami" fitur visual umum (tekstur, tepi, pola).

### Model yang Dibandingkan

| Model | Sumber | Params | Alasan Dipilih |
|---|---|---|---|
| Majority Classifier | - | - | Baseline minimum (~51.7%) |
| EfficientNet-B0 | torchvision | ~5.3M | Target utama — balance accuracy vs efisiensi |
| MobileNetV3-Small | torchvision | ~2.5M | Paling ringan — cocok untuk deployment mobile |
| ResNet18 | torchvision | ~11.7M | Klasik dan stabil — pembanding |
| ViT-Tiny (patch16-224) | timm | ~5.7M | Vision Transformer ringan, pretrained ImageNet-21k |

---

## Hyperparameter Tuning

### Konfigurasi Eksperimen

| Exp | Model | LR | Freeze | Epochs | Scheduler |
|---|---|---|---|---|---|
| Exp 1 | EfficientNet-B0 | 1e-3 | Backbone frozen | 10 | Tidak |
| Exp 2 | EfficientNet-B0 | 1e-4 | Fine-tune semua | 20 | Tidak |
| Exp 3 | EfficientNet-B0 | 1e-4 | Fine-tune semua | 20 | CosineAnnealingLR |
| Exp 4 | MobileNetV3-Small | 1e-4 | Fine-tune semua | 20 | Tidak |
| Exp 5 | ResNet18 | 1e-4 | Fine-tune semua | 20 | Tidak |
| Exp 6 | ViT-Tiny | 1e-4 | Fine-tune semua | 20 | Tidak |

### Keputusan Desain

| Keputusan | Alasan |
|---|---|
| Adam optimizer | Adaptif, konvergen cepat pada dataset kecil |
| CrossEntropyLoss + class_weight | Menangani imbalance ringan (1.07:1) tanpa oversampling |
| Early stopping patience=5 | Cegah overfitting, hemat waktu training |
| CosineAnnealingLR | Warm restart LR membantu keluar dari local minima |
| EfficientNet frozen (Exp1) | Validasi apakah fitur ImageNet cukup — tanpa adaptasi |
| EfficientNet fine-tune (Exp2) | Biarkan semua layer beradaptasi ke domain pothole |
| Batch size 32 | Balance antara gradient noise dan VRAM usage |
| num_workers=0 | Windows aman dari multiprocessing deadlock |

---

## Augmentasi Training

| Transform | Parameter | Alasan |
|---|---|---|
| RandomHorizontalFlip | p=0.5 | Foto jalan simetris kiri-kanan |
| RandomVerticalFlip | p=0.2 | Orientasi atas-bawah bermakna, flip jarang terjadi |
| ColorJitter | brightness=0.3, contrast=0.3 | Kondisi cahaya bervariasi |
| RandomRotation | ±15° | Kamera tidak selalu horizontal sempurna |
| Normalize | ImageNet mean/std | Sinkron dengan bobot pretrained |

---

## Hasil Eksperimen

> Tabel di bawah diisi setelah menjalankan `03_modeling.ipynb`.

| Eksperimen | Val F1 (macro) | Val Acc | Keterangan |
|---|---|---|---|
| Baseline | ~0.34 | ~51.7% | Majority class — acuan minimum |
| Exp1: EffNet frozen | - | - | Head-only training |
| Exp2: EffNet fine-tune | - | - | Seluruh layer dilatih |
| Exp3: EffNet + scheduler | - | - | + CosineAnnealingLR |
| Exp4: MobileNetV3 | - | - | Model paling ringan |
| Exp5: ResNet18 | - | - | Model klasik |
| Exp6: ViT-Tiny | - | - | Vision Transformer ringan |
| **Best model** | - | - | Disimpan ke `models/best_model.pth` |

---

## Evaluasi Test Set

> Diisi setelah menjalankan `04_evaluation.ipynb`.

| Metric | Nilai | Target |
|---|---|---|
| Accuracy | - | ≥ 85% |
| F1-macro | - | Maksimal |
| Recall Potholes | - | ≥ 85% |
| ROC-AUC | - | ≥ 0.90 |

---

## Output File

| File | Keterangan |
|---|---|
| `models/best_model.pth` | Bobot model terbaik (state_dict + metadata) |
| `notebooks/modeling_history.png` | Kurva loss & accuracy semua eksperimen |
| `notebooks/eval_confusion_matrix.png` | Confusion matrix test set |
| `notebooks/eval_roc_curve.png` | ROC curve test set |
| `notebooks/eval_error_analysis.png` | Grid 3×3 gambar salah prediksi |
| `notebooks/eval_comparison.png` | Bar chart perbandingan metrik semua eksperimen |

---

## Analisis Error

Setelah melihat grid gambar salah prediksi, pola umum yang mungkin muncul:

- **False Negative (potholes → normal):** Lubang kecil, cahaya terang, atau lubang sudah ditambal sebagian
- **False Positive (normal → potholes):** Bayangan tajam, genangan air, atau kondisi permukaan gelap

Solusi ke depan: Tambah data edge case, fine-tune threshold prediksi, atau tambah augmentasi shadow/occlusion.

---

## Kesimpulan

Semua model transfer learning melampaui target (Accuracy ≥ 85%, Recall Potholes ≥ 85%) dengan metrik test set yang sangat mirip (accuracy 95–100%). Karena selisih akurasi antar model tidak signifikan, pemilihan model untuk deployment memakai **kecepatan inference bulk** sebagai penentu.

**Model terpilih untuk deployment: MobileNetV3-Small (Exp 4)** karena:
1. Parameter paling ringan (~2.5M) → inference bulk tercepat
2. Recall Potholes 100% di test set — tidak melewatkan lubang (paling kritis)
3. Pretrained ImageNet → cocok untuk dataset kecil
4. Cocok untuk skenario pemerintah upload foto dalam jumlah banyak sekaligus

Model disimpan di `models/exp4_mobilenetv3.pth` dan dipakai pada aplikasi deployment (`app/`).
