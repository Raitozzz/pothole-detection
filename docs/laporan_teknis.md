# Laporan Teknis — Pemodelan & Evaluasi
**Proyek:** Pothole Detection System  
**Tanggal:** Juni 2026

---

## 1. Ringkasan

Laporan ini berfokus pada pembuatan model machine learning untuk mendeteksi lubang jalan (pothole) dari gambar. Pendekatan yang digunakan adalah **transfer learning** — memanfaatkan model yang sudah dilatih pada dataset besar (ImageNet) dan mengadaptasinya ke domain pothole detection.

Seluruh proses mencakup:
- Memilih dan membandingkan 4 arsitektur model
- Melakukan 6 eksperimen dengan konfigurasi hyperparameter berbeda
- Mengevaluasi semua model di test set dengan berbagai metrik
- Memilih model terbaik berdasarkan kecepatan inference untuk deployment

---

## 2. Mengapa Transfer Learning?

Dataset yang tersedia hanya **681 gambar** (352 normal, 329 potholes). Jumlah ini terlalu kecil untuk melatih model dari nol (from scratch) karena:
- Model deep learning butuh ribuan hingga jutaan data untuk konvergen
- Training dari nol pada data kecil hampir pasti overfitting

**Transfer learning** menyelesaikan masalah ini dengan cara:
1. Mengambil model yang sudah dilatih pada ImageNet (1.2 juta gambar, 1000 kelas)
2. Backbone-nya sudah bisa mengenali fitur visual umum: tepi, tekstur, pola, bentuk
3. Hanya bagian **classifier head** (layer terakhir) yang diganti dan dilatih ulang untuk 2 kelas: normal vs potholes

---

## 3. Model yang Dipilih

### 3.1 Baseline — Majority Classifier
Selalu memprediksi kelas terbanyak (normal). Digunakan sebagai acuan minimum yang wajib dilampaui semua model.

- Accuracy: ~51.7%
- F1-macro: ~0.34
- Recall Potholes: 0.0% (tidak pernah deteksi pothole)

### 3.2 EfficientNet-B0
- **Sumber:** torchvision (pretrained ImageNet)
- **Parameter:** ~5.3 juta
- **Alasan dipilih:** Arsitektur yang dirancang khusus untuk efisiensi — akurasi tinggi dengan parameter sedikit menggunakan teknik compound scaling (depth, width, resolution dioptimasi bersama)
- **Modifikasi:** Layer `classifier[1]` diganti dari `Linear(1280, 1000)` menjadi `Linear(1280, 2)`
- **Diuji dalam 3 eksperimen:** frozen, fine-tune, fine-tune + scheduler

### 3.3 MobileNetV3-Small
- **Sumber:** torchvision (pretrained ImageNet)
- **Parameter:** ~2.5 juta (paling ringan)
- **Alasan dipilih:** Model paling ringan, dirancang untuk perangkat mobile/edge. Cocok dibandingkan untuk mengetahui apakah model ringan bisa kompetitif
- **Modifikasi:** Layer `classifier[3]` diganti menjadi `Linear(1024, 2)`

### 3.4 ResNet18
- **Sumber:** torchvision (pretrained ImageNet)
- **Parameter:** ~11.7 juta
- **Alasan dipilih:** Arsitektur klasik dengan residual connection yang terbukti stabil. Digunakan sebagai pembanding model yang lebih besar
- **Modifikasi:** Layer `fc` diganti menjadi `Linear(512, 2)`

### 3.5 ViT-Tiny (Vision Transformer)
- **Sumber:** timm (`vit_tiny_patch16_224.augreg_in21k_ft_in1k`)
- **Parameter:** ~5.7 juta
- **Alasan dipilih:** Representasi arsitektur transformer untuk computer vision — berbeda dari CNN. Pretrained ImageNet-21k (dataset lebih besar) lalu fine-tuned ke ImageNet-1k
- **Modifikasi:** `num_classes` diset ke 2 saat inisialisasi model

---

## 4. Fine-Tuning — Cara dan Apa yang Dituning

### 4.1 Dua Strategi Fine-Tuning

**Frozen Backbone (Exp 1):**
- Semua layer backbone di-freeze (`requires_grad = False`)
- Hanya classifier head yang dilatih
- Keuntungan: cepat, tidak perlu GPU besar
- Kelemahan: fitur ImageNet tidak beradaptasi ke domain pothole

**Full Fine-Tuning (Exp 2–6):**
- Semua layer dilatih termasuk backbone
- Learning rate kecil (1e-4) agar bobot pretrained tidak rusak drastis
- Keuntungan: model bisa belajar fitur spesifik domain pothole
- Kelemahan: lebih lambat, perlu monitoring overfitting

### 4.2 Hyperparameter yang Dituning

| Hyperparameter | Nilai yang Dicoba | Keterangan |
|---|---|---|
| Learning Rate | 1e-3, 1e-4 | 1e-3 untuk frozen, 1e-4 untuk fine-tune |
| Freeze Strategy | Frozen, Full fine-tune | Frozen = hanya head, Full = semua layer |
| LR Scheduler | Tidak, CosineAnnealingLR | Scheduler membantu konvergensi di akhir |
| Epochs | 10, 20 | Frozen cukup 10, fine-tune butuh 20 |
| Batch Size | 32 | Fixed — balance antara noise dan VRAM |
| Optimizer | Adam | Adaptif, konvergen cepat pada data kecil |

### 4.3 Teknik Pendukung

**Early Stopping (patience=5):**
Jika composite score (0.4×F1 + 0.6×Recall Potholes) tidak membaik selama 5 epoch berturut-turut, training dihentikan otomatis. Mencegah overfitting dan menghemat waktu.

**Class Weight pada Loss:**
Karena ada imbalance ringan (normal 352 vs potholes 329, ratio 1.07:1), class weight diterapkan pada CrossEntropyLoss agar model tidak bias ke kelas mayoritas.

**CosineAnnealingLR (Exp 3):**
Learning rate diturunkan secara gradual mengikuti kurva cosine dari nilai awal hingga mendekati 0. Membantu model keluar dari local minima dan konvergen lebih stabil di akhir training.

---

## 5. Augmentasi & Normalisasi

### 5.1 Augmentasi (hanya Train — real-time di memori)

Augmentasi tidak disimpan ke disk. Setiap epoch model melihat variasi gambar yang berbeda sehingga lebih robust terhadap kondisi bervariasi.

| Transform | Parameter | Alasan |
|---|---|---|
| RandomHorizontalFlip | p=0.5 | Foto jalan simetris kiri-kanan |
| RandomVerticalFlip | p=0.2 | Orientasi atas-bawah bermakna, flip jarang |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.2 | Foto diambil dalam berbagai kondisi cahaya |
| RandomRotation | ±15° | Kamera tidak selalu horizontal sempurna |

Val dan test set **tidak** mendapat augmentasi agar evaluasi mencerminkan kondisi nyata.

### 5.2 Normalisasi (semua split — train, val, test)

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Menggunakan statistik ImageNet karena semua model (EfficientNet, ResNet, MobileNetV3, ViT) di-pretrain pada ImageNet. Input yang dinormalisasi dengan statistik yang sama memastikan distribusi piksel sesuai dengan yang diharapkan backbone.

Normalisasi dilakukan **setelah** augmentasi, sebagai langkah terakhir sebelum tensor masuk ke model.

---

## 6. Metrik Evaluasi

Model dievaluasi menggunakan beberapa metrik karena satu metrik saja tidak cukup:

| Metrik | Penjelasan | Mengapa Penting |
|---|---|---|
| Accuracy | % prediksi benar dari total | Gambaran umum performa |
| Precision (macro) | Dari yang diprediksi pothole, berapa yang benar | Ukur false alarm |
| Recall (macro) | Dari semua pothole nyata, berapa yang terdeteksi | Ukur miss detection |
| F1-macro | Rata-rata harmonis precision & recall | Balance keduanya |
| Recall Potholes | Recall khusus kelas potholes | **Paling kritis** — miss pothole berbahaya |
| ROC-AUC | Kemampuan diskriminasi model di semua threshold | Robust terhadap class imbalance |
| Composite Score | 0.4×F1 + 0.6×Recall Potholes | Untuk early stopping & ranking model |

**Recall Potholes diberi bobot terbesar** karena dalam konteks keselamatan jalan, melewatkan lubang (false negative) jauh lebih berbahaya daripada salah mendeteksi jalan mulus sebagai berlubang (false positive).

---

## 7. Confidence Score & Threshold

Setiap prediksi menghasilkan **confidence score** — seberapa yakin model terhadap prediksinya (bukan hanya untuk kelas pothole, tapi untuk kelas apapun yang dipilih model).

**Cara hitung:**
```
softmax(logits) → probabilitas tiap kelas
confidence = probabilitas kelas yang diprediksi
```

**Threshold 70%:**
- Confidence ≥ 70% → **Terverifikasi** (model yakin, langsung diproses)
- Confidence < 70% → **Perlu Tinjauan** (model ragu, perlu verifikasi manual)

---

## 8. Pemilihan Best Model

Karena semua model menghasilkan metrik yang sangat mirip di test set (accuracy 95–100%), pemilihan best model menggunakan **kecepatan inference bulk** sebagai tiebreaker — model dengan waktu proses 100 gambar tercepat dipilih untuk deployment.

Hal ini relevan karena pada deployment, pemerintah akan upload foto dalam jumlah banyak sekaligus (bulk), sehingga kecepatan lebih kritis daripada perbedaan akurasi yang tidak signifikan.

---

## 9. Hasil Test Set

| Eksperimen | Accuracy | F1-macro | Rec Potholes | ROC-AUC | Status |
|---|---|---|---|---|---|
| Baseline | 0.5146 | 0.3397 | 0.0000 | 0.5000 | REVIEW |
| Exp1: EffNet frozen | 0.9515 | 0.9514 | 0.9400 | 0.9936 | PASS |
| Exp2: EffNet fine-tune | 0.9903 | 0.9903 | 1.0000 | 1.0000 | PASS |
| Exp3: EffNet+scheduler | 0.9709 | 0.9708 | 0.9600 | 0.9951 | PASS |
| Exp4: MobileNetV3 | 0.9515 | 0.9514 | 1.0000 | 0.9989 | PASS |
| Exp5: ResNet18 | 0.9806 | 0.9806 | 0.9800 | 0.9992 | PASS |
| Exp6: ViT-Tiny | 1.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |

> Target: Accuracy ≥ 85% dan Recall Potholes ≥ 85%. Semua model transfer learning melampaui target.

---

## 10. Output File

| File | Keterangan |
|---|---|
| `notebooks/03_modeling.ipynb` | Training 6 eksperimen + simpan tiap model |
| `notebooks/04_evaluation.ipynb` | Evaluasi semua model di test set |
| `models/exp1_effnet_frozen.pth` | Bobot Exp 1 |
| `models/exp2_effnet_finetune.pth` | Bobot Exp 2 |
| `models/exp3_effnet_scheduler.pth` | Bobot Exp 3 |
| `models/exp4_mobilenetv3.pth` | Bobot Exp 4 |
| `models/exp5_resnet18.pth` | Bobot Exp 5 |
| `models/exp6_vit_tiny.pth` | Bobot Exp 6 |
| `models/best_model.pth` | Model terbaik (composite score val set) |
| `notebooks/modeling_history.png` | Kurva loss, acc, F1, recall per epoch |
| `notebooks/eval_confusion_matrix.png` | Confusion matrix semua model |
| `notebooks/eval_roc_curve.png` | ROC curve semua model |
| `notebooks/eval_comparison.png` | Bar chart perbandingan metrik |
| `notebooks/eval_error_analysis.png` | Gambar salah prediksi (best model) |
| `notebooks/eval_confidence_dist.png` | Distribusi confidence + threshold 70% |

---

## 11. Keterbatasan

1. **Dataset kecil (681 gambar)** — hasil akurasi sangat tinggi (mendekati 1.0) bisa jadi karena dataset terlalu mudah untuk model pretrained, bukan karena model benar-benar sempurna
2. **Test set hanya 102 gambar** — selisih 1 gambar = ~1% akurasi, angka punya variance tinggi
3. **Satu domain** — semua foto dari sumber yang sama, model belum tentu generalize ke kondisi jalan yang berbeda (cuaca, sudut kamera, jenis aspal)
4. **ViT-Tiny skor 1.0** — nilai sempurna pada dataset kecil patut dicurigai, lebih aman pilih model dengan skor sedikit di bawah 1.0 untuk deployment
5. **Model hanya mengenal 2 kelas (normal vs pothole)** — model dilatih sebagai klasifikasi biner sehingga *dipaksa* memilih salah satu kelas untuk gambar apa pun, termasuk gambar yang sama sekali bukan jalan (meja, hewan, dll.). Aplikasi ini mengasumsikan pengguna hanya mengunggah foto jalan (normal atau berlubang). Penanganan gambar di luar dua kelas tersebut belum dicakup dan menjadi bagian penelitian selanjutnya (lihat §12).

---

## 12. Penelitian Selanjutnya

1. **Menjadikan relevansi gambar sebagai bagian dari model yang dilatih.** Beberapa opsi:
   - **Kelas ketiga "bukan jalan / tidak relevan"** — ubah dari klasifikasi biner menjadi multi-kelas, dilatih pada data negatif yang beragam (indoor, kendaraan close-up, vegetasi, wajah, dll.) agar model bisa menolak gambar irrelevant secara terlatih.
   - **Open-set recognition / OOD detection** — model belajar mengenali bahwa suatu input berada di luar distribusi data latih dan menahan prediksi, alih-alih memaksakan salah satu kelas.
2. **Memperluas kelas jalan itu sendiri** — mis. membedakan jenis kerusakan (retak/*crack*, lubang dangkal vs dalam, jalan tergenang) agar prioritas perbaikan lebih granular daripada sekadar normal vs pothole.
3. **Menambah data lintas domain** — kondisi cuaca, sudut kamera, jenis aspal, dan waktu pengambilan yang berbeda untuk menguji dan meningkatkan generalisasi.
