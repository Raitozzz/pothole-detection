# 🛣️ Pothole Detection System

[![CI](https://github.com/Raitozzz/pothole-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Raitozzz/pothole-detection/actions/workflows/ci.yml)

**Deteksi lubang jalan dari foto secara otomatis — lalu tahu mana yang perlu diperbaiki lebih dulu.**

Cukup unggah foto jalan, sistem langsung mengenali apakah jalan itu mulus atau berlubang, mengelompokkan laporan berdasarkan lokasi, dan menampilkan peta prioritas perbaikan. Dibuat untuk membantu menilai kondisi jalan dengan cepat, tanpa harus memeriksa satu per satu secara manual.

---

## ✨ Yang Bisa Dilakukan

- **Deteksi otomatis** — unggah satu foto atau banyak sekaligus, hasilnya muncul seketika.
- **Tingkat keyakinan** — setiap hasil menampilkan seberapa yakin sistem, sehingga yang meragukan bisa ditinjau ulang.
- **Anti duplikat** — foto yang sama tidak akan dihitung dua kali.
- **Peta prioritas** — lokasi dengan lubang terbanyak ditandai agar diperbaiki lebih dulu.
- **Riwayat tersimpan** — hasil deteksi disimpan di database sehingga tidak hilang saat aplikasi ditutup.
- **Ekspor laporan** — seluruh hasil bisa diunduh dalam bentuk file CSV.

---

## 🚀 Cara Memakai

1. Buka aplikasi.
2. Unggah foto jalan (boleh langsung banyak).
3. Lihat hasil deteksi beserta tingkat keyakinannya.
4. Buka tab **Dashboard** untuk melihat peta dan daftar prioritas perbaikan.
5. Unduh laporan bila diperlukan.

> Foto yang menyimpan data lokasi (GPS) akan otomatis muncul di peta. Jika tidak ada, lokasi bisa diisi secara manual.

---

## 💻 Menjalankan di Komputer Sendiri

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

Aplikasi akan terbuka di browser pada alamat `http://localhost:8501`.

Tersedia juga REST API untuk deteksi lewat program lain:

```bash
uvicorn api.server:app --reload
```

Dokumentasi API interaktif dapat dibuka di `http://localhost:8000/docs`.

---

## 📂 Isi Proyek

| Folder | Isi |
|---|---|
| `app/` | Aplikasi web |
| `api/` | REST API deteksi |
| `models/` | Model deteksi yang sudah dilatih |
| `notebooks/` | Proses dari eksplorasi data hingga pelatihan model |
| `tests/` | Pengujian otomatis |
| `docs/` | Laporan dan dokumentasi |

---

## 🧠 Sekilas Cara Kerjanya

Sistem ini belajar mengenali ciri jalan berlubang dari ratusan contoh foto. Ketika foto baru masuk, sistem membandingkannya dengan yang sudah dipelajari lalu memutuskan apakah jalan itu mulus atau berlubang — mirip cara manusia mengenali sesuatu setelah sering melihat contohnya.

---

*Dibuat sebagai proyek pembelajaran deteksi kerusakan jalan berbasis citra.*
