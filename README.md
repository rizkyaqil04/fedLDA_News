# 🐍 FedLDA Topic Modeling Project

Proyek ini menggunakan Python dan FastAPI untuk menjalankan server dan klien secara terpisah. Ikuti langkah-langkah berikut untuk menjalankan aplikasi ini di lingkungan lokal Anda.

---

## ✅ Persyaratan

Sebelum menjalankan aplikasi, pastikan Anda telah menginstal:

- **Python** versi **3.11** atau **3.12**
- **pip** (Python package manager)

---

## ⚙️ Instalasi Dependensi

Untuk menginstal semua dependensi yang dibutuhkan, jalankan perintah berikut di terminal:

```bash
pip install -r requirements.txt
```

Pastikan Anda menjalankannya dari direktori root proyek (folder utama repositori)

## 🚀 Menjalankan Aplikasi

### 1. Menjalankan Server FastAPI

Buka terminal pertama, lalu jalankan perintah berikut dari direktori root proyek:
```
uvicorn main:app --host 0.0.0.0 --port 8000
```
Server FastAPI akan aktif dan dapat diakses melalui http://0.0.0.0:8000 atau http://localhost:8000

### 2. Menjalankan Klien

Setelah server berjalan, buka terminal lain (atau tab terminal baru), lalu jalankan perintah berikut dari direktori yang sama:
```
python run_clients.py
```
Skrip ini akan menjalankan proses klien yang berinteraksi dengan server
