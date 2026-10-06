# Sistem Cosine Similarity — Streamlit

Sistem ini tetap menggunakan **rumus dan proses Cosine Similarity yang sama** seperti tugas:
- representasi kata biner 0/1
- `X · Y` = jumlah kata yang sama
- `|X| = √(jumlah kata pada X)`
- `|Y| = √(jumlah kata pada Y)`
- `Cos(X,Y) = (X · Y) / (|X| × |Y|)`

## Input yang tersedia

### 1. Input Manual
User dapat memasukkan jumlah kalimat dan teks kalimat secara langsung.

### 2. Upload PDF
User dapat mengunggah PDF. Sistem:
- mencoba membaca text layer PDF;
- jika PDF berupa scan/gambar, mencoba OCR Tesseract;
- menampilkan teks hasil pembacaan PDF untuk diperiksa;
- kemudian menggunakan kalimat yang berhasil dikenali sebagai data perhitungan.

### 3. Data Contoh PDF
Memuat 5 kalimat contoh S1–S5 dari tugas.

## Menjalankan

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

Upload:
- `app.py`
- `requirements.txt`
- `packages.txt`

ke repository GitHub, kemudian deploy `app.py`.

`packages.txt` dipakai untuk menyediakan Tesseract OCR di server Streamlit.

## Catatan
Untuk PDF hasil scan, kualitas OCR dapat memengaruhi hasil pembacaan kata. Sistem menyediakan area "Lihat teks hasil pembacaan PDF" agar hasil ekstraksi dapat diperiksa sebelum perhitungan.
