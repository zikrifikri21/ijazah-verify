# Ijazah Verifier — Prototype Verifikasi Ijazah (OCR + Deteksi Tanda Tangan)

Mini project pengolahan citra digital. Sistem menerima citra ijazah lalu menghasilkan:

```
Input : ijazah_001.jpg
Output:
Nomor Ijazah : 571012022000056
Tanda Tangan : PRESENT
```

## 1. Cara menjalankan

### Prasyarat
- Python 3.9+
- **Tesseract OCR** (program terpisah, bukan paket pip)

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows: unduh installer dari https://github.com/UB-Mannheim/tesseract/wiki
# lalu pastikan tesseract.exe masuk PATH
```

### Instalasi
```bash
git clone <URL-REPOSITORY-ANDA>
cd ijazah-verifier
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Verifikasi satu citra
```bash
python src/verify.py data/samples/01_HighQuality_Enhanced.jpg
python src/verify.py ijazah_001.jpg -v --debug results/debug   # -v: detail, --debug: simpan crop & mask
```
Opsi: `--rotate {auto,0,90,180,270}` (default `auto`) dan `--method <nama>` (metode enhancement untuk area nomor).

### Mengulang eksperimen CER
```bash
python src/evaluate.py --samples data/samples --truth 571012022000056 --out results
```
Keluaran di folder `results/`: `cer_summary.csv`, `cer_matrix.csv`, `stress_matrix.csv`, `signature_test.csv`, `evaluation_log.txt`.

## 2. Struktur repositori
```
src/enhance.py     17 metode enhancement yang dibandingkan
src/ocr.py         OCR Tesseract + perhitungan CER (Levenshtein)
src/signature.py   deteksi tanda tangan (threshold -> morfologi -> komponen terhubung)
src/verify.py      pipeline + CLI
src/evaluate.py    eksperimen CER, uji stres, kontrol negatif tanda tangan
data/samples/      9 citra uji (berbagai degradasi)
results/           hasil eksperimen
```

## 3. Pipeline dan metode

```
Citra -> (koreksi orientasi) -> Grayscale -> Enhancement global (contrast stretching)
      |-> Crop area nomor -> Enhancement (median filter) -> OCR Tesseract -> Nomor ijazah
      |-> Crop area TTD (Rektor, Dekan) -> Thresholding -> Morfologi -> Deteksi TTD
                                  \-> Hasil verifikasi
```

| Tahap | Metode | Alasan |
|---|---|---|
| Orientasi | Citra tersimpan portrait → diputar 90° searah jarum jam | Seluruh sampel berisi dokumen landscape yang tersimpan miring |
| Grayscale | `cv2.cvtColor` BGR→GRAY | Membuang pengaruh warna latar (kuning/hijau/tint) |
| Enhancement global | Contrast stretching persentil 1–99 | Memulihkan citra pudar/gelap tanpa memperkuat noise |
| Lokasi area | ROI berupa **pecahan** lebar/tinggi citra | Tahan terhadap perbedaan resolusi (1132 px s.d. 2481 px) |
| Enhancement nomor | Median filter 3×3 | Terbaik pada eksperimen CER (bagian 4) |
| OCR | Tesseract (`--psm 7`, whitelist karakter), ROI dinormalkan ke tinggi 96 px, lalu diambil deret digit terpanjang | Satu baris teks; whitelist menekan salah baca huruf↔angka |
| Thresholding TTD | Background flattening (pembagian dengan estimasi latar) + ambang Otsu **yang dibatasi maksimum 190** | Lihat catatan di bawah |
| Morfologi TTD | Opening 2×2 (buang bintik) lalu closing 7×7 (sambung goresan putus) | Goresan tanda tangan menjadi satu komponen besar |
| Keputusan TTD | Ada komponen besar dengan lebar ≥ 35% ROI dan rasio tinta ≥ 0,4% | Teks cetak terpecah menjadi komponen-komponen kecil |

Catatan penting: pada versi awal, Otsu murni dipakai tanpa batas. Uji kontrol negatif (tanda tangan dihapus) menunjukkan detektor tetap menyatakan "ada" pada banyak citra, karena Otsu selalu membelah histogram meskipun area hanya berisi noise kertas. Ambang dibatasi supaya tinta wajib jauh lebih gelap dari latar. Setelah perbaikan, kontrol negatif lulus.

Karena ijazah universitas ditandatangani **Rektor dan Dekan**, kedua area diperiksa. `PRESENT` hanya bila keduanya terdeteksi.

## 4. Metode enhancement mana yang paling efektif (berdasarkan CER)

CER = jarak Levenshtein(hasil OCR, ground truth) / panjang ground truth. Ground truth: `571012022000056`.

Dua percobaan dilakukan pada area nomor (`results/cer_summary.csv`):
1. **9 citra sampel** (low contrast, blur, noise, low resolution, faded, color shift, JPEG, kombinasi).
2. **Uji stres**: 14 degradasi sintetis (noise σ=15/30/50, blur σ=1.5/2.5/3.5, kontras rendah, gelap, resolusi rendah, JPEG q=15/6) pada citra 01, seed tetap.

| Metode | CER rata-rata, 9 sampel | CER rata-rata, uji stres | Stres tanpa kesalahan |
|---|---|---|---|
| **median 3×3** | **0,000** | **0,000** | **14/14** |
| gamma / gaussian | 0,000 | 0,005 | 13/14 |
| tanpa enhancement | 0,000 | 0,052 | 12/14 |
| contrast stretching | 0,000 | 0,067 | 12/14 |
| CLAHE | 0,000 | 0,071 | 13/14 |
| unsharp mask | 0,000 | 0,138 | 11/14 |
| gabungan (median+stretch+CLAHE+unsharp+2×) | 0,000 | 0,157 | 10/14 |
| adaptive threshold | 0,111 | 0,229 | 8/14 |
| histogram equalization | 1,000 | 0,995 | 0/14 |

Kesimpulan:
- **Median filter paling efektif** dan satu-satunya yang tanpa kesalahan di kedua percobaan. Gangguan utama pada sampel adalah noise dan tekstur kertas, dan median meredamnya tanpa mengaburkan tepi angka.
- **Histogram equalization paling buruk** (CER ≈ 1). Dugaan penyebab: area nomor didominasi latar kertas, sehingga equalization memperkuat tekstur kertas menjadi bintik gelap yang menutupi angka. Dugaan ini belum diverifikasi secara visual.
- Metode yang menajamkan atau memperkuat kontras lokal (unsharp, CLAHE, gabungan, adaptive threshold) cenderung **memperburuk** CER karena ikut memperkuat noise.
- Pengujian kombinasi tahap global × tahap ROI (`results/pipeline_combo.csv`) menunjukkan CLAHE pada tahap global menaikkan CER uji stres menjadi sekitar 0,15, sedangkan contrast stretching tidak merugikan. Karena itu pipeline memakai **contrast stretching** untuk enhancement global dan **median** untuk area nomor.

## 5. Hasil pada 9 sampel
Semua 9 citra menghasilkan `Nomor Ijazah : 571012022000056` (CER 0) dan `Tanda Tangan : PRESENT`.

Uji kontrol negatif tanda tangan (area dihapus secara sintetis, 9 citra): tanda tangan asli terdeteksi 9/9 untuk Rektor dan Dekan; setelah salah satunya dihapus, area yang dihapus terdeteksi 0/9 dan area lainnya tetap 9/9.

## 6. Keterbatasan (penting dibaca)
- **Hanya satu dokumen.** Kesembilan citra adalah satu ijazah yang sama dengan degradasi berbeda, sehingga hasil menunjukkan ketahanan terhadap degradasi, bukan kemampuan generalisasi ke ijazah lain.
- **Tidak ada sampel negatif asli.** Semua sampel bertanda tangan. Kemampuan menyatakan `ABSENT` hanya diuji dengan penghapusan sintetis, bukan dokumen tanpa tanda tangan sungguhan.
- **ROI tetap** (pecahan posisi) hanya cocok untuk templat ijazah ini. Templat lain perlu ROI baru atau deteksi tata letak otomatis.
- **Koreksi orientasi** hanya heuristik (portrait → putar 90°). Gunakan `--rotate` bila citra miring ke arah lain.
- **Selisih CER antar metode papan atas kecil.** Uji stres memakai satu citra dasar dan satu seed, jadi perbedaan sekitar 0,05 tidak signifikan secara statistik. Temuan yang kuat: median aman, histogram equalization dan adaptive threshold buruk.
- **Deteksi tanda tangan hanya memeriksa keberadaan goresan tinta**, bukan keaslian. Coretan acak atau tanda tangan yang ditempel digital juga akan lolos. Prototipe ini bukan alat pemeriksa keabsahan ijazah.
