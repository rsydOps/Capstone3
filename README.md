# Prediksi Permintaan Sepeda Sewa per Jam (Bike Sharing Demand Forecasting)

Model regresi untuk memprediksi jumlah penyewaan sepeda per jam (`cnt`), dibangun sebagai dasar keputusan operasional rebalancing armada dan penjadwalan staf pada perusahaan bike-sharing.

## Ringkasan

Tim operasional pada perusahaan bike-sharing saat ini tidak memiliki cara sistematis untuk mengantisipasi permintaan penyewaan sepeda per jam, sehingga muncul dua jenis kerugian: kekurangan sepeda pada jam permintaan tinggi dan kelebihan sepeda pada jam permintaan rendah. Proyek ini menjawab masalah tersebut dengan membangun pipeline regresi berbasis XGBoost, dituning menggunakan `RandomizedSearchCV` dengan skema validasi `TimeSeriesSplit`, dan dilengkapi dua perbaikan yang teruji lewat walk-forward validation: fitur tren waktu (`days_since_start`) yang menangkap kenaikan permintaan tahunan sebesar kurang lebih 61 persen dari 2011 ke 2012, serta koreksi margin khusus jam commute yang diselaraskan dengan struktur biaya operasional, bukan sekadar RMSE.

Dievaluasi pada test set berbasis waktu (20 persen data terakhir, Agustus sampai Desember 2012), model titik-prediksi mencatatkan RMSE 70,29 dan MAE 43,50 (R² 0,898), jauh di bawah baseline rata-rata per jam (RMSE 165,84, MAE 112,35). Setelah koreksi margin commute diterapkan, RMSE dan MAE model final sedikit meningkat menjadi 84,88 dan 51,17, sebuah trade-off yang disengaja karena margin dioptimalkan untuk menekan biaya dolar, bukan error statistik. Hasilnya, simulasi biaya operasional pada test set yang sama menunjukkan penurunan biaya sebesar 82,5 persen dibanding kondisi tanpa model.

## Daftar Isi

1. [Latar Belakang dan Rumusan Masalah](#1-latar-belakang-dan-rumusan-masalah)
2. [Dataset](#2-dataset)
3. [Metodologi](#3-metodologi)
4. [Hasil dan Evaluasi](#4-hasil-dan-evaluasi)
5. [Keterbatasan](#5-keterbatasan)
6. [Rekomendasi Pengembangan Selanjutnya](#6-rekomendasi-pengembangan-selanjutnya)
7. [Struktur Direktori](#7-struktur-direktori)
8. [Cara Menjalankan / Reproduksi](#8-cara-menjalankan--reproduksi)
9. [Cara Menggunakan Model Tersimpan](#9-cara-menggunakan-model-tersimpan)
10. [Berkas dalam Repository](#10-berkas-dalam-repository)

## 1. Latar Belakang dan Rumusan Masalah

**Stakeholder.** Tim Operations / Fleet Management pada perusahaan bike-sharing, yang bertanggung jawab mendistribusikan ulang armada sepeda antar stasiun dan mengatur jumlah staf operasional per jam.

**Permasalahan.** Tanpa kemampuan mengantisipasi demand per jam, tim operasional bereaksi setelah kondisi terjadi, bukan mengantisipasinya. Ini menimbulkan dua jenis kerugian: kehilangan potensi pendapatan saat sepeda kehabisan di jam sibuk, dan biaya rebalancing yang terbuang saat sepeda menganggur di jam sepi.

**Tujuan.** Membangun model regresi yang memprediksi `cnt` per jam dengan error yang cukup rendah untuk mendukung keputusan rebalancing armada dan penjadwalan staf secara proaktif.

**Pendekatan analitis.** Masalah ini adalah regresi karena target (`cnt`) adalah variabel numerik kontinu. Alur kerja: pemahaman dan pembersihan data, eksplorasi data, pembuatan baseline, pemodelan bertahap dari yang sederhana (OLS) ke yang lebih kompleks (Ridge, Decision Tree, Random Forest, XGBoost), tuning hyperparameter, dan analisis residual untuk memahami kondisi ketika model dapat maupun tidak dapat dipercaya.

**Metrik evaluasi.** Metrik utama adalah RMSE, dipilih karena kesalahan prediksi pada jam permintaan puncak jauh lebih mahal konsekuensinya dibanding kesalahan kecil pada jam sepi, dan RMSE menghukum error besar lebih berat dibanding error kecil. Metrik pendukung: MAE, MAPE, RMSLE, dan R². Definisi "cukup baik" ditentukan lewat proses (seberapa jauh model mengalahkan baseline, seberapa kecil gap train-test, titik diminishing returns), bukan angka yang ditetapkan di awal sebelum melihat data.

## 2. Dataset

**Sumber dan dimensi.** `data_bike_sharing.csv` berisi 12.165 baris data historis penyewaan sepeda per jam, periode 2011 sampai 2012, dengan 11 kolom mentah. Tidak ditemukan baris duplikat maupun missing value pada seluruh kolom.

**Data dictionary**

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `dteday` | string (tanggal) | Tanggal observasi, format YYYY-MM-DD |
| `hr` | int (0-23) | Jam dalam sehari saat observasi dicatat |
| `season` | int (1-4) | Musim: 1 = winter, 2 = spring, 3 = summer, 4 = fall |
| `holiday` | int (0/1) | Apakah hari tersebut hari libur |
| `weathersit` | int (1-4) | Kondisi cuaca, dari 1 (cerah) hingga 4 (buruk/ekstrem), bersifat ordinal |
| `temp` | float (0-1) | Suhu yang sudah dinormalisasi |
| `atemp` | float (0-1) | Suhu terasa (feels-like), sudah dinormalisasi |
| `hum` | float (0-1) | Kelembapan, sudah dinormalisasi |
| `casual` | int | Jumlah penyewa non-terdaftar |
| `registered` | int | Jumlah penyewa terdaftar |
| `cnt` | int | Target prediksi, total penyewaan sepeda per jam |

**Catatan kualitas data.** Terkonfirmasi bahwa `cnt = casual + registered` berlaku secara eksak pada seluruh baris. Karena itu, `casual` dan `registered` dikeluarkan dari daftar fitur prediktor, karena keduanya secara matematis mendefinisikan target dan akan menyebabkan data leakage jika ikut dipakai sebagai input model. Fitur akhir (`X`) yang dipakai sebelum rekayasa fitur berjumlah 8 kolom: `dteday`, `hum`, `weathersit`, `holiday`, `season`, `atemp`, `temp`, `hr`.

## 3. Metodologi

### 3.1 Eksplorasi Data

Distribusi target (`cnt`) condong ke kanan, sebagian besar jam memiliki permintaan rendah dengan sedikit jam bervolume sangat tinggi. Variabel `hr` menunjukkan pola non-linear yang jelas mengikuti jam commute. Temuan paling berpengaruh terhadap keseluruhan proyek adalah tren waktu: rata-rata `cnt` naik dari sekitar 145 pada 2011 menjadi sekitar 233 pada 2012, kenaikan sekitar 61 persen tahun ke tahun.

### 3.2 Skema Train/Test Split

Split dilakukan berbasis waktu, bukan acak: data diurutkan berdasarkan tanggal, lalu 20 persen data paling akhir dijadikan test set. Model dilatih hanya dari masa lalu dan dievaluasi terhadap periode yang belum pernah dilihat, mensimulasikan penggunaan nyata (meramalkan ke depan). Keputusan ini diambil karena tren pertumbuhan tahunan yang kuat membuat split acak berpotensi membocorkan informasi periode "demand tinggi" ke train maupun test sekaligus, membuat skor test terlihat lebih baik dari performa sesungguhnya.

- Train: 9.715 baris (80 persen), 1 Januari 2011 sampai 6 Agustus 2012
- Test: 2.450 baris (20 persen), 7 Agustus 2012 sampai 31 Desember 2012

Validasi tidak berhenti pada satu split. Keputusan-keputusan besar (fitur mana yang benar-benar membantu, koreksi mana yang benar-benar menurunkan biaya) diverifikasi ulang lewat walk-forward validation pada beberapa potongan waktu berurutan, untuk memastikan efeknya konsisten dan bukan kebetulan satu kali percobaan.

### 3.3 Rekayasa Fitur

- `dteday` diuraikan menjadi `day_of_week`, `month`, `is_weekend`, dan `days_since_start` (fitur tren, pengganti variabel tahun yang tidak tersedia di dataset ini)
- `hr` diubah menjadi cyclical encoding (`hr_sin`, `hr_cos`) agar jarak antar jam (misalnya jam 23 ke jam 0) direpresentasikan secara kontinu
- `temp`, `atemp`, `hum` dipakai langsung (passthrough) karena sudah ternormalisasi pada rentang 0-1
- `season` dan `weathersit` di-encode dengan `OneHotEncoder`
- Fitur commute: `is_rush_hour` (jam 7, 8, 9, 17, 18, 19) dan `is_commute` (rush hour pada hari kerja, bukan hari libur)

Seluruh transformasi dirangkai dalam satu `Pipeline` scikit-learn (`ColumnTransformer` + `FunctionTransformer`), menghasilkan 31 kolom numerik dari 8 kolom mentah.

### 3.4 Pemodelan dan Tuning

Lima kandidat algoritma dibandingkan pada parameter default: Linear Regression/OLS, Ridge, Decision Tree, Random Forest, dan XGBoost. Tiga kandidat terbaik (XGBoost, Random Forest, Decision Tree) dilanjutkan ke tahap tuning.

Grid pencarian awal (`GridSearchCV`, rentang sempit) terbukti kalah dari parameter default XGBoost pada dua kali percobaan. Ruang pencarian kemudian diperlebar menggunakan `RandomizedSearchCV` (25 iterasi) dengan skema cross-validation `TimeSeriesSplit`, konsisten dengan keputusan split berbasis waktu. Dengan ruang pencarian yang lebih luas, tuning berhasil mengalahkan default, meski dengan margin tipis (RMSE turun dari 71,85 menjadi 70,29, sekitar 2,2 persen).

Hyperparameter XGBoost terpilih:

| Parameter | Nilai |
|---|---|
| n_estimators | 400 |
| max_depth | 7 |
| learning_rate | 0,03 |
| subsample | 0,9 |
| colsample_bytree | 0,7 |
| min_child_weight | 10 |
| reg_alpha | 0,01 |
| reg_lambda | 2 |

Model titik-prediksi final adalah XGBoost dengan hyperparameter di atas, dilatih di atas pipeline preprocessing yang menyertakan fitur tren dan fitur commute, dipilih karena marjin kemenangan tuning konsisten dan fitur `days_since_start` terbukti menang pada seluruh fold walk-forward validation.

### 3.5 Koreksi Margin Jam Commute

RMSE sebagai metrik pemilihan model tidak sepenuhnya identik dengan biaya bisnis riil: kehilangan satu ride jauh lebih mahal daripada satu sepeda menganggur. Sebagai koreksi, prediksi model pada baris jam commute dikalikan dengan sebuah margin (multiplier), dipilih dari validation set terpisah (bukan test set) dengan mencari nilai yang meminimalkan biaya total simulasi. Margin terpilih adalah 1,2, dan divalidasi ulang lewat walk-forward validation (menang pada sebagian besar fold, dengan margin optimal bervariasi antara sekitar 1,1 hingga 1,4 antar fold).

Regresi kuantil global sempat dicoba sebagai pendekatan alternatif namun ditolak (lihat 3.6), karena margin yang diterapkan seragam ke semua jam menyebabkan model membesar-besarkan prediksi pada jam sepi.

### 3.6 Eksplorasi yang Diuji dan Ditolak

Enam ide perbaikan lain diuji secara eksplisit dan ditolak setelah divalidasi ulang, dicatat di sini agar keputusan akhir tidak terlihat seolah satu-satunya jalan yang dicoba adalah yang berhasil.

| # | Ide | Hasil Pengujian | Keputusan |
|---|---|---|---|
| 1 | One-hot encoding untuk `hr` (24 kategori) menggantikan cyclical encoding | RMSE walk-forward lebih buruk (naik dari sekitar 69 ke sekitar 83 pada satu fold uji) | Ditolak |
| 2 | Kombinasi cyclical encoding dan one-hot `hr` sekaligus | Tidak lebih baik dari cyclical encoding saja | Ditolak |
| 3 | Sample weighting lebih besar untuk baris jam commute saat training | Hasil tidak konsisten antar bobot yang dicoba, RMSE/MAE cenderung memburuk, penurunan biaya hanya sekitar 2 persen dan tidak stabil | Ditolak |
| 4 | Regresi kuantil global pada kuantil optimal dari rasio biaya (newsvendor, sekitar 0,86) | Biaya turun besar secara agregat (sekitar 35 persen), tapi model membesar-besarkan prediksi pada jam sepi secara ekstrem karena margin diterapkan seragam ke semua jam | Ditolak (menang secara matematis, tidak masuk akal secara operasional) |
| 5 | Transformasi target `log1p(cnt)` | Terlihat menang pada satu split, namun saat diuji ulang lewat walk-forward (4 fold), hasilnya seri untuk RMSE dan kalah untuk biaya dolar pada 3 dari 4 fold | Ditolak (temuan awal kebetulan satu split, tidak tahan uji ulang) |
| 6 | Fitur interaksi cuaca (`temp x hum`, flag cuaca buruk, `temp^2`) | Perbaikan RMSE walk-forward sangat kecil (rata-rata di bawah 0,5 poin), dalam batas noise | Ditolak (model berbasis pohon sudah menangkap interaksi ini sendiri) |

## 4. Hasil dan Evaluasi

### 4.1 Metrik Performa (Test Set, Split Waktu)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline naif (rata-rata per jam) | 112,35 | 165,84 | - |
| Model titik-prediksi (XGBoost tuned, tanpa margin) | 43,50 | 70,29 | 0,898 |
| Model final (dengan margin commute) | 51,17 | 84,88 | 0,851 |

Metrik lengkap model final: MAPE 52,55 persen, RMSPE 146,89 persen, RMSLE 0,579. MAPE dan RMSPE tetap tinggi karena distribusi target yang right-skewed (lihat 3.1), mengonfirmasi keputusan untuk tidak memakai keduanya sebagai metrik utama.

Perlu dicatat bahwa MAE dan RMSE model final sedikit lebih buruk dibanding model titik-prediksi sebelum margin diterapkan. Ini adalah trade-off yang disengaja: margin dipilih untuk meminimalkan biaya dolar (lihat 4.2), bukan RMSE, dan sudah dipertimbangkan sejak tahap pemilihan margin di validation set.

**Kapan model ini bisa dipercaya.** Prediksi paling akurat pada jam sepi (dini hari) di luar hari libur. Prediksi paling perlu kehati-hatian pada jam commute pagi/sore dan hari libur, di mana error absolut tetap lebih tinggi meskipun margin sudah mengurangi dampak biayanya.

### 4.2 Dampak Bisnis (Cost-Benefit Analysis)

Simulasi biaya operasional dilakukan dengan asumsi ilustratif: kehilangan satu ride (understock) senilai kurang lebih 3,00 dolar, satu sepeda menganggur (overstock) senilai kurang lebih 0,50 dolar per jam, rasio 6 banding 1 yang meniru logika RMSE di bagian evaluasi (understock jauh lebih mahal daripada overstock). Empat skenario dibandingkan pada test set yang sama.

| Skenario | Biaya Total (Test Set) | Perubahan vs Tanpa Model |
|---|---|---|
| Tanpa model (rata-rata statis) | $969.743 | - |
| Baseline naif (rata-rata per jam) | $707.005 | -27,1% |
| Model titik-prediksi (tanpa margin) | $191.810 | -80,2% |
| Model final (dengan margin commute) | $169.702 | -82,5% |

Margin commute memberi penurunan biaya tambahan sebesar 11,5 persen di atas model titik-prediksi yang RMSE-nya sudah baik, bukti bahwa menyelaraskan koreksi model dengan struktur biaya bisnis, bukan hanya mengejar RMSE, memberi nilai tambah yang terukur. Angka dolar di atas bergantung penuh pada dua asumsi biaya yang belum divalidasi ke data finansial riil perusahaan. Yang independen dari asumsi tersebut adalah arah dan besar relatif perbaikannya, dan itu yang direkomendasikan untuk dikomunikasikan ke stakeholder, sambil merekomendasikan tim finance/operasional memasukkan angka biaya riil mereka sendiri.

### 4.3 Explainability

Feature importance dan SHAP value, dihitung dari model titik-prediksi, menunjukkan `is_commute` dan representasi jam (`hr_sin`/`hr_cos`) sebagai fitur paling dominan, konsisten dengan pola jam sibuk yang teridentifikasi sejak tahap eksplorasi data dan analisis residual. Fitur cuaca tetap berkontribusi namun jauh di bawah faktor waktu dan commute.

## 5. Keterbatasan

- Split berbasis waktu mengungkap tantangan yang lebih realistis dibanding split acak: dataset memiliki tren pertumbuhan tahunan yang kuat (sekitar 61 persen dari 2011 ke 2012), dan skor test yang dilaporkan mencerminkan tantangan itu secara jujur.
- Dataset ini adalah subset dari UCI Bike Sharing Dataset lengkap, tidak memiliki `windspeed`, `workingday`, `weekday`, dan `yr`. Fitur `days_since_start` adalah pengganti internal untuk `yr` yang terbukti sangat membantu, namun model berbasis pohon punya keterbatasan mengekstrapolasi tren linear murni jauh ke luar rentang yang pernah dilihat.
- Margin commute bukan konstanta yang stabil sempurna. Nilai optimalnya bervariasi antar fold walk-forward (kurang lebih 1,1 sampai 1,4 kali). Direkomendasikan untuk dikalibrasi ulang secara berkala memakai data terbaru, bukan dipakai sebagai angka tetap selamanya.
- Enam ide perbaikan dicoba dan ditolak (lihat 3.6), dicatat secara eksplisit agar proses eksplorasi terlihat menyeluruh, bukan berhenti di percobaan pertama yang terlihat menjanjikan.
- Asumsi biaya pada analisis dampak bisnis bersifat ilustratif dan belum divalidasi ke data finansial riil perusahaan.

## 6. Rekomendasi Pengembangan Selanjutnya

1. Validasi asumsi biaya bersama tim finance/operasional menggunakan data riil.
2. Kalibrasi ulang margin commute secara berkala (misalnya tiap kuartal) memakai data terbaru, mengikuti prosedur pemilihan margin di validation set yang sama, bukan angka tetap.
3. Lengkapi fitur dengan `windspeed`, `workingday`, `weekday`, dan `yr` dari dataset UCI versi penuh jika tersedia, kemungkinan bisa menggantikan `days_since_start` dengan `yr` asli yang lebih baik untuk ekstrapolasi jangka panjang.
4. Uji margin per jam yang lebih granular (bukan satu pengali untuk seluruh jam commute), sebagai pengembangan lanjutan dari pendekatan yang sudah terbukti.
5. Opsional: deploy sebagai tool self-serve menggunakan Streamlit.

## 7. Struktur Direktori

```
.
├── data/
│   └── data_bike_sharing.csv
├── models/
│   ├── final_model_xgboost.pkl
│   └── commute_margin_meta.json
├── notebooks/
│   └── capstone_v2.ipynb
└── README.md
```

## 8. Cara Menjalankan / Reproduksi

**Kebutuhan.** Python 3.10 atau lebih baru, dan pustaka berikut:

```
pip install pandas numpy scipy scikit-learn xgboost shap statsmodels seaborn matplotlib jupyter
```

**Langkah:**

1. Clone repository ini dan pastikan struktur direktori sesuai bagian 7 (folder `data/` dan `models/` sejajar dengan folder `notebooks/`).
2. (Opsional tapi disarankan) Buat virtual environment terlebih dahulu:
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install pustaka yang dibutuhkan dengan perintah pip di atas.
4. Jalankan `jupyter notebook` atau `jupyter lab` dari root repository, lalu buka `notebooks/capstone_v2.ipynb`.
5. Jalankan seluruh cell dari atas ke bawah (menu Kernel > Restart & Run All), karena beberapa cell di bagian akhir bergantung pada variabel yang didefinisikan di cell sebelumnya.

## 9. Cara Menggunakan Model Tersimpan

Berkas `final_model_xgboost.pkl` hanya berisi pipeline preprocessing dan model XGBoost. Margin commute sengaja **tidak** dibakar ke dalam objek pickle, karena dirancang sebagai aturan post-processing yang transparan dan mudah dikalibrasi ulang. Siapa pun yang memuat model ini wajib menerapkan koreksi margin secara eksplisit setelah `.predict()`, menggunakan nilai yang tersimpan di `commute_margin_meta.json`.

```python
import pickle
import json
import numpy as np
import pandas as pd

with open('models/final_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/commute_margin_meta.json') as f:
    margin_meta = json.load(f)

def commute_mask(X):
    dow = pd.to_datetime(X['dteday']).dt.dayofweek
    return (
        (X['hr'].isin(margin_meta['commute_hours']))
        & (dow < 5)
        & (X['holiday'] == 0)
    ).astype(int).values

def predict_with_commute_margin(model, X, margin):
    raw_pred = np.clip(model.predict(X), 0, None)
    mask = commute_mask(X)
    adjusted = raw_pred.copy()
    adjusted[mask == 1] = adjusted[mask == 1] * margin
    return adjusted

# X_new harus berisi kolom mentah: dteday, hum, weathersit, holiday, season, atemp, temp, hr
y_pred = predict_with_commute_margin(model, X_new, margin_meta['commute_margin'])
```

Input (`X_new`) harus berupa DataFrame dengan delapan kolom mentah yang sama seperti data pelatihan: `dteday`, `hum`, `weathersit`, `holiday`, `season`, `atemp`, `temp`, `hr`. Seluruh transformasi (penguraian tanggal, cyclical encoding jam, fitur commute, one-hot encoding) sudah termasuk di dalam pipeline yang tersimpan.

## 10. Berkas dalam Repository

| Berkas | Deskripsi |
|---|---|
| `notebooks/capstone_v2.ipynb` | Notebook analisis lengkap, dari business understanding sampai model tersimpan |
| `data/data_bike_sharing.csv` | Dataset historis penyewaan sepeda per jam (12.165 baris, 2011-2012) |
| `models/final_model_xgboost.pkl` | Pipeline preprocessing dan model XGBoost final, hasil refit pada seluruh data |
| `models/commute_margin_meta.json` | Metadata margin commute (nilai margin, jam commute, catatan penggunaan) |
| `README.md` | Berkas ini |
