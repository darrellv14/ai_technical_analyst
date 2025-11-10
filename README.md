# 🤖 Brutal Stock Tool v10 — ML Enhanced + TA AI

Aplikasi analisis saham Indonesia menggunakan Machine Learning (LightGBM + Optuna) dan Technical Analysis dengan AI Commentary (Google Gemini).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 🌐 Live Demo

**🚀 Try it now:** [Your App URL Here] _(akan tersedia setelah deploy)_

## ✨ Fitur

- **Machine Learning Quant**
  - Feature engineering dengan 50+ indikator teknikal
  - Hyperparameter tuning menggunakan Optuna
  - Walk-forward cross-validation dengan embargo
  - Prediksi harga T+1 dengan win-rate analysis
  - Support untuk fitur eksternal (IHSG, VIX)

- **Technical Analysis + Trade Plan**
  - Analisis indikator: RSI, MACD, EMA, ATR, Volume
  - Deteksi level Support & Resistance
  - Breakout strategy dengan entry zone, stop-loss, dan target profit
  - AI Commentary menggunakan Google Gemini API

- **Visualisasi Interaktif**
  - Chart candlestick dengan level trading
  - Perbandingan prediksi ML vs harga aktual
  - Dark theme yang eye-friendly

## 📋 Requirements

- Python 3.8+
- Internet connection (untuk download data Yahoo Finance)

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/darrellv14/ai_technical_analyst.git
cd ai_technical_analyst
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup API Key (Opsional - untuk fitur AI Commentary)

Jika Anda ingin menggunakan fitur AI Commentary dengan Google Gemini:

1. Dapatkan API key dari [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Buat file `.env` di root folder (sudah ada template)
3. Edit file `.env` dan masukkan API key Anda:

```
GEMINI_API_KEY=your_api_key_here
```

**Note:** File `.env` sudah ada di folder ini dengan API key yang Anda berikan. Pastikan tidak membagikan file ini ke publik!

## 🎯 Cara Menjalankan

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser Anda di alamat: `http://localhost:8501`

## ☁️ Deploy ke Cloud (GRATIS!)

Ingin aplikasi bisa diakses online? Deploy ke Streamlit Cloud dalam 5 menit!

**📘 Panduan Lengkap:** Lihat [DEPLOYMENT.md](DEPLOYMENT.md)

**⚡ Quick Start:** Lihat [QUICKSTART.md](QUICKSTART.md)

### Ringkasan Cepat:
1. Push code ke GitHub
2. Buka https://share.streamlit.io/
3. Connect repository Anda
4. Set API key di Secrets
5. Deploy! 🚀

## 📖 Cara Menggunakan

1. **Input Ticker Saham**
   - Masukkan kode saham BEI di sidebar (contoh: BBCA, BBRI, BMRI)
   - Pisahkan dengan spasi atau koma untuk multiple tickers
   - Centang "Auto-append .JK" untuk otomatis menambahkan suffix .JK

2. **Atur Parameter**
   - **Start Date**: Pilih tanggal mulai pengambilan data historis
   - **ML Settings**: 
     - Optuna Trials: Jumlah iterasi hyperparameter tuning (lebih banyak = lebih lama tapi lebih akurat)
     - TimeSeries Splits: Jumlah fold untuk cross-validation
     - Embargo: Gap antara training dan validation untuk menghindari look-ahead bias
   - **TA Settings**: 
     - Centang "Komentar AI Gemini" untuk mendapatkan analisis dari AI

3. **Jalankan Analisis**
   - Klik tombol **🚀 RUN**
   - Tunggu proses download data dan training model
   - Hasil akan ditampilkan untuk setiap ticker

4. **Interpretasi Hasil**

   **ML Quant:**
   - Chart prediksi vs aktual dengan win-rate
   - Tabel prediksi T+1 dengan arah (NAIK/TURUN)
   - Metrics: Win-rate, MAE, R²

   **TA + Trade Plan:**
   - Chart candlestick dengan level support/resistance
   - Entry zone, stop-loss, dan target profit (TP1, TP2)
   - View market: BULLISH atau BEARISH
   - AI Commentary dengan analisis lengkap

## ⚙️ Konfigurasi Advanced

### Menonaktifkan Fitur

Di sidebar, Anda bisa:
- Uncheck "Jalankan ML Quant" untuk skip machine learning
- Uncheck "Jalankan TA + Trade Plan" untuk skip technical analysis
- Uncheck fitur IHSG/VIX jika tidak diperlukan (untuk mempercepat)

### Data Eksternal

- **IHSG (^JKSE)**: Menambahkan fitur relative strength terhadap index
- **VIX (^VIX)**: Menambahkan fitur volatility market global

## 🔧 Troubleshooting

### Error saat install pandas-ta
```bash
pip install --upgrade pip
pip install pandas-ta --no-cache-dir
```

### Error "No data found"
- Pastikan ticker yang diinput benar
- Coba tambahkan .JK secara manual (contoh: BBCA.JK)
- Periksa koneksi internet

### Error Gemini API
- Pastikan API key valid dan aktif
- Check kuota API di [Google AI Studio](https://makersuite.google.com/)
- Pastikan internet connection stabil

### Aplikasi lambat
- Kurangi Optuna Trials (default: 30)
- Kurangi jumlah ticker yang diproses sekaligus
- Uncheck fitur IHSG/VIX jika tidak diperlukan

## 📊 Struktur Project

```
ai_technical_analyst/
├── app.py              # Main aplikasi Streamlit
├── requirements.txt    # Python dependencies
├── .env               # API keys (jangan commit ke git!)
├── .gitignore         # Files yang diabaikan git
└── README.md          # Dokumentasi ini
```

## ⚠️ Disclaimer

**PENTING:** 

Aplikasi ini adalah tool analisis dan **BUKAN rekomendasi investasi**. Hasil prediksi machine learning dan analisis teknikal adalah untuk tujuan edukasi dan riset saja.

- Past performance tidak menjamin hasil di masa depan
- Selalu lakukan riset sendiri (DYOR - Do Your Own Research)
- Investasi saham mengandung risiko
- Konsultasikan dengan financial advisor sebelum membuat keputusan investasi

Pembuat aplikasi tidak bertanggung jawab atas kerugian yang timbul dari penggunaan tool ini.

## 📝 License

MIT License - Silakan gunakan dan modifikasi sesuai kebutuhan

## 🤝 Contributing

Pull requests are welcome! Untuk perubahan besar, silakan buka issue terlebih dahulu.

## 📧 Contact

- GitHub: [@darrellv14](https://github.com/darrellv14)
- Repository: [ai_technical_analyst](https://github.com/darrellv14/ai_technical_analyst)

## 🙏 Credits

- [Streamlit](https://streamlit.io/) - Web framework
- [yfinance](https://github.com/ranaroussi/yfinance) - Data provider
- [LightGBM](https://lightgbm.readthedocs.io/) - ML framework
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [pandas-ta](https://github.com/twopirllc/pandas-ta) - Technical indicators
- [Google Gemini](https://deepmind.google/technologies/gemini/) - AI commentary

---

**Happy Trading! 📈🚀**

*Remember: The best investment is in yourself and your knowledge!*
