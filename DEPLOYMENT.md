# 🚀 Panduan Deploy ke Streamlit Cloud

Ikuti langkah-langkah berikut untuk deploy aplikasi Anda ke Streamlit Cloud (GRATIS!).

## 📋 Persiapan

### 1. Push ke GitHub

Pastikan code Anda sudah di push ke GitHub repository: `https://github.com/darrellv14/ai_technical_analyst`

```bash
# Di folder project Anda
git add .
git commit -m "Initial commit - Brutal Stock Tool v10"
git push origin main
```

**PENTING:** File `.env` tidak akan ke-push karena sudah ada di `.gitignore`. Ini bagus untuk keamanan!

---

## 🌐 Deploy ke Streamlit Cloud

### 2. Buka Streamlit Cloud

1. Buka browser dan kunjungi: **https://share.streamlit.io/**
2. Klik **"Sign in"** atau **"Sign up"** dengan akun GitHub Anda
3. Authorize Streamlit untuk akses repository GitHub Anda

### 3. Deploy Aplikasi Baru

1. Klik tombol **"New app"** atau **"Create app"**
2. Isi form deployment:
   
   **Repository:** `darrellv14/ai_technical_analyst`
   
   **Branch:** `main` (atau branch yang Anda gunakan)
   
   **Main file path:** `app.py`
   
   **App URL (optional):** `brutal-stock-tool-v10` (atau nama custom Anda)

3. Klik **"Advanced settings"** untuk set Python version (opsional):
   - Python version: `3.12` (sesuaikan dengan versi Anda)

4. **JANGAN klik Deploy dulu!** Kita perlu set API key dulu.

### 4. Setup API Key (Secrets)

1. Di form deployment, scroll ke bagian **"Secrets"** atau **"Advanced settings"**
2. Di bagian **"Secrets"**, masukkan:

```toml
GEMINI_API_KEY = "AIzaSyBlVav8YOmFiUw-amkQYgyIWQSTerJ44KY"
```

3. **PENTING:** Ganti dengan API key asli Anda!

### 5. Deploy!

1. Klik tombol **"Deploy!"**
2. Tunggu proses deployment (biasanya 2-5 menit)
3. Streamlit akan otomatis install semua dependencies dari `requirements.txt`

---

## ✅ Selesai!

Aplikasi Anda akan tersedia di URL seperti:
```
https://brutal-stock-tool-v10.streamlit.app
```

atau

```
https://share.streamlit.io/darrellv14/ai_technical_analyst/main/app.py
```

---

## 🔧 Troubleshooting

### Error saat deployment?

#### 1. **ModuleNotFoundError**
- Pastikan semua library ada di `requirements.txt`
- Cek nama package yang benar (contoh: `scikit-learn` bukan `sklearn`)

#### 2. **API Key Error**
- Pastikan sudah set `GEMINI_API_KEY` di Secrets
- Cek format toml: `GEMINI_API_KEY = "key-anda"`
- Pastikan tidak ada typo

#### 3. **Memory Error / Resource Limits**
- Free tier Streamlit Cloud punya limit: 1 GB RAM
- Kurangi `n_trials` default di slider
- Batasi jumlah ticker yang diproses sekaligus

#### 4. **Build Failed**
- Cek build logs di dashboard Streamlit Cloud
- Mungkin ada dependency conflict
- Coba hapus versi spesifik di requirements.txt

### Cara Update Aplikasi

Setiap kali Anda push changes ke GitHub, Streamlit Cloud akan otomatis rebuild & redeploy!

```bash
git add .
git commit -m "Update: fix bug xyz"
git push origin main
```

Tunggu beberapa menit, aplikasi akan update otomatis! 🎉

---

## 🔐 Keamanan API Key

### ⚠️ JANGAN PERNAH:
- ❌ Commit file `.env` ke GitHub
- ❌ Hardcode API key di code
- ❌ Share API key di public

### ✅ SELALU:
- ✅ Gunakan Streamlit Secrets untuk API key
- ✅ Pastikan `.env` ada di `.gitignore`
- ✅ Ganti API key jika tidak sengaja ter-expose

---

## 📊 Monitoring

Setelah deploy, Anda bisa:
- **Lihat logs:** Di dashboard Streamlit Cloud
- **Monitor usage:** Cek analytics di dashboard
- **Manage app:** Start/Stop/Delete app kapan saja

---

## 💡 Tips Pro

### 1. Custom Domain (Gratis!)
Streamlit Cloud memberikan subdomain gratis:
- Format: `https://your-app-name.streamlit.app`
- Bisa custom di settings deployment

### 2. Password Protection (Paid)
Untuk protect app dengan password, upgrade ke Streamlit Cloud Team/Enterprise

### 3. Optimasi Performance
- Cache data dengan `@st.cache_data`
- Minimize data download
- Gunakan session state untuk persist data

### 4. Community Cloud Limits (Free Tier)
- **Resources:** 1 CPU core, 1 GB RAM
- **Apps:** Unlimited public apps
- **Runtime:** Apps sleep after 7 days inactivity
- **Build time:** ~5 minutes max

---

## 📞 Need Help?

- **Streamlit Docs:** https://docs.streamlit.io/
- **Community Forum:** https://discuss.streamlit.io/
- **Status Page:** https://streamlitstatus.com/

---

## 🎉 Bonus: Share Aplikasi Anda!

Setelah deploy, Anda bisa:
1. Share URL ke teman/client
2. Embed di website dengan iframe
3. Post di LinkedIn/Twitter untuk portfolio!

**Contoh Post LinkedIn:**
```
🚀 Excited to share my latest project: Brutal Stock Tool v10!

A powerful stock analysis tool combining:
📊 Machine Learning (LightGBM + Optuna)
📈 Technical Analysis 
🤖 AI Commentary (Google Gemini)

Try it here: [your-app-url]

Built with: #Python #Streamlit #MachineLearning #StockMarket

[screenshot of your app]
```

---

**Good luck with your deployment! 🚀📈**
