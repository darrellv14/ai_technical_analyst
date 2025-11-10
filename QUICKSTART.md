# Quick Start - Deploy ke Streamlit Cloud 🚀

## Step 1: Push ke GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

## Step 2: Deploy
1. Buka https://share.streamlit.io/
2. Login dengan GitHub
3. Klik "New app"
4. Pilih repository: `darrellv14/ai_technical_analyst`
5. Main file: `app.py`

## Step 3: Set Secrets
Di bagian "Advanced settings" → "Secrets", tambahkan:
```toml
GEMINI_API_KEY = "your-api-key-here"
```

## Step 4: Deploy!
Klik "Deploy!" dan tunggu ~3 menit

---

**📖 Panduan lengkap:** Lihat [DEPLOYMENT.md](DEPLOYMENT.md)
