# ⚙️ Streamlit Secrets Configuration Template

## Untuk Local Development

Buat file `.streamlit/secrets.toml` dan isi dengan:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

## Untuk Streamlit Cloud Deployment

1. Buka dashboard app Anda di Streamlit Cloud
2. Klik "Settings" → "Secrets"
3. Paste konfigurasi berikut:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

4. Klik "Save"

## Cara Mendapatkan Gemini API Key

1. Kunjungi: https://makersuite.google.com/app/apikey
2. Login dengan akun Google
3. Klik "Create API Key"
4. Copy API key yang dihasilkan
5. Paste ke secrets configuration

## ⚠️ PENTING!

- **JANGAN** commit file `secrets.toml` ke GitHub
- **JANGAN** share API key di public
- File `.gitignore` sudah dikonfigurasi untuk ignore `secrets.toml`
- Ganti API key segera jika tidak sengaja ter-expose

## 📝 Notes

- File `secrets.toml` hanya untuk local development
- Di Streamlit Cloud, gunakan UI Secrets Manager
- API key akan otomatis ter-load oleh aplikasi
