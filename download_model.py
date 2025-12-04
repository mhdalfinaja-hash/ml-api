import gdown
import os

# 🔗 Ganti URL ini dengan link "direct download" dari Google Drive Anda
# Cara dapatkan: ganti "/view?usp=sharing" → jadi "/uc?id=FILE_ID"
MODEL_URL = "https://drive.google.com/uc?id=10DKeKxyBA-CT3jdrvOHb_bh7x7Mr9HJ5"
MODEL_PATH = "model_pakan_rf_full.pkl"
SCALER_PATH = "scaler_pakan_rf_full.pkl"

if not os.path.exists(MODEL_PATH):
    print("📥 Mengunduh model dari Google Drive...")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model berhasil diunduh:", MODEL_PATH)
    except Exception as e:
        print("❌ Gagal mengunduh model:", e)
        exit(1)

# Scaler kecil — Anda bisa upload ke Drive juga, atau simpan langsung di repo
# Jika scaler sudah diupload ke Drive, tambahkan logika download serupa
if not os.path.exists(SCALER_PATH):
    print("⚠️ Scaler tidak ditemukan — pastikan 'scaler_pakan_rf_full.pkl' ada di repo.")