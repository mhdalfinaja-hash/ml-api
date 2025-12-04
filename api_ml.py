from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# ======================================================
#  LOAD MODEL
# ======================================================
MODEL_FILE = "model_pakan.pkl"
SCALER_FILE = "scaler_pakan.pkl"

if not os.path.exists(MODEL_FILE):
    raise Exception(f"❌ Model tidak ditemukan: {MODEL_FILE}")

if not os.path.exists(SCALER_FILE):
    raise Exception(f"❌ Scaler tidak ditemukan: {SCALER_FILE}")

try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("✅ Model & scaler berhasil dimuat.")
except Exception as e:
    print("❌ Gagal memuat model:", e)
    raise SystemExit


# ======================================================
#  HELPER FUNCTIONS
# ======================================================
def kategori_suhu(s):
    if s < 24:
        return "DINGIN - Metabolisme turun; Nafsu menurun"
    elif 24 <= s <= 30:
        return "NORMAL - Kondisi stabil; Nafsu normal"
    else:
        return "PANAS - Oksigen turun; Risiko stres panas"


def tentukan_frekuensi(umur):
    if umur <= 1:
        return 4
    elif umur <= 3:
        return 3
    elif umur <= 5:
        return 3
    else:
        return 2


def buat_jadwal(frekuensi, mulai="08:00"):
    h0, m0 = map(int, mulai.split(":"))
    start = datetime.now().replace(hour=h0, minute=m0, second=0, microsecond=0)
    interval = 24.0 / max(1, frekuensi)
    return [(start + timedelta(hours=interval * i)).strftime("%H:%M") for i in range(frekuensi)]


# ======================================================
#  API ROUTES
# ======================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "ML API aktif"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # Ambil data input
        jumlah = float(data.get('jumlah_ikan', 100))
        umur = int(float(data.get('umur_minggu', 3)))
        pakan_per_bukaan = float(data.get('pakan_per_bukaan', 5))
        protein = float(data.get('protein_pct', 35))
        lemak = float(data.get('lemak_pct', 8))
        serat = float(data.get('serat_pct', 5))
        suhu = float(data.get('suhu_c', 28))

        # Prediksi model
        X = [[jumlah, umur, pakan_per_bukaan, protein, lemak, serat, suhu]]
        X_scaled = scaler.transform(X)
        total_pakan = round(float(model.predict(X_scaled)[0]), 2)

        # Hitung hasil tambahan
        frekuensi = tentukan_frekuensi(umur)
        jadwal = buat_jadwal(frekuensi)
        jadwal_str = ";".join(jadwal)
        bukaan = max(1, int(round(total_pakan / frekuensi / pakan_per_bukaan))) if frekuensi > 0 else 1

        return jsonify({
            "status": "success",
            "data": {
                "rekomendasi_pakan": total_pakan,
                "frekuensi_pakan": frekuensi,
                "waktu_pakan": jadwal_str,
                "bukaan_per_jadwal": bukaan,
                "kategori_suhu": kategori_suhu(suhu)
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ======================================================
#  RUN
# ======================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # render memakai PORT ENV
    app.run(host='0.0.0.0', port=port, debug=False)
