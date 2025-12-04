from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from datetime import datetime, timedelta

# Import CORS
from flask_cors import CORS

app = Flask(__name__)

# Aktifkan CORS untuk /predict
CORS(app, resources={r"/predict": {"origins": "https://kel5.myiot.fun"}})

# ======================================================
#  LOAD MODEL
# ======================================================
MODEL_FILE = "model_pakan_rf_full.pkl"
SCALER_FILE = "scaler_pakan_rf_full.pkl"

model = None
scaler = None

def load_model():
    """Load model dan scaler"""
    global model, scaler
    try:
        if not os.path.exists(MODEL_FILE):
            print(f"❌ Model tidak ditemukan: {MODEL_FILE}")
            return False
        
        if not os.path.exists(SCALER_FILE):
            print(f"❌ Scaler tidak ditemukan: {SCALER_FILE}")
            return False
        
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        print("✅ Model & scaler berhasil dimuat.")
        return True
    except Exception as e:
        print(f"❌ Gagal memuat model/scaler: {e}")
        return False

# Load model saat startup
if not load_model():
    print("⚠️  Peringatan: Model tidak dimuat, API akan berjalan tanpa model!")

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
    try:
        h0, m0 = map(int, mulai.split(":"))
        start = datetime.now().replace(hour=h0, minute=m0, second=0, microsecond=0)
        interval = 24.0 / max(1, frekuensi)
        return [(start + timedelta(hours=interval * i)).strftime("%H:%M") for i in range(frekuensi)]
    except:
        default_times = ["08:00", "12:00", "16:00", "20:00"]
        return default_times[:max(1, min(frekuensi, 4))]


# ======================================================
#  API ROUTES
# ======================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok", 
        "message": "ML API aktif",
        "model_loaded": model is not None,
        "python_version": os.environ.get("PYTHON_VERSION", "unknown")
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Cek model sudah dimuat
        if model is None or scaler is None:
            return jsonify({
                "status": "error", 
                "message": "Model belum dimuat. Silakan periksa server logs."
            }), 503
        
        # Ambil data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # Validasi dan parse input
        required_fields = ['jumlah_ikan', 'umur_minggu', 'pakan_per_bukaan', 
                          'protein_pct', 'lemak_pct', 'serat_pct', 'suhu_c']
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400
        
        try:
            jumlah = float(data['jumlah_ikan'])
            umur = int(float(data['umur_minggu']))
            pakan_per_bukaan = float(data['pakan_per_bukaan'])
            protein = float(data['protein_pct'])
            lemak = float(data['lemak_pct'])
            serat = float(data['serat_pct'])
            suhu = float(data['suhu_c'])
        except ValueError as e:
            return jsonify({'error': f'Invalid numeric value: {str(e)}'}), 400

        # Validasi range input
        if jumlah <= 0:
            return jsonify({'error': 'jumlah_ikan harus > 0'}), 400
        if umur <= 0:
            return jsonify({'error': 'umur_minggu harus > 0'}), 400
        if pakan_per_bukaan <= 0:
            return jsonify({'error': 'pakan_per_bukaan harus > 0'}), 400

        # Prediksi model
        X = [[jumlah, umur, pakan_per_bukaan, protein, lemak, serat, suhu]]
        X_scaled = scaler.transform(X)
        total_pakan = round(float(model.predict(X_scaled)[0]), 2)

        # Hitung hasil tambahan
        frekuensi = tentukan_frekuensi(umur)
        jadwal = buat_jadwal(frekuensi)
        jadwal_str = ";".join(jadwal)
        
        if frekuensi > 0 and pakan_per_bukaan > 0:
            bukaan = max(1, int(round(total_pakan / frekuensi / pakan_per_bukaan)))
        else:
            bukaan = 1

        return jsonify({
            "status": "success",
            "data": {
                "rekomendasi_pakan": total_pakan,
                "frekuensi_pakan": frekuensi,
                "waktu_pakan": jadwal_str,
                "bukaan_per_jadwal": bukaan,
                "kategori_suhu": kategori_suhu(suhu),
                "input_values": {
                    "jumlah_ikan": jumlah,
                    "umur_minggu": umur,
                    "pakan_per_bukaan": pakan_per_bukaan,
                    "protein_pct": protein,
                    "lemak_pct": lemak,
                    "serat_pct": serat,
                    "suhu_c": suhu
                }
            }
        })

    except Exception as e:
        print(f"❌ Error in predict: {str(e)}")  # Log error
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500


# ======================================================
#  RUN untuk development lokal saja
# ======================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
