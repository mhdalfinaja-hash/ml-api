from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# ✅ Download model jika belum ada
if not os.path.exists('model_pakan_rf_full.pkl'):
    os.system('python download_model.py')

app = Flask(__name__)

# --- Load model & scaler ---
try:
    model = joblib.load('model_pakan_rf_full.pkl')
    scaler = joblib.load('scaler_pakan_rf_full.pkl')
    print("✅ Model & scaler berhasil dimuat.")
except Exception as e:
    print("❌ Gagal muat model:", e)
    raise SystemExit

# --- Helper functions (dari script Anda) ---
def kategori_suhu(s):
    if s < 24:
        return "DINGIN - Metabolisme turun; Nafsu menurun"
    elif 24 <= s <= 30:
        return "NORMAL - Kondisi stabil; Nafsu normal"
    else:
        return "PANAS - Oksigen turun; Risiko stres panas"

def tentukan_frekuensi(umur):
    if umur <= 1: return 4
    elif umur <= 3: return 3
    elif umur <= 5: return 3
    else: return 2

def buat_jadwal(frekuensi, mulai="08:00"):
    from datetime import datetime, timedelta
    h0, m0 = map(int, mulai.split(':'))
    start = datetime.now().replace(hour=h0, minute=m0, second=0, microsecond=0)
    interval = 24.0 / max(1, frekuensi)
    return [(start + timedelta(hours=interval * i)).strftime("%H:%M") for i in range(frekuensi)]

# --- Endpoint utama ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # Ambil data (sesuai kolom model Anda)
        jumlah = float(data.get('jumlah_ikan', 100))
        umur = int(float(data.get('umur_minggu', 3)))
        pakan_per_bukaan = float(data.get('pakan_per_bukaan', 5))
        protein = float(data.get('protein_pct', 35))
        lemak = float(data.get('lemak_pct', 8))
        serat = float(data.get('serat_pct', 5))
        suhu = float(data.get('suhu_c', 28))

        # Prediksi
        X = [[jumlah, umur, pakan_per_bukaan, protein, lemak, serat, suhu]]
        X_scaled = scaler.transform(X)
        total_pakan = round(float(model.predict(X_scaled)[0]), 2)

        # Hitung jadwal & frekuensi
        frekuensi = tentukan_frekuensi(umur)
        jadwal = buat_jadwal(frekuensi)
        jadwal_str = ";".join(jadwal)
        bukaan = max(1, int(round(total_pakan / frekuensi / pakan_per_bukaan))) if frekuensi > 0 else 1

        # Format respons
        return jsonify({
            'status': 'success',
            'data': {
                'rekomendasi_pakan': total_pakan,
                'frekuensi_pakan': frekuensi,
                'waktu_pakan': jadwal_str,
                'bukaan_per_jadwal': bukaan,
                'kategori_suhu': kategori_suhu(suhu)
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- Jalankan Flask ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # ✅ debug=False = sekali muat