from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allows frontend & Postman to talk to the API

# ── Load model & scaler once at startup ──────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, '..', 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = ['Close', 'Volume', 'Daily_Return', 'MA_7', 
            'MA_21', 'Price_Range', 'Volume_Change']

# ── Routes ────────────────────────────────────────────────────

# 1. Home route — serves frontend
@app.route('/')
def home():
    return render_template('index.html')


# 2. Health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status' : 'API is running ✅',
        'model'  : type(model).__name__,
        'features': FEATURES
    })


# 3. Predict route — main endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ── Validate input ──
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({
                'error': f'Missing fields: {missing}'
            }), 400

        # ── Extract & scale features ──
        input_values = np.array([[data[f] for f in FEATURES]])
        input_scaled = scaler.transform(input_values)

        # ── Predict ──
        prediction  = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        return jsonify({
            'prediction'        : int(prediction),
            'signal'            : 'UP 📈' if prediction == 1 else 'DOWN 📉',
            'confidence_up'     : round(float(probability[1]) * 100, 2),
            'confidence_down'   : round(float(probability[0]) * 100, 2),
            'model_used'        : type(model).__name__,
            'input_received'    : data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 4. Compare all models route
@app.route('/compare', methods=['GET'])
def compare():
    import joblib
    results = {
        'Logistic Regression' : {},
        'Random Forest'       : {},
        'XGBoost'             : {}
    }

    model_files = {
        'Logistic Regression' : '../models/logistic_regression.pkl',
        'Random Forest'       : '../models/random_forest.pkl',
        'XGBoost'             : '../models/xgboost.pkl'
    }

    return jsonify({
        'message'      : 'Models available for comparison',
        'models'       : list(model_files.keys()),
        'best_model'   : type(model).__name__,
        'note'         : 'See notebook 04 for full comparison metrics'
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)