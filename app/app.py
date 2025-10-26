# app/app.py
from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
from datetime import datetime

MODEL_DIR = "models/saved"
ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
SCALER_SEQ_PATH = os.path.join(MODEL_DIR, "scaler_seq.pkl")

# Load ML models
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Warning loading {path}: {e}")
        return None

clf_lgb = safe_load(os.path.join(MODEL_DIR, "lightgbm.pkl"))
clf_rf = safe_load(os.path.join(MODEL_DIR, "random_forest.pkl"))
clf_xgb = safe_load(os.path.join(MODEL_DIR, "xgboost.pkl"))
stack_rf = safe_load(os.path.join(MODEL_DIR, "stack_rf.pkl"))  # example if saved
stack_meta = safe_load(os.path.join(MODEL_DIR, "stack_meta.pkl"))

scaler = safe_load(SCALER_PATH)
encoders = safe_load(ENCODER_PATH)

# Load TF model
import tensorflow as tf
gru_model = None
try:
    gru_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "gru_model"))
except Exception as e:
    print("GRU model not loaded:", e)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def preprocess_input_form(form):
    ip = form.get('ip', '0.0.0.0')
    app_id = form.get('app_id', 'app_0')
    device = form.get('device', 'mobile')
    timestamp = form.get('timestamp', datetime.utcnow().isoformat())
    try:
        ts = datetime.fromisoformat(timestamp)
    except:
        try:
            ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except:
            ts = datetime.utcnow()
    hour = ts.hour
    dayofweek = ts.weekday()
    if encoders:
        try:
            app_enc = encoders['app_id'].transform([str(app_id)])[0]
            dev_enc = encoders['device'].transform([str(device)])[0]
        except Exception:
            app_enc = 0
            dev_enc = 0
    else:
        app_enc = 0
        dev_enc = 0
    ip_click_count = 1
    dwell_time = 0.0
    session_clicks = 1
    session_duration = 0.0
    clicks_per_session = 1
    feature_vector = np.array([ip_click_count, app_enc, dev_enc, hour, dayofweek, dwell_time, session_clicks, session_duration, clicks_per_session], dtype=float).reshape(1, -1)
    if scaler is not None:
        try:
            feature_vector = scaler.transform(feature_vector)
        except:
            pass
    seq_feat = np.array([app_enc, dev_enc, hour, dwell_time], dtype=float)
    return feature_vector, seq_feat

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    X_vec, seq_feat = preprocess_input_form(data)
    probs = []
    for clf in [clf_rf, clf_xgb, clf_lgb]:
        if clf is not None:
            try:
                p = clf.predict_proba(X_vec)[:,1][0]
                probs.append(p)
            except:
                pass
    ml_prob = float(np.mean(probs)) if len(probs) > 0 else 0.0

    # Build sequence
    seq_len = 20
    feat = seq_feat
    seq = np.tile(feat, (seq_len,1))[np.newaxis,...]
    try:
        scaler_seq = joblib.load(SCALER_SEQ_PATH)
        n, s, f = seq.shape
        seq_flat = seq.reshape(n, s*f)
        seq_flat = scaler_seq.transform(seq_flat)
        seq = seq_flat.reshape(n, s, f)
    except:
        pass

    dl_prob = 0.0
    if gru_model is not None:
        try:
            dl_prob = float(gru_model.predict(seq).ravel()[0])
        except Exception:
            dl_prob = 0.0

    final_prob = 0.6 * ml_prob + 0.4 * dl_prob
    label = "Genuine" if final_prob >= 0.5 else "Fraudulent"
    return jsonify({
        "ml_prob": ml_prob,
        "dl_prob": dl_prob,
        "final_prob": final_prob,
        "label": label
    })

if __name__ == "__main__":
    app.run(debug=True)
