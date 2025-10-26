# models/train.py
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from preprocessing.preprocess import (
    load_data, basic_clean, add_time_features, build_sessions,
    dwell_time_features, session_aggregates, ip_features,
    encode_categoricals, build_temporal_rolling, select_features,
    build_sequences
)

from models import ml_models, dl_models, stacking
from utils.evaluation import evaluate_classification

MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)

def main(data_path="data/ad_clicks.csv"):
    print("Loading data...")
    df = load_data(data_path)
    df = basic_clean(df)
    df = add_time_features(df)
    df = build_sessions(df)
    df = dwell_time_features(df)
    df = session_aggregates(df)
    df = ip_features(df)
    df, encoders = encode_categoricals(df)

    # Tabular features and labels
    X_tab_df, y = select_features(df)
    X_tab = X_tab_df.fillna(0).values
    # scale
    scaler = StandardScaler()
    X_tab_scaled = scaler.fit_transform(X_tab)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # balance
    print("Applying SMOTE to handle imbalance...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_tab_scaled, y)
    print("Original class counts:", np.bincount(y), "Resampled:", np.bincount(y_res))

    # Train ML models on resampled data
    print("Training ML models...")
    lr = ml_models.train_logistic(X_res, y_res)
    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic.pkl"))
    rf = ml_models.train_random_forest(X_res, y_res)
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    xgb = ml_models.train_xgb(X_res, y_res)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgboost.pkl"))
    lgb = ml_models.train_lgbm(X_res, y_res)
    joblib.dump(lgb, os.path.join(MODEL_DIR, "lightgbm.pkl"))

    # Evaluate ML on holdout
    X_train, X_test, y_train, y_test = train_test_split(X_tab_scaled, y, test_size=0.2, stratify=y, random_state=42)
    evaluate_classification(lr, X_test, y_test, "Logistic Regression")
    evaluate_classification(rf, X_test, y_test, "Random Forest")
    evaluate_classification(xgb, X_test, y_test, "XGBoost")
    evaluate_classification(lgb, X_test, y_test, "LightGBM")

    # Sequences for DL models (session-level)
    print("Building sequences for DL models...")
    seq_len = 20
    feature_cols = ['app_id_enc','device_enc','hour','dwell_time']
    X_seq, y_seq = build_sequences(df, seq_len=seq_len, feature_cols=feature_cols)
    if X_seq.shape[0] == 0:
        print("No session sequences found. Skipping DL training.")
    else:
        n, s, f = X_seq.shape
        X_flat = X_seq.reshape(n, s*f)
        from sklearn.preprocessing import StandardScaler
        scaler_seq = StandardScaler()
        X_flat_scaled = scaler_seq.fit_transform(X_flat)
        X_seq_scaled = X_flat_scaled.reshape(n, s, f)
        joblib.dump(scaler_seq, os.path.join(MODEL_DIR, "scaler_seq.pkl"))

        Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_seq_scaled, y_seq, test_size=0.2, stratify=y_seq, random_state=42)

        print("Training GRU...")
        gru = dl_models.build_gru(seq_len, f)
        gru, history_gru = dl_models.train_tf_model(gru, Xs_train, ys_train, epochs=8, batch_size=128, model_path=os.path.join(MODEL_DIR, "gru_model"))
        _, auc = gru.evaluate(Xs_test, ys_test, verbose=0)
        print("GRU test AUC:", auc)

        print("Training LSTM...")
        lstm = dl_models.build_lstm(seq_len, f)
        lstm, history_lstm = dl_models.train_tf_model(lstm, Xs_train, ys_train, epochs=8, batch_size=128, model_path=os.path.join(MODEL_DIR, "lstm_model"))
        _, auc = lstm.evaluate(Xs_test, ys_test, verbose=0)
        print("LSTM test AUC:", auc)

        print("Training CNN-LSTM...")
        cnnlstm = dl_models.build_cnn_lstm(seq_len, f)
        cnnlstm, history_cnnlstm = dl_models.train_tf_model(cnnlstm, Xs_train, ys_train, epochs=8, batch_size=128, model_path=os.path.join(MODEL_DIR, "cnnlstm_model"))
        _, auc = cnnlstm.evaluate(Xs_test, ys_test, verbose=0)
        print("CNN-LSTM test AUC:", auc)

    # Stacking ensemble for ML models (use tabular X_res & y_res)
    print("Building stacking ensemble...")
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    base_clfs = [
        ("rf", RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)),
        ("lgb", LGBMClassifier(n_estimators=200)),
        ("xgb", XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss'))
    ]
    trained_bases, meta = stacking.build_stacking_meta(X_res, y_res, base_clfs)
    stacking.save_trained_bases(trained_bases, meta, path_prefix=os.path.join(MODEL_DIR, "stack_"))

    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    print("Training complete. Models saved in", MODEL_DIR)

if __name__ == "__main__":
    main()
