# preprocessing/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta

def load_data(path="data/ad_clicks.csv", nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    return df

def basic_clean(df):
    required = ['ip', 'app_id', 'device', 'timestamp', 'is_downloaded']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['ip', 'device', 'timestamp']).reset_index(drop=True)
    return df

def add_time_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df

def build_sessions(df, session_gap_minutes=30):
    df = df.copy()
    gap = pd.Timedelta(minutes=session_gap_minutes)
    df['prev_ts'] = df.groupby(['ip', 'device'])['timestamp'].shift(1)
    df['new_session'] = (df['timestamp'] - df['prev_ts']) > gap
    df['new_session'] = df['new_session'].fillna(True)
    df['session_id'] = df.groupby(['ip', 'device'])['new_session'].cumsum()
    df['session_key'] = df['ip'].astype(str) + "_" + df['device'].astype(str) + "_" + df['session_id'].astype(str)
    df.drop(['prev_ts', 'new_session'], axis=1, inplace=True)
    return df

def dwell_time_features(df):
    df = df.copy()
    df['prev_ts_session'] = df.groupby(['session_key'])['timestamp'].shift(1)
    df['dwell_time'] = (df['timestamp'] - df['prev_ts_session']).dt.total_seconds()
    df['dwell_time'] = df['dwell_time'].fillna(0)
    df.drop(['prev_ts_session'], axis=1, inplace=True)
    return df

def session_aggregates(df):
    agg = df.groupby('session_key').agg(
        session_clicks=('timestamp', 'count'),
        session_duration=('timestamp', lambda s: (s.max() - s.min()).total_seconds())
    ).reset_index()
    df = df.merge(agg, on='session_key', how='left')
    return df

def ip_features(df):
    ip_counts = df.groupby('ip')['timestamp'].count().rename('ip_click_count').reset_index()
    df = df.merge(ip_counts, on='ip', how='left')
    return df

def encode_categoricals(df, encoders=None):
    df = df.copy()
    cats = ['app_id', 'device']
    if encoders is None:
        encoders = {}
        for c in cats:
            le = LabelEncoder()
            df[c] = df[c].astype(str)
            df[c+'_enc'] = le.fit_transform(df[c])
            encoders[c] = le
    else:
        for c in cats:
            le = encoders[c]
            df[c] = df[c].astype(str)
            df[c+'_enc'] = le.transform(df[c])
    return df, encoders

def build_temporal_rolling(df):
    df['clicks_per_session'] = df['session_clicks']
    return df

def select_features(df):
    features = [
        'ip_click_count',
        'app_id_enc',
        'device_enc',
        'hour',
        'dayofweek',
        'dwell_time',
        'session_clicks',
        'session_duration',
        'clicks_per_session'
    ]
    X = df[features].fillna(0)
    y = df['is_downloaded'].astype(int)
    return X, y

def build_sequences(df, seq_len=20, feature_cols=None):
    if feature_cols is None:
        feature_cols = ['app_id_enc','device_enc','hour','dwell_time']
    sessions = []
    labels = []
    for key, g in df.groupby('session_key'):
        X = g[feature_cols].values
        y = 1 if g['is_downloaded'].any() else 0
        if len(X) >= seq_len:
            Xs = X[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(X), X.shape[1]))
            Xs = np.vstack([pad, X])
        sessions.append(Xs)
        labels.append(y)
    X_arr = np.stack(sessions) if sessions else np.zeros((0, seq_len, len(feature_cols)))
    y_arr = np.array(labels)
    return X_arr, y_arr

if __name__ == "__main__":
    df = load_data(nrows=2000)
    df = basic_clean(df)
    df = add_time_features(df)
    df = build_sessions(df)
    df = dwell_time_features(df)
    df = session_aggregates(df)
    df = ip_features(df)
    df, enc = encode_categoricals(df)
    X, y = select_features(df)
    print("Feature shape:", X.shape)
    Xs, ys = build_sequences(df)
    print("Seq shape:", Xs.shape)

 