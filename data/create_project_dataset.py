"""
Generate Ad Click Fraud Detection Dataset
-----------------------------------------
Creates a realistic CSV dataset for your project (data/ad_clicks.csv)
that mimics the structure of the Kaggle TalkingData dataset.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# ✅ Adjustable Parameters
NUM_USERS = 1500               # total unique IPs
CLICKS_PER_USER = (5, 50)      # min, max clicks per IP
FRAUD_RATIO = 0.25             # 25% fraudulent users
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 10, 1)

APP_IDS = [f"app_{i}" for i in range(1, 30)]
DEVICES = ["mobile", "tablet", "desktop"]
CHANNELS = [f"ch_{i}" for i in range(1, 10)]

def random_ip():
    """Generate a random IP address."""
    return ".".join(str(random.randint(1, 255)) for _ in range(4))

def random_timestamp():
    """Generate random timestamp within date range."""
    delta = END_DATE - START_DATE
    random_second = random.randint(0, int(delta.total_seconds()))
    return START_DATE + timedelta(seconds=random_second)

def generate_clicks_for_user(ip, is_fraud):
    """Generate clicks for a single user (IP)."""
    n_clicks = random.randint(*CLICKS_PER_USER)
    clicks = []
    base_time = random_timestamp()

    for i in range(n_clicks):
        if is_fraud:
            # Fraud bots click rapidly (few seconds apart)
            gap = random.randint(1, 15)
        else:
            # Genuine users click naturally (1–10 minutes apart)
            gap = random.randint(60, 600)
        base_time += timedelta(seconds=gap)

        record = {
            "ip": ip,
            "app_id": random.choice(APP_IDS),
            "device": random.choice(DEVICES),
            "channel": random.choice(CHANNELS),
            "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
            # Fraudulent clicks rarely lead to download
            "is_downloaded": np.random.choice([0, 1], p=[0.9, 0.1]) if is_fraud else np.random.choice([0, 1], p=[0.6, 0.4])
        }
        clicks.append(record)
    return clicks

def create_dataset():
    os.makedirs("data", exist_ok=True)
    fraud_users = int(NUM_USERS * FRAUD_RATIO)
    genuine_users = NUM_USERS - fraud_users

    all_clicks = []
    for i in range(fraud_users):
        ip = random_ip()
        all_clicks.extend(generate_clicks_for_user(ip, is_fraud=True))

    for i in range(genuine_users):
        ip = random_ip()
        all_clicks.extend(generate_clicks_for_user(ip, is_fraud=False))

    df = pd.DataFrame(all_clicks)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    out_path = "data/ad_clicks.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset created successfully: {out_path}")
    print(f"Total clicks: {len(df)}  |  Fraud ratio: {FRAUD_RATIO*100}%")
    print(df.head(10))

if __name__ == "__main__":
    create_dataset()
