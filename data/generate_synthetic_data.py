"""
Synthetic Ad Click Data Generator
Generates data/ad_clicks.csv
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

NUM_USERS = 2000
CLICKS_PER_USER = (5, 40)
FRAUD_RATE = 0.25
START_DATE = datetime(2025, 1, 1)

APP_IDS = [f"app_{i}" for i in range(1, 21)]
DEVICES = ["mobile", "tablet", "desktop"]

def random_ip():
    return ".".join(str(random.randint(1, 255)) for _ in range(4))

def generate_user_clicks(ip, app_id, device, is_fraud):
    n_clicks = random.randint(*CLICKS_PER_USER)
    clicks = []
    time = START_DATE + timedelta(days=random.randint(0, 180), hours=random.randint(0, 23), minutes=random.randint(0,59))
    for i in range(n_clicks):
        if is_fraud:
            gap = random.randint(1, 20)  # seconds
        else:
            gap = random.randint(30, 600)  # seconds
        time += timedelta(seconds=gap)
        if is_fraud:
            is_downloaded = np.random.choice([0, 1], p=[0.95, 0.05])
        else:
            is_downloaded = np.random.choice([0, 1], p=[0.6, 0.4])
        clicks.append({
            "ip": ip,
            "app_id": app_id,
            "device": device,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_downloaded": int(is_downloaded)
        })
    return clicks

def generate_dataset(num_users=NUM_USERS, fraud_rate=FRAUD_RATE):
    all_clicks = []
    num_fraud_users = int(num_users * fraud_rate)
    fraud_ips = [random_ip() for _ in range(num_fraud_users)]
    genuine_ips = [random_ip() for _ in range(num_users - num_fraud_users)]

    for ip in fraud_ips:
        app_id = random.choice(APP_IDS)
        device = random.choice(DEVICES)
        all_clicks.extend(generate_user_clicks(ip, app_id, device, True))

    for ip in genuine_ips:
        app_id = random.choice(APP_IDS)
        device = random.choice(DEVICES)
        all_clicks.extend(generate_user_clicks(ip, app_id, device, False))

    df = pd.DataFrame(all_clicks)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def main():
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    out = "data/ad_clicks.csv"
    df.to_csv(out, index=False)
    print(f"Synthetic dataset written to {out}")
    print(df.head())

if __name__ == "__main__":
    main()
