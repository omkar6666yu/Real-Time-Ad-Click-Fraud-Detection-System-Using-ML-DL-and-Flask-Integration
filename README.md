# Ad Click Fraud Detection Using Machine Learning and Deep Learning Algorithms

> **Abstract**  
This project focuses on detecting fraudulent ad clicks in mobile advertising, a growing concern causing major financial losses. Using a dataset with features like IP address, app ID, device type, and timestamps, the system predicts whether a click results in an actual app download.  
A wide range of machine learning models—such as Logistic Regression, Random Forest, SVM, XGBoost, and LightGBM—alongside deep learning models including ANN, CNN, LSTM, and GRU were implemented.  
A Stacking Classifier further improves performance by combining multiple models.  
The system is deployed via a Flask web application, enabling users to input click data and receive real-time fraud predictions.

---

## Project Members
1. SAWANT OMKAR MARUTI  [ Team Leader ] 
2. GUJJETI SHRIKAR VIDYASAGAR 
3. JAIN NIKHIL KANTI

---

## Project Guides
1. PROF. MANILA GUPTA  [ Primary Guide ] 
2. PROF. RAMYA PRABHAKARAN


---

## Deployment Steps
1. Clone the repository
2. Install dependencies using requirements.txt
3. Place GeoLite2-Country.mmdb file in root directory
4. Run: python app.py
5. Open: http://127.0.0.1:5000

---

## Subject Details
- Class : BE (COMP) Div A - 2025-2026
- Subject : Major Project 1 (MajPrj-1)
- Project Type : Major Project

---

## Platform, Libraries and Frameworks used
1. Python (Flask)
2. Scikit-learn
3. TensorFlow / Keras
4. XGBoost
5. LightGBM
6. NumPy, Pandas
7. GeoIP2 # Note: GeoLite2 database is not included. Download from MaxMind.

---

## Dataset Used
1. https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection

---

## References
- https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection
- https://scikit-learn.org
- https://www.tensorflow.org


## 🌐 Supernova — Interactive Fraud Simulation Environment # Note: while opening the supernova make sure u open it from folder file not from live server

Supernova (ShopNova) is a real-time interactive demo platform built to simulate an e-commerce environment where ad click fraud detection can be visualized live.

It allows users to generate both legitimate and fraudulent clicks and observe how the system detects anomalies using engineered fraud signals.

---

### Key Capabilities

- Simulated e-commerce website (ads, products, user interactions)
- Bot simulation using draggable "Bot Ball"
- Trigger-based fraud generation (18 engineered fraud signals)
- Real-time API integration with Flask backend
- Live fraud monitoring dashboard
- GeoIP-based location tracking
- Continuous click streaming using SSE (Server-Sent Events)

---

### Fraud Simulation Features

Supernova enables testing of multiple fraud scenarios:

- Click burst attacks (high frequency clicks)
- Device and OS mismatch
- Impossible geo-location switching
- Subnet-based bot attacks
- Low inter-click intervals (ICI)
- High CTR anomalies
- User-Agent spoofing

---

### Interactive Testing

Users can:

- Click normally → simulate legitimate traffic
- Drag and drop bot → simulate automated fraud
- Trigger specific fraud patterns manually
- Observe system response in real time

---

### Integration

Supernova connects to backend APIs:

- `/api/predict` — real-time fraud prediction
- `/api/track` — click tracking from website
- `/api/stream` — live event monitoring

---

### Purpose

The goal of Supernova is to:

- Demonstrate real-world fraud detection behavior
- Provide a visual and interactive testing environment
- Help understand how ML models respond to different attack patterns

---

---
