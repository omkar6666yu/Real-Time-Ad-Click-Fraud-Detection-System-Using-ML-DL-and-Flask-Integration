 
# AdClickFraudDetection

Project: Ad Click Fraud Detection Using Machine Learning and Deep Learning Algorithms
# Ad Click Fraud Detection Using Machine Learning and Deep Learning Algorithms

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)
![TensorFlow](https://img.shields.io/badge/DeepLearning-TensorFlow-orange.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)

---

> **Abstract:**  
> This project focuses on detecting fraudulent ad clicks in online and mobile advertising — a major issue causing significant financial losses for advertisers.  
> Using a dataset containing features such as **IP address, App ID, Device Type, Channel, and Timestamp**, the system predicts whether a click is **genuine or fraudulent**.  
> A combination of **Machine Learning models** (Logistic Regression, Random Forest, XGBoost, LightGBM) and **Deep Learning models** (ANN, LSTM, GRU, CNN) were implemented.  
> A **Stacking Classifier** integrates all models to improve detection accuracy.  
> The final system is deployed using a **Flask Web Application**, enabling users to input click data and receive **real-time fraud predictions** through a simple web interface.

---

## 👨‍💻 Project Members
| Name | Role |
|------|------|
| **Sawant Omkar Maruti** | Team Leader |
| **Gujjeti Shrikar Vidyasagar** | Team Member |

---

## 🧑‍🏫 Project Guide
- **Prof. Ramya Prabhakaran** — Primary Guide  

---

## 🎓 Subject Details
- **Class:** BE (Computer Engineering), Division A  
- **Academic Year:** 2025–2026  
- **Subject:** Major Project 1 *(MajPrj-1)*  
- **Project Type:** Major Project  
- **Institute:** Rizvi College of Engineering, Mumbai  

---

## ⚙️ Deployment Steps

Follow these steps to set up and run the project on your local machine:

1. **Clone the Repository**
   ```bash
   git clone  origin  https://github.com/omkar6666yu/Real-Time-Ad-Click-Fraud-Detection-System-Using-ML-DL-and-Flask-Integration.git.
origin  https://github.com/omkar6666yu/ 

   cd Ad-Click-Fraud-Detection-ML-DL


## Quick start

1. Create and activate virtual environment (recommended).
2. Install dependencies:
   pip install -r requirements.txt

3. Generate synthetic data (optional):
   python data/generate_synthetic_data.py

4. Train models:
   python models/train.py

5. Run Flask app:
   python app/app.py

6. Open browser to:
   http://127.0.0.1:5000

## Notes
- Training saves models to models/saved/
- Flask app loads saved models for inference

