# models/ml_models.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

def train_logistic(X, y):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, y)
    return model

def train_random_forest(X, y, n_jobs=-1):
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=n_jobs, random_state=42)
    model.fit(X, y)
    return model

def train_xgb(X, y):
    model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', n_jobs=4)
    model.fit(X, y)
    return model

def train_lgbm(X, y):
    model = LGBMClassifier(n_estimators=300)
    model.fit(X, y)
    return model

def train_svm(X, y):
    model = SVC(probability=True)
    model.fit(X, y)
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

