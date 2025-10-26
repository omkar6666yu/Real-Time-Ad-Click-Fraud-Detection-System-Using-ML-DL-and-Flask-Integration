# utils/evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

def evaluate_classification(model, X_test, y_test, name="Model"):
    try:
        y_proba = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)
    except:
        # TensorFlow model
        y_proba = model.predict(X_test).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = float('nan')
    print(f"== {name} Results ==")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    return {'accuracy':acc, 'precision':prec, 'recall':rec, 'f1':f1, 'auc':auc}
 