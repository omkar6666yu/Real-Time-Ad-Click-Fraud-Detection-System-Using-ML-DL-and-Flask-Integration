# models/stacking.py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

def get_oof_predictions(clf, X, y, n_splits=5):
    oof = np.zeros((X.shape[0],))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        oof[val_idx] = clf.predict_proba(X[val_idx])[:,1]
    clf.fit(X, y)
    return oof, clf

def build_stacking_meta(X, y, base_clfs):
    meta_features = []
    trained_bases = {}
    for name, clf in base_clfs:
        oof, trained = get_oof_predictions(clf, X, y)
        meta_features.append(oof.reshape(-1,1))
        trained_bases[name] = trained
        print(f"Base {name} OOF AUC:", roc_auc_score(y, oof))
    meta_X = np.hstack(meta_features)
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(meta_X, y)
    return trained_bases, meta_clf

def save_trained_bases(trained_bases, meta_clf, path_prefix="models/saved/stack_"):
    import joblib
    for name, clf in trained_bases.items():
        joblib.dump(clf, f"{path_prefix}{name}.pkl")
    joblib.dump(meta_clf, f"{path_prefix}meta.pkl")
