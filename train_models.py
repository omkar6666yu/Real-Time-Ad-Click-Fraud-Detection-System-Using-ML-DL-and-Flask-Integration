"""
train_models.py — Train all 9 ML + DL models on TalkingData dataset
Run: python train_models.py
Output: saves best model to models/best_model.pkl
"""
import pandas as pd, numpy as np, joblib, os, json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

os.makedirs('models', exist_ok=True)

print("Loading dataset...")
# Download from Kaggle: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection
df = pd.read_csv('train_sample.csv')
print(f"Rows: {len(df)}")

print("Engineering features...")
df['click_time'] = pd.to_datetime(df['click_time'])
df['hour'] = df['click_time'].dt.hour
df['day']  = df['click_time'].dt.day
df['wday'] = df['click_time'].dt.dayofweek
df['ip_count']        = df.groupby('ip')['ip'].transform('count')
df['app_count']       = df.groupby('app')['app'].transform('count')
df['ip_app_count']    = df.groupby(['ip','app'])['ip'].transform('count')
df['ip_device_count'] = df.groupby(['ip','device'])['ip'].transform('count')
df['ip_hour_count']   = df.groupby(['ip','hour'])['ip'].transform('count')
df['ip_app_os_count'] = df.groupby(['ip','app','os'])['ip'].transform('count')

FEATURES = ['ip','app','device','os','channel','hour','day','wday',
            'ip_count','app_count','ip_app_count','ip_device_count',
            'ip_hour_count','ip_app_os_count']

X = df[FEATURES].fillna(0)
y = df['is_attributed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

results = {}
def evaluate(name, model):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else pred
    acc = round(accuracy_score(y_test,pred)*100, 2)
    f1  = round(f1_score(y_test,pred)*100, 2)
    auc = round(roc_auc_score(y_test,prob), 4)
    print(f"  {name:25s}  Acc={acc}%  F1={f1}%  AUC={auc}")
    results[name] = {'accuracy':acc,'f1':f1,'auc':auc}
    return model

print("\nTraining ML models...")
lr   = LogisticRegression(max_iter=500, class_weight='balanced').fit(X_train,y_train); evaluate('Logistic Regression',lr)
rf   = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X_train,y_train); evaluate('Random Forest',rf)
xgb  = XGBClassifier(scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(), n_estimators=200, random_state=42, eval_metric='auc').fit(X_train,y_train); evaluate('XGBoost',xgb)
lgbm = LGBMClassifier(class_weight='balanced', n_estimators=200, random_state=42).fit(X_train,y_train); evaluate('LightGBM',lgbm)

print("\nTraining Stacking Classifier...")
stacking = StackingClassifier(
    estimators=[('lr',LogisticRegression(max_iter=200,class_weight='balanced')),
                ('rf',RandomForestClassifier(n_estimators=50,class_weight='balanced',random_state=42)),
                ('xgb',XGBClassifier(scale_pos_weight=10,n_estimators=100,eval_metric='auc')),
                ('lgbm',LGBMClassifier(class_weight='balanced',n_estimators=100))],
    final_estimator=LogisticRegression(), cv=3, n_jobs=-1
).fit(X_train,y_train)
evaluate('Stacking Classifier', stacking)

print("\nTraining Deep Learning models...")
try:
    import tensorflow as tf
    from tensorflow import keras
    X_tr, X_te = X_train.values.astype('float32'), X_test.values.astype('float32')
    cw = {0:1, 1:int((y_train==0).sum()/(y_train==1).sum())}

    ann = keras.Sequential([keras.layers.Dense(128,activation='relu',input_shape=(X_tr.shape[1],)),
                             keras.layers.BatchNormalization(),keras.layers.Dropout(0.3),
                             keras.layers.Dense(64,activation='relu'),keras.layers.Dropout(0.2),
                             keras.layers.Dense(1,activation='sigmoid')])
    ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC'])
    ann.fit(X_tr,y_train,epochs=10,batch_size=1024,validation_split=0.1,verbose=0,class_weight=cw)
    ann.save('models/ann_model.h5')
    p = (ann.predict(X_te)>0.5).astype(int).flatten()
    print(f"  ANN  Acc={round(accuracy_score(y_test,p)*100,2)}%  F1={round(f1_score(y_test,p)*100,2)}%")

    X_seq = X_tr.reshape(X_tr.shape[0],1,X_tr.shape[1])
    X_te_seq = X_te.reshape(X_te.shape[0],1,X_te.shape[1])
    lstm = keras.Sequential([keras.layers.LSTM(64,input_shape=(1,X_tr.shape[1])),
                              keras.layers.Dropout(0.3),keras.layers.Dense(32,activation='relu'),
                              keras.layers.Dense(1,activation='sigmoid')])
    lstm.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC'])
    lstm.fit(X_seq,y_train,epochs=10,batch_size=1024,validation_split=0.1,verbose=0,class_weight=cw)
    lstm.save('models/lstm_model.h5')
    p = (lstm.predict(X_te_seq)>0.5).astype(int).flatten()
    print(f"  LSTM Acc={round(accuracy_score(y_test,p)*100,2)}%  F1={round(f1_score(y_test,p)*100,2)}%")
except ImportError:
    print("  TensorFlow not installed — skipping DL models")

best = max(results, key=lambda k: results[k]['f1'])
print(f"\nBest model: {best} (F1={results[best]['f1']}%)")
model_map = {'Logistic Regression':lr,'Random Forest':rf,'XGBoost':xgb,'LightGBM':lgbm,'Stacking Classifier':stacking}
joblib.dump(model_map.get(best, stacking), 'models/best_model.pkl')
joblib.dump({'features':FEATURES,'results':results,'best':best}, 'models/metadata.pkl')
print("Saved to models/best_model.pkl")
print("\nAll done! Run: python app.py")
