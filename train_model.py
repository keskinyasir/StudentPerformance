import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# --- VERİYİ YÜKLE ---
data = pd.read_csv("/Users/yasirkeskin/StudentPerformance-2/StudentsPerformance.csv")  # CSV dosyanın adı
X = data.drop(columns=["target", "is_dropout"])
y = data["is_dropout"]

# --- EĞİTİM/TEST AYRIMI ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- RANDOM FOREST MODELİ ---
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)

# --- MODEL DEĞERLENDİRME ---
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# --- MODELİ KAYDET ---
joblib.dump(rf_model, "rf_model.pkl")
print("Model rf_model.pkl olarak kaydedildi.")
