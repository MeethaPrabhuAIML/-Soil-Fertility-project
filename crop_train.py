import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

print(">>> RUNNING train_crop.py <<<")

df = pd.read_csv("data_processed/crop_ready.csv")

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET = 'label_encoded'   # the encoded version we created in prepare_data.py

X = df[FEATURES]
y = df[TARGET]

# Dataset is perfectly balanced (100 per class), so no class weighting needed here.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    eval_metric='mlogloss'
)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft',
    weights=[1, 1.2]
)
ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nEnsemble accuracy: {acc:.4f}")

# Load the label encoder we saved earlier, to show real crop names in the report
crop_label_encoder = joblib.load("models/crop_label_encoder.joblib")
print("\nPer-class report:")
print(classification_report(
    y_test, y_pred,
    target_names=crop_label_encoder.classes_
))

# Fit a standalone xgb too, for SHAP later
xgb_for_shap = XGBClassifier(
    n_estimators=250, max_depth=6, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85,
    random_state=42, eval_metric='mlogloss'
)
xgb_for_shap.fit(X_train, y_train)

joblib.dump(ensemble, "models/crop_ensemble.joblib")
joblib.dump(xgb_for_shap, "models/crop_xgb_only.joblib")
joblib.dump(FEATURES, "models/crop_features.joblib")
print("\nSaved model to models/crop_ensemble.joblib")

from sklearn.metrics import confusion_matrix
import numpy as np
cm = confusion_matrix(y_test, y_pred)
print("Any off-diagonal errors:", np.sum(cm) - np.trace(cm), "out of", np.sum(cm))