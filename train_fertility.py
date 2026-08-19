import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

print(">>> RUNNING UPDATED SCRIPT <<<")   # sanity check this new version is executing

df = pd.read_csv("data_processed/fertility_ready.csv")
FEATURES = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']
TARGET = 'Output'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# class 0 = Low, 1 = Medium, 2 = High
class_weight_dict = {0: 1.0, 1: 1.0, 2: 6.0}
sample_weights = y_train.map(class_weight_dict).values
print(f"Weight dict in use: {class_weight_dict}")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=250,
    max_depth=5,
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

# THE FIX: pass sample_weight directly to ensemble.fit() so it actually
# reaches both underlying models when VotingClassifier fits its own clones.
ensemble.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nEnsemble accuracy: {acc:.4f}")
print("\nPer-class report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# Fit a standalone xgb (with the same weights) purely for SHAP explainability later
xgb_for_shap = XGBClassifier(
    n_estimators=250, max_depth=5, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85,
    random_state=42, eval_metric='mlogloss'
)
xgb_for_shap.fit(X_train, y_train, sample_weight=sample_weights)

joblib.dump(ensemble, "models/fertility_ensemble.joblib")
joblib.dump(xgb_for_shap, "models/fertility_xgb_only.joblib")
joblib.dump(FEATURES, "models/fertility_features.joblib")
print("\nSaved model to models/fertility_ensemble.joblib")