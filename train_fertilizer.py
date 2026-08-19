import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

print(">>> RUNNING train_fertilizer.py <<<")

df = pd.read_csv("data_processed/fertilizer_ready.csv")

# Use the encoded versions of the categorical columns we created earlier
FEATURES = ['Temparature', 'Humidity ', 'Moisture', 'Soil_Type_encoded',
            'Crop_Type_encoded', 'Nitrogen', 'Potassium', 'Phosphorous']
TARGET = 'Fertilizer_encoded'

X = df[FEATURES]
y = df[TARGET]

print(f"Total rows: {len(df)}, Classes: {y.nunique()}")

# Only 99 rows total — stratify is essential here so tiny classes
# still appear in both train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Automatic balanced weighting — with this little data, don't over-engineer
# manual weights yet, see how it does first.
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,           # shallow — small dataset, avoid overfitting
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=150,      # fewer trees — small dataset
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric='mlogloss'
)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft',
    weights=[1, 1.2]
)
ensemble.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nEnsemble accuracy: {acc:.4f}")

fertilizer_encoder = joblib.load("models/fertilizer_name_encoder.joblib")
print("\nPer-class report:")
print(classification_report(
    y_test, y_pred,
    target_names=fertilizer_encoder.classes_,
    zero_division=0
))

xgb_for_shap = XGBClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9,
    random_state=42, eval_metric='mlogloss'
)
xgb_for_shap.fit(X_train, y_train, sample_weight=sample_weights)

joblib.dump(ensemble, "models/fertilizer_ensemble.joblib")
joblib.dump(xgb_for_shap, "models/fertilizer_xgb_only.joblib")
joblib.dump(FEATURES, "models/fertilizer_features.joblib")
print("\nSaved model to models/fertilizer_ensemble.joblib")

print(df.groupby('Crop Type')['Fertilizer Name'].nunique())
print(df.groupby('Soil Type')['Fertilizer Name'].nunique())

from sklearn.model_selection import cross_val_score

print("\n--- 5-fold cross-validation (more reliable with small data) ---")
cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
print(f"CV accuracy per fold: {cv_scores}")
print(f"CV mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")