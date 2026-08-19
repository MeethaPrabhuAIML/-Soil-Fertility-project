import pandas as pd
import joblib
import shap

print(">>> RUNNING explain_fertility.py <<<")

# Load the standalone XGBoost model we saved specifically for SHAP
xgb_model = joblib.load("models/fertility_xgb_only.joblib")
FEATURES = joblib.load("models/fertility_features.joblib")

# Load the data again to grab a real sample to explain
df = pd.read_csv("data_processed/fertility_ready.csv")
X = df[FEATURES]

# Pick one real sample to explain (row 4, arbitrary choice)
sample = X.iloc[[4]]
print("\nSample being explained:")
print(sample)

actual_class = df['Output'].iloc[4]
class_names = ['Low', 'Medium', 'High']
print(f"\nActual label: {class_names[actual_class]}")

# ---------- Create the SHAP explainer ----------
explainer = shap.TreeExplainer(xgb_model)

# Get SHAP values for this sample
shap_values = explainer.shap_values(sample)

# For multi-class models, shap_values has one array of contributions PER CLASS
print(f"\nSHAP values shape: {shap_values.shape}")

# Predict which class the model actually chose
predicted_class = xgb_model.predict(sample)[0]
print(f"Model's predicted class: {class_names[predicted_class]}")

# Get the contributions specifically for the predicted class
if shap_values.ndim == 3:
    class_shap = shap_values[0, :, predicted_class]
else:
    class_shap = shap_values[0]

# Pair each feature with its contribution and sort by absolute impact
contributions = list(zip(FEATURES, sample.iloc[0].values, class_shap))
contributions.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"\nTop factors driving the '{class_names[predicted_class]}' prediction:")
for feature, value, shap_val in contributions[:5]:
    direction = "pushes UP" if shap_val > 0 else "pushes DOWN"
    print(f"  {feature} = {value:.2f}  ->  {direction} by {abs(shap_val):.4f}")