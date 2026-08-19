import pandas as pd
import joblib
import shap

print(">>> RUNNING explain_crop.py <<<")

xgb_model = joblib.load("models/crop_xgb_only.joblib")
FEATURES = joblib.load("models/crop_features.joblib")
crop_label_encoder = joblib.load("models/crop_label_encoder.joblib")

df = pd.read_csv("data_processed/crop_ready.csv")
X = df[FEATURES]

# Pick one real sample to explain
sample_idx = 500   # arbitrary — pick any row you like
sample = X.iloc[[sample_idx]]
print("\nSample being explained:")
print(sample)

actual_label = df['label'].iloc[sample_idx]
print(f"\nActual crop: {actual_label}")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(sample)
print(f"\nSHAP values shape: {shap_values.shape}")

predicted_class_idx = xgb_model.predict(sample)[0]
predicted_crop = crop_label_encoder.inverse_transform([predicted_class_idx])[0]
print(f"Model's predicted crop: {predicted_crop}")

if shap_values.ndim == 3:
    class_shap = shap_values[0, :, predicted_class_idx]
else:
    class_shap = shap_values[0]

contributions = list(zip(FEATURES, sample.iloc[0].values, class_shap))
contributions.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"\nTop factors driving the '{predicted_crop}' recommendation:")
for feature, value, shap_val in contributions[:5]:
    direction = "pushes UP" if shap_val > 0 else "pushes DOWN"
    print(f"  {feature} = {value:.2f}  ->  {direction} by {abs(shap_val):.4f}")