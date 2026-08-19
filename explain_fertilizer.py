import pandas as pd
import joblib
import shap

print(">>> RUNNING explain_fertilizer.py <<<")

xgb_model = joblib.load("models/fertilizer_xgb_only.joblib")
FEATURES = joblib.load("models/fertilizer_features.joblib")
fertilizer_encoder = joblib.load("models/fertilizer_name_encoder.joblib")

df = pd.read_csv("data_processed/fertilizer_ready.csv")
X = df[FEATURES]

sample_idx = 10   # arbitrary — try changing this later
sample = X.iloc[[sample_idx]]
print("\nSample being explained:")
print(sample)

actual_label = df['Fertilizer Name'].iloc[sample_idx]
print(f"\nActual fertilizer: {actual_label}")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(sample)
print(f"\nSHAP values shape: {shap_values.shape}")

predicted_class_idx = xgb_model.predict(sample)[0]
predicted_fertilizer = fertilizer_encoder.inverse_transform([predicted_class_idx])[0]
print(f"Model's predicted fertilizer: {predicted_fertilizer}")

if shap_values.ndim == 3:
    class_shap = shap_values[0, :, predicted_class_idx]
else:
    class_shap = shap_values[0]

contributions = list(zip(FEATURES, sample.iloc[0].values, class_shap))
contributions.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"\nTop factors driving the '{predicted_fertilizer}' recommendation:")
for feature, value, shap_val in contributions[:5]:
    direction = "pushes UP" if shap_val > 0 else "pushes DOWN"
    print(f"  {feature} = {value:.2f}  ->  {direction} by {abs(shap_val):.4f}")