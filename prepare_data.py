import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Make folders to keep things organized
os.makedirs("data_processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------- 1. FERTILITY DATASET ----------
# Already fully numeric, target already 0/1/2 — no encoding needed.
df1 = pd.read_csv("datasets/dataset1.csv")
df1.to_csv("data_processed/fertility_ready.csv", index=False)
print("Fertility dataset: no encoding needed, saved as-is.")
print(f"  Shape: {df1.shape}\n")

# ---------- 2. CROP RECOMMENDATION DATASET ----------
df2 = pd.read_csv("datasets/Crop_recommendation.csv")

crop_label_encoder = LabelEncoder()
df2['label_encoded'] = crop_label_encoder.fit_transform(df2['label'])

joblib.dump(crop_label_encoder, "models/crop_label_encoder.joblib")
df2.to_csv("data_processed/crop_ready.csv", index=False)

print("Crop dataset encoded.")
print(f"  Shape: {df2.shape}")
print(f"  Example mapping: {df2['label'].iloc[0]} -> {df2['label_encoded'].iloc[0]}\n")

# ---------- 3. FERTILIZER DATASET ----------
df3 = pd.read_csv("datasets/Fertilizer Prediction.csv")

soil_encoder = LabelEncoder()
crop_type_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

df3['Soil_Type_encoded'] = soil_encoder.fit_transform(df3['Soil Type'])
df3['Crop_Type_encoded'] = crop_type_encoder.fit_transform(df3['Crop Type'])
df3['Fertilizer_encoded'] = fertilizer_encoder.fit_transform(df3['Fertilizer Name'])

joblib.dump(soil_encoder, "models/soil_type_encoder.joblib")
joblib.dump(crop_type_encoder, "models/crop_type_encoder.joblib")
joblib.dump(fertilizer_encoder, "models/fertilizer_name_encoder.joblib")

df3.to_csv("data_processed/fertilizer_ready.csv", index=False)

print("Fertilizer dataset encoded.")
print(f"  Shape: {df3.shape}")
print(f"  Soil types found: {list(soil_encoder.classes_)}")
print(f"  Crop types found: {list(crop_type_encoder.classes_)}")
print(f"  Fertilizers found: {list(fertilizer_encoder.classes_)}")