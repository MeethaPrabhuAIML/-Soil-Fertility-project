import pandas as pd
import joblib
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Soil Fertility & Crop Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models... (this happens once, when the server starts)")

fertility_ensemble = joblib.load("models/fertility_ensemble.joblib")
fertility_xgb = joblib.load("models/fertility_xgb_only.joblib")
fertility_features = joblib.load("models/fertility_features.joblib")
fertility_explainer = shap.TreeExplainer(fertility_xgb)

crop_ensemble = joblib.load("models/crop_ensemble.joblib")
crop_xgb = joblib.load("models/crop_xgb_only.joblib")
crop_features = joblib.load("models/crop_features.joblib")
crop_label_encoder = joblib.load("models/crop_label_encoder.joblib")
crop_explainer = shap.TreeExplainer(crop_xgb)

fertilizer_ensemble = joblib.load("models/fertilizer_ensemble.joblib")
fertilizer_xgb = joblib.load("models/fertilizer_xgb_only.joblib")
fertilizer_features = joblib.load("models/fertilizer_features.joblib")
fertilizer_encoder = joblib.load("models/fertilizer_name_encoder.joblib")
soil_type_encoder = joblib.load("models/soil_type_encoder.joblib")
crop_type_encoder = joblib.load("models/crop_type_encoder.joblib")
fertilizer_explainer = shap.TreeExplainer(fertilizer_xgb)

print("All models loaded successfully.")

FERTILITY_CLASSES = ['Low', 'Medium', 'High']

# The crop recommendation model knows 22 crops (including fruits), but the
# fertilizer dataset only has 11 categories (mostly grains/cash crops).
# This maps each of the 22 possible predictions to its closest fertilizer
# category, so Stage 3 always has a valid input — documented here rather
# than hidden, since it's a real approximation.
CROP_TO_FERTILIZER_CATEGORY = {
    'rice': 'Paddy',
    'maize': 'Maize',
    'chickpea': 'Pulses',
    'kidneybeans': 'Pulses',
    'pigeonpeas': 'Pulses',
    'mothbeans': 'Pulses',
    'mungbean': 'Pulses',
    'blackgram': 'Pulses',
    'lentil': 'Pulses',
    'cotton': 'Cotton',
    'jute': 'Cotton',
    'coconut': 'Oil seeds',
    'coffee': 'Tobacco',
    'pomegranate': 'Ground Nuts',
    'banana': 'Ground Nuts',
    'mango': 'Ground Nuts',
    'grapes': 'Ground Nuts',
    'watermelon': 'Ground Nuts',
    'muskmelon': 'Ground Nuts',
    'apple': 'Ground Nuts',
    'orange': 'Ground Nuts',
    'papaya': 'Ground Nuts',
}


class SoilInput(BaseModel):
    N: float
    P: float
    K: float
    pH: float
    EC: float
    OC: float
    S: float
    Zn: float
    Fe: float
    Cu: float
    Mn: float
    B: float
    temperature: float
    humidity: float
    rainfall: float
    moisture: float
    soil_type: str


from fastapi.responses import FileResponse

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

@app.get("/health")
def health_check():
    return {"status": "API is running", "models_loaded": True}


@app.post("/predict")
def predict(soil: SoilInput):
    # ---------------- STAGE 1: FERTILITY ----------------
    fertility_row = pd.DataFrame([{
        'N': soil.N, 'P': soil.P, 'K': soil.K, 'pH': soil.pH,
        'EC': soil.EC, 'OC': soil.OC, 'S': soil.S, 'Zn': soil.Zn,
        'Fe': soil.Fe, 'Cu': soil.Cu, 'Mn': soil.Mn, 'B': soil.B
    }])[fertility_features]

    fertility_pred_idx = int(fertility_ensemble.predict(fertility_row)[0])
    fertility_probs = fertility_ensemble.predict_proba(fertility_row)[0]
    fertility_class = FERTILITY_CLASSES[fertility_pred_idx]
    fertility_confidence = float(fertility_probs[fertility_pred_idx])

    fert_shap = fertility_explainer.shap_values(fertility_row)
    fert_class_shap = fert_shap[0, :, fertility_pred_idx] if fert_shap.ndim == 3 else fert_shap[0]
    fert_contributions = sorted(
        zip(fertility_features, fert_class_shap),
        key=lambda x: abs(x[1]), reverse=True
    )[:3]
    fertility_top_factors = [f"{name} ({'+' if val > 0 else ''}{val:.3f})" for name, val in fert_contributions]

    # ---------------- STAGE 2: CROP ----------------
    crop_row = pd.DataFrame([{
        'N': soil.N, 'P': soil.P, 'K': soil.K,
        'temperature': soil.temperature, 'humidity': soil.humidity,
        'ph': soil.pH, 'rainfall': soil.rainfall
    }])[crop_features]

    crop_pred_idx = int(crop_ensemble.predict(crop_row)[0])
    crop_probs = crop_ensemble.predict_proba(crop_row)[0]
    predicted_crop = crop_label_encoder.inverse_transform([crop_pred_idx])[0]
    crop_confidence = float(crop_probs[crop_pred_idx])

    crop_shap = crop_explainer.shap_values(crop_row)
    crop_class_shap = crop_shap[0, :, crop_pred_idx] if crop_shap.ndim == 3 else crop_shap[0]
    crop_contributions = sorted(
        zip(crop_features, crop_class_shap),
        key=lambda x: abs(x[1]), reverse=True
    )[:3]
    crop_top_factors = [f"{name} ({'+' if val > 0 else ''}{val:.3f})" for name, val in crop_contributions]

    # ---------------- STAGE 3: FERTILIZER (chained off predicted crop) ----------------
    mapped_crop = CROP_TO_FERTILIZER_CATEGORY.get(predicted_crop, 'Ground Nuts')
    crop_was_approximated = mapped_crop != predicted_crop

    soil_type_enc = int(soil_type_encoder.transform([soil.soil_type])[0])
    crop_type_enc = int(crop_type_encoder.transform([mapped_crop])[0])

    fertilizer_row = pd.DataFrame([{
        'Temparature': soil.temperature, 'Humidity ': soil.humidity,
        'Moisture': soil.moisture, 'Soil_Type_encoded': soil_type_enc,
        'Crop_Type_encoded': crop_type_enc, 'Nitrogen': soil.N,
        'Potassium': soil.K, 'Phosphorous': soil.P
    }])[fertilizer_features]

    fert_pred_idx = int(fertilizer_ensemble.predict(fertilizer_row)[0])
    fert_probs = fertilizer_ensemble.predict_proba(fertilizer_row)[0]
    predicted_fertilizer = fertilizer_encoder.inverse_transform([fert_pred_idx])[0]
    fertilizer_confidence = float(fert_probs[fert_pred_idx])

    fertilizer_shap = fertilizer_explainer.shap_values(fertilizer_row)
    fertilizer_class_shap = fertilizer_shap[0, :, fert_pred_idx] if fertilizer_shap.ndim == 3 else fertilizer_shap[0]
    fertilizer_contributions = sorted(
        zip(fertilizer_features, fertilizer_class_shap),
        key=lambda x: abs(x[1]), reverse=True
    )[:3]
    fertilizer_top_factors = [f"{name} ({'+' if val > 0 else ''}{val:.3f})" for name, val in fertilizer_contributions]

    # ---------------- RETURN ----------------
    return {
        "fertility": {
            "class": fertility_class,
            "confidence": round(fertility_confidence, 4),
            "top_factors": fertility_top_factors
        },
        "recommended_crop": {
            "crop": predicted_crop,
            "confidence": round(crop_confidence, 4),
            "top_factors": crop_top_factors
        },
        "recommended_fertilizer": {
            "fertilizer": predicted_fertilizer,
            "confidence": round(fertilizer_confidence, 4),
            "top_factors": fertilizer_top_factors,
            "note": (
                f"Fertilizer dataset doesn't include '{predicted_crop}' directly; "
                f"approximated using the closest category '{mapped_crop}'."
            ) if crop_was_approximated else None
        }
    }