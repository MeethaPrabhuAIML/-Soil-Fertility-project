import pandas as pd
import joblib
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Soil Fertility & Crop Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

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


class ChatMessage(BaseModel):
    message: str
    language: str = "en"


FAQ_KNOWLEDGE = [
    {
        "keywords": ["nitrogen", "n level", "high n", "low n"],
        "en": "Nitrogen (N) is essential for leafy growth and chlorophyll production. Too little nitrogen causes yellowing leaves and stunted growth; too much can delay flowering and make plants overly leafy at the expense of fruit or grain. Ideal levels depend on the crop, but most crops thrive between 200-350 kg/ha.",
        "ta": "நைட்ரஜன் (N) இலை வளர்ச்சிக்கும் பச்சையம் உற்பத்திக்கும் அவசியமானது. நைட்ரஜன் குறைவாக இருந்தால் இலைகள் மஞ்சளாகி வளர்ச்சி குன்றும்; அதிகமாக இருந்தால் பூக்கும் தாமதமாகி விளைச்சலை விட இலைகள் அதிகமாக வளரும். பயிரைப் பொறுத்து 200-350 கிலோ/ஹெக்டேர் அளவு சிறந்தது."
    },
    {
        "keywords": ["phosphorus", "p level"],
        "en": "Phosphorus (P) supports root development, flowering, and fruiting. Deficiency shows as purplish leaves and poor root growth. It's especially important in early plant growth stages for establishing a strong root system.",
        "ta": "பாஸ்பரஸ் (P) வேர் வளர்ச்சி, பூக்கும் தன்மை மற்றும் காய்ப்பதற்கு உதவுகிறது. குறைபாடு இருந்தால் இலைகள் ஊதா நிறமாகி வேர் வளர்ச்சி பலவீனமாக இருக்கும். செடி வளர்ச்சியின் ஆரம்ப கட்டத்தில் இது மிக முக்கியம்."
    },
    {
        "keywords": ["potassium", "k level"],
        "en": "Potassium (K) helps plants regulate water, resist disease, and improve overall crop quality (like fruit size and taste). Low potassium often shows as brown, scorched-looking leaf edges.",
        "ta": "பொட்டாசியம் (K) செடிகள் நீரை கட்டுப்படுத்தவும், நோய் எதிர்ப்பு சக்தி பெறவும், விளைச்சலின் தரத்தை மேம்படுத்தவும் உதவுகிறது. பொட்டாசியம் குறைவாக இருந்தால் இலை விளிம்புகள் பழுப்பு நிறமாக காணப்படும்."
    },
    {
        "keywords": ["ph", "acidic", "alkaline"],
        "en": "Soil pH measures acidity or alkalinity, on a scale from 0-14 (7 is neutral). Most crops prefer a slightly acidic to neutral pH of 6.0-7.5, where nutrients are most available. Very acidic or alkaline soil can lock up nutrients even if they're present.",
        "ta": "மண்ணின் pH அமிலத்தன்மை அல்லது காரத்தன்மையை அளவிடுகிறது (0-14 அளவீட்டில், 7 நடுநிலையானது). பெரும்பாலான பயிர்கள் 6.0-7.5 pH ஐ விரும்புகின்றன, அங்கு ஊட்டச்சத்துக்கள் அதிகம் கிடைக்கும். மிக அமிலத்தன்மை அல்லது காரத்தன்மை கொண்ட மண் ஊட்டச்சத்துக்களை பூட்டி வைக்கும்."
    },
    {
        "keywords": ["organic carbon", "organic matter"],
        "en": "Organic carbon indicates how much decomposed plant/animal matter is in your soil. Higher organic carbon improves water retention, nutrient availability, and soil structure. It's one of the strongest indicators of long-term soil health.",
        "ta": "கரிம கார்பன் உங்கள் மண்ணில் எவ்வளவு சிதைந்த தாவர/விலங்கு பொருள் உள்ளது என்பதைக் குறிக்கிறது. அதிக கரிம கார்பன் நீர் தேக்கத்தையும், ஊட்டச்சத்து கிடைப்பையும், மண் அமைப்பையும் மேம்படுத்துகிறது."
    },
    {
        "keywords": ["fertilizer", "urea", "dap", "npk fertilizer"],
        "en": "Urea is a nitrogen-rich fertilizer, best for leafy growth. DAP (Di-Ammonium Phosphate) provides both nitrogen and phosphorus, ideal for root and flower development. Choosing the right fertilizer depends on which nutrient your soil test shows is lacking.",
        "ta": "யூரியா நைட்ரஜன் நிறைந்த உரம், இலை வளர்ச்சிக்கு சிறந்தது. DAP (டை-அம்மோனியம் பாஸ்பேட்) நைட்ரஜன் மற்றும் பாஸ்பரஸ் இரண்டையும் வழங்குகிறது, வேர் மற்றும் பூ வளர்ச்சிக்கு ஏற்றது."
    },
    {
        "keywords": ["crop rotation", "rotate crops"],
        "en": "Crop rotation means growing different crops in the same field across seasons. It prevents soil nutrient depletion, breaks pest and disease cycles, and improves long-term soil health compared to growing the same crop repeatedly.",
        "ta": "பயிர் சுழற்சி என்பது ஒரே வயலில் பருவங்களுக்கு இடையே வெவ்வேறு பயிர்களை வளர்ப்பதாகும். இது மண் ஊட்டச்சத்து குறைவதைத் தடுக்கிறது, பூச்சி மற்றும் நோய் சுழற்சிகளை உடைக்கிறது."
    },
    {
        "keywords": ["how does this app work", "how does this work", "explain the app"],
        "en": "This app uses three machine learning models: one predicts your soil's fertility (Low/Medium/High) from nutrient readings, a second recommends the best crop for your conditions, and a third recommends a fertilizer — each explained using SHAP, a technique that shows exactly which factors drove the prediction.",
        "ta": "இந்த ஆப் மூன்று இயந்திர கற்றல் மாதிரிகளைப் பயன்படுத்துகிறது: ஒன்று உங்கள் மண்ணின் வளத்தன்மையை கணிக்கிறது, இரண்டாவது சிறந்த பயிரை பரிந்துரைக்கிறது, மூன்றாவது உரத்தை பரிந்துரைக்கிறது."
    }
]

DEFAULT_REPLY = {
    "en": "I can help with questions about NPK nutrients, soil pH, organic carbon, fertilizers, and crop rotation. Try asking about one of these topics, or check your prediction results above for specifics on your own soil sample.",
    "ta": "நான் NPK ஊட்டச்சத்துக்கள், மண் pH, கரிம கார்பன், உரங்கள் மற்றும் பயிர் சுழற்சி பற்றிய கேள்விகளுக்கு உதவ முடியும். இந்த தலைப்புகளில் ஒன்றைப் பற்றி கேளுங்கள்."
}


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health_check():
    return {"status": "API is running", "models_loaded": True}


@app.post("/chat")
def chat(chat_msg: ChatMessage):
    lang = chat_msg.language if chat_msg.language in ("en", "ta") else "en"
    message_lower = chat_msg.message.lower()

    for entry in FAQ_KNOWLEDGE:
        if any(keyword in message_lower for keyword in entry["keywords"]):
            return {"reply": entry[lang]}

    return {"reply": DEFAULT_REPLY[lang]}


@app.post("/predict")
def predict(soil: SoilInput):
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