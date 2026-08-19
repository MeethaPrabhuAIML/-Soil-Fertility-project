# AI-Driven Soil Fertility Prediction & Sustainable Crop Recommendation

An ensemble machine learning system that predicts soil fertility, recommends a
suitable crop, and recommends a fertilizer — chained across three models, with
every prediction explained via SHAP.

**Live demo:** _add your deployed URL here after deploying_

## What it does

1. You enter soil readings (N, P, K, pH, EC, organic carbon, micronutrients)
   plus climate data (temperature, humidity, rainfall) and soil type.
2. A **Random Forest + XGBoost soft-voting ensemble** classifies fertility as
   Low / Medium / High.
3. A second ensemble recommends a **crop** from 22 possible crops based on
   soil + climate.
4. A third ensemble recommends a **fertilizer**, using the soil type and the
   *crop that was just predicted* — this is a real chained pipeline, not
   three independent models.
5. **SHAP (TreeExplainer)** breaks down which factors drove each prediction.

## Results (honestly reported)

| Model | Accuracy | Notes |
|---|---|---|
| Fertility | 88.6% overall | "High" class recall improved from 38%→50% via class weighting (39/880 rows are High — small sample, documented tradeoff toward recall) |
| Crop Recommendation | 99.55% | Verified no data leakage; reflects a well-separated, balanced dataset (2,200 rows, 22 classes) |
| Fertilizer Recommendation | 99% (5-fold CV) | Only 99 total rows — small dataset, cross-validated rather than trusting a single split |

## A real limitation, documented rather than hidden

The crop recommendation dataset (22 crops, including fruits) and the
fertilizer dataset (11 categories, mostly grains/cash crops) don't share the
same taxonomy. When the predicted crop has no direct match in the fertilizer
dataset (e.g. "jute" or a fruit), the API maps it to the closest available
category and says so explicitly in the response (`"note"` field) — it doesn't
silently guess.

## Tech stack

| Layer | Tech |
|---|---|
| ML | scikit-learn (Random Forest), XGBoost, SHAP |
| Backend / API | FastAPI, Pydantic, Uvicorn |
| Frontend | Vanilla HTML/CSS/JS, served directly by the backend |
| Data | 3 real public datasets (Kaggle) |
| Deployment | Docker, Render/Railway |

## Datasets used

- Soil Fertility Dataset (N, P, K, pH, EC, OC + micronutrients → fertility class)
- Crop Recommendation Dataset (N, P, K, temperature, humidity, pH, rainfall → crop)
- Fertilizer Prediction Dataset (soil type, crop type, NPK, temp, humidity, moisture → fertilizer)

## Project structure
````
Soil Fertility/
├── datasets/              # raw CSVs, untouched
├── data_processed/         # cleaned, encoded CSVs
├── models/                 # trained model files + encoders
├── frontend/index.html     # UI, served by the backend
├── main.py                 # FastAPI app
├── prepare_data.py         # encoding pipeline
├── train_fertility.py
├── train_crop.py
├── train_fertilizer.py
├── explain_fertility.py    # SHAP demo scripts
├── explain_crop.py
├── explain_fertilizer.py
├── requirements.txt
├── Dockerfile
└── README.md
````