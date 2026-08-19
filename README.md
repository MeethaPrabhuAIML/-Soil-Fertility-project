# TerraIQ — AI-Driven Soil Fertility Prediction & Sustainable Crop Recommendation

**Live app:** [terraiq-hvqx.onrender.com](https://terraiq-hvqx.onrender.com)

*(Free-tier hosting: if the app has been idle, the first load can take 30–60 seconds while the server wakes up — this is normal and not a bug.)*

An ensemble machine learning system that predicts soil fertility, recommends a
crop, and recommends a fertilizer — three models chained together, every
prediction explained with SHAP, wrapped in a bilingual (English/Tamil)
interface with live charts and a built-in Q&A assistant.

## What it does

1. Enter soil readings (N, P, K, pH, EC, organic carbon, micronutrients) plus
   climate data (temperature, humidity, rainfall) and soil type.
2. A **Random Forest + XGBoost soft-voting ensemble** classifies soil
   fertility as Low / Medium / High.
3. A second ensemble recommends the best-fit **crop** from 22 possibilities.
4. A third ensemble recommends a **fertilizer**, using the soil type and the
   crop *just predicted* — a genuinely chained, multi-stage pipeline, not
   three independent models bolted together.
5. **SHAP (TreeExplainer)** breaks down exactly which factors drove each
   prediction, visualized as a bar chart, not just returned as raw numbers.
6. A built-in **assistant** answers common soil/farming questions (NPK,
   pH, fertilizers, crop rotation) in English or Tamil.

## Results (reported honestly, not cherry-picked)

| Model | Accuracy | Notes |
|---|---|---|
| Fertility | 88.6% overall | "High" class recall improved 38%→50% via class weighting — only 39/880 rows are High, a small sample, and this tradeoff (favoring recall over precision on the rare class) is a deliberate, documented choice |
| Crop Recommendation | 99.55% | Verified no data leakage; reflects a clean, balanced, well-separated dataset (2,200 rows, 22 classes) |
| Fertilizer Recommendation | 99% (5-fold CV) | Only 99 total rows — small dataset, so results are cross-validated rather than trusted from a single train/test split |

## A real limitation, documented rather than hidden

The crop dataset (22 crops, including many fruits) and the fertilizer dataset
(11 categories, mostly grains and cash crops) don't share the same taxonomy.
When the predicted crop has no direct match in the fertilizer dataset (e.g.
"jute" or a fruit like "mango"), the API maps it to the closest available
category and says so explicitly in the response — it never silently guesses.

## Tech stack

| Layer | Tech |
|---|---|
| ML | scikit-learn (Random Forest), XGBoost, SHAP |
| Backend / API | FastAPI, Pydantic, Uvicorn |
| Frontend | Vanilla HTML/CSS/JS, Chart.js, served directly by the backend |
| Assistant | Rule-based bilingual keyword matcher (no external API, no cost) |
| Data | 3 real public datasets (Kaggle) |
| Deployment | Docker, Render |

## Datasets used

- **Soil Fertility Dataset** — N, P, K, pH, EC, organic carbon + 6
  micronutrients → fertility class
- **Crop Recommendation Dataset** — N, P, K, temperature, humidity, pH,
  rainfall → crop (22 classes, perfectly balanced)
- **Fertilizer Prediction Dataset** — soil type, crop type, NPK, temperature,
  humidity, moisture → fertilizer name

## Project structure

```
Soil Fertility/
├── datasets/                 # raw CSVs, untouched
├── data_processed/           # cleaned, encoded CSVs
├── models/                   # trained model files + encoders
├── frontend/index.html       # bilingual UI with charts + chat, served by the backend
├── main.py                   # FastAPI app: /predict, /chat, serves the frontend
├── prepare_data.py           # encoding pipeline
├── train_fertility.py
├── train_crop.py
├── train_fertilizer.py
├── explain_fertility.py      # SHAP demo scripts
├── explain_crop.py
├── explain_fertilizer.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Run it locally

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** — the UI, prediction API, and chat assistant
are all served from the same process.

Interactive API docs: **http://127.0.0.1:8000/docs**

## API reference

`POST /predict` — takes soil + climate readings, returns fertility class,
recommended crop, recommended fertilizer, and SHAP-ranked reasoning for each.

`POST /chat` — takes a question and a language code (`en` or `ta`), returns
an answer from the built-in soil/farming knowledge base.

`GET /health` — service status check.

## Deployment

Deployed on [Render](https://render.com) via the included `Dockerfile` —
push to GitHub, connect the repo as a Web Service, Render builds and runs it
automatically. No environment variables or secrets are required, since the
assistant is fully rule-based and needs no external API key.

## Model notes

- The fertility and fertilizer ensembles use **soft voting** between Random
  Forest and XGBoost, with class weighting applied to handle imbalanced rare
  classes (documented above).
- SHAP values are computed via `TreeExplainer` on each model's XGBoost
  component — the same feature set the ensemble votes on, so it's a faithful
  proxy for "why" the ensemble decided what it did.
- The chat assistant is intentionally rule-based rather than LLM-backed, to
  keep the app fully free to run and deploy with no API costs or rate
  limits, while still answering real, common soil-science questions
  accurately in both languages.