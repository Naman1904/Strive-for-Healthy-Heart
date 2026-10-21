# Strive for a Healthy Heart

A small Flask application that demonstrates a heart-disease classification
model trained on the Cleveland heart dataset. Predictions are educational
only and are not medical advice or a diagnosis.

## Project layout

```text
heart_health/       Flask application and model code
  __init__.py       Application factory and routes
  predictor.py      Model loading, input scaling, and inference
  training.py       Model training entry point
templates/          Jinja pages
static/             Stylesheets and other static assets
app.py              WSGI entry point
prediction.py       Backwards-compatible model training command
heart_cleveland_upload.csv
heart-disease-prediction-knn-model.pkl
```

## Run locally

Use a Python version supported by the pinned dependencies in `requirements.txt`.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000> in a browser.

## Train the model

From the project root, run:

```powershell
python prediction.py
```

This retrains the Random Forest classifier and replaces the model artifact.
The Flask application can also be served with the included `Procfile`.

The checked-in classifier artifact does not include its fitted scaler. For
that existing artifact, the application rebuilds the scaler from the same
deterministic training split, so keep the CSV beside the model file.