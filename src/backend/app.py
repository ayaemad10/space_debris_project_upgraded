"""
FastAPI backend for the Space Debris Collision Prediction System.

Fixes vs the originally uploaded app.py:
  1. The dataset has NO `mission_id` column (it's already-scaled, feature-
     engineered conjunction event data — 103 float columns, no identifiers).
     `/missions` and `/statistics` previously crashed on that assumption.
     Replaced with a synthetic *grouping over existing rows*
     (`event_batch = row_index // BATCH_SIZE`), not synthetic data.
  2. `risk` in this dataset is log10(collision probability), ranging
     roughly -30 to -3, NOT a 0-1 score. The decision thresholds (0.5/0.8)
     only make sense applied to the model's predicted probability output,
     not to the raw `risk` column. Fixed accordingly — see
     src/scripts/decision_engine.py for the shared logic.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from decision_engine import decide  # noqa: E402

app = FastAPI(title="Space Debris Collision Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "lstm_model.keras"
DATA_PATH = BASE_DIR / "data" / "engineered_data.csv"
BATCH_SIZE = 500  # rows per synthetic "event_batch" grouping

model = None
df = None


@app.on_event("startup")
def load_artifacts():
    global model, df
    import tensorflow as tf  # imported lazily so /health works even if TF is missing
    model = tf.keras.models.load_model(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    df["event_batch"] = df.index // BATCH_SIZE


class PredictionRequest(BaseModel):
    features: list


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_level: str
    recommended_action: str
    confidence: float


@app.get("/")
def read_root():
    return {"message": "Space Debris Collision Prediction API"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = np.array(request.features, dtype="float32").reshape(1, -1)
        reshaped = features.reshape((1, 1, features.shape[1]))
        probability = float(model.predict(reshaped, verbose=0)[0][0])

        decision = decide(probability)

        return PredictionResponse(
            risk_probability=probability,
            risk_level=decision.risk_level,
            recommended_action=decision.action,
            confidence=abs(probability - 0.5) * 2,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/statistics")
def get_statistics():
    return {
        "total_events": len(df),
        "avg_log_risk": float(df["risk"].mean()),
        "avg_miss_distance": float(df["miss_distance"].mean()),
        "high_risk_events_pc_gt_1e-6": int((df["risk"] > -6).sum()),
    }


@app.get("/missions")
def get_missions():
    """Returns synthetic event-batch IDs (see module docstring) since the
    dataset has no true mission identifier column."""
    batches = sorted(df["event_batch"].unique().tolist())
    return {"event_batches": batches[:20]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
