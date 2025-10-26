# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib

app = FastAPI()

# ---------- Input Model ----------
class FarmerInput(BaseModel):
    soil_type: str          # Loamy / Clay / Sandy / Black
    season: str             # Rabi / Kharif / Summer
    state: str = None       # optional
    district: str = None    # optional
    market: str = None      # optional

# ---------- Load Model ----------
try:
    model = tf.keras.models.load_model("model.keras", compile=False)
    scaler, label_encoders = joblib.load("preprocessors.pkl")
    print("✅ Model and preprocessors loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None

# ---------- Soil & Season Mapping ----------
soil_map = {
    "Loamy": {"N": 50, "P": 40, "K": 50, "ph": 6.5},
    "Clay": {"N": 40, "P": 30, "K": 40, "ph": 7.0},
    "Sandy": {"N": 30, "P": 20, "K": 30, "ph": 6.0},
    "Black": {"N": 60, "P": 50, "K": 60, "ph": 7.5},
}

season_map = {
    "Rabi": {"temperature": 20, "humidity": 50, "rainfall": 100},
    "Kharif": {"temperature": 28, "humidity": 70, "rainfall": 300},
    "Summer": {"temperature": 35, "humidity": 40, "rainfall": 50},
}

# ---------- Prediction Endpoint ----------
@app.post("/predict")
async def predict_crop(farmer_input: FarmerInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Map soil & season
        if farmer_input.soil_type not in soil_map:
            raise HTTPException(status_code=400, detail="Invalid soil type")
        if farmer_input.season not in season_map:
            raise HTTPException(status_code=400, detail="Invalid season")

        soil = soil_map[farmer_input.soil_type]
        season = season_map[farmer_input.season]

        # Encode state, district, market (if provided)
        state_encoded = (
            label_encoders['state'].transform([farmer_input.state])[0]
            if farmer_input.state and farmer_input.state in label_encoders['state'].classes_
            else 0
        )
        district_encoded = (
            label_encoders['district'].transform([farmer_input.district])[0]
            if farmer_input.district and farmer_input.district in label_encoders['district'].classes_
            else 0
        )
        market_encoded = (
            label_encoders['market'].transform([farmer_input.market])[0]
            if farmer_input.market and farmer_input.market in label_encoders['market'].classes_
            else 0
        )

        # Prepare input array
        X = np.array([[
            soil["N"], soil["P"], soil["K"],
            season["temperature"], season["humidity"],
            soil["ph"], season["rainfall"],
            state_encoded, district_encoded, market_encoded
        ]])

        # Scale numeric features (first 7 columns)
        scaled_X = scaler.transform(X[:, :7])
        final_X = np.hstack([scaled_X, X[:, 7:]])

        # Predict
        predicted_price = float(model.predict(final_X)[0][0])

        # Return top crops (mock example)
        top_crops = ["Rice", "Wheat", "Corn"]
        suitable_crop = top_crops[0]

        return {
            "predicted_price": predicted_price,
            "suitable_crop": suitable_crop,
            "top_crops": top_crops
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
