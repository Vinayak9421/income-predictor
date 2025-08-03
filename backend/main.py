# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load the model
model = joblib.load("income_model.pkl")

# Create app
app = FastAPI()

# Allow requests from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body schema
class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]
    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 3)
    }
