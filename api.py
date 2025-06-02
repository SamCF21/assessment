from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import numpy as np

app = FastAPI()

# Cargar el modelo entrenado y el codificador
modelo = joblib.load("modelo_crop.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")  # si guardaste el LabelEncoder

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predict")
def predict_crop(data: CropInput):
    features = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])
    tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = modelo(tensor)
        prediction = output.argmax(dim=1).item()

    crop_name = crop_encoder.inverse_transform([prediction])[0]
    return {"recommended_crop": crop_name}
