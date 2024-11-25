import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from pycaret.regression import load_model, predict_model

# Load model
model = load_model('model_version_1')

# Create the app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    Year: int
    Country_Name: str
    NDVI: float
    MtCo2: float
    NightLight: float
    Land_Use_Tgc: float
    percipitation_winter: float
    percipitation_summer: float
    percipitation_spring: float
    percipitation_autumn: float
    Max_temperature: float
    Mean_temperature: float
    Min_temperature: float

@app.post('/predict')
def predict(request: PredictionRequest):
    data = {
        "Year": [request.Year],
        "Country Name": [request.Country_Name],
        "NDVI": [request.NDVI],
        "NightLight": [request.NightLight],
        "MtCo2": [request.MtCo2],
        "Land Use(Tgc)": [request.Land_Use_Tgc],
        "percipitation_winter": [request.percipitation_winter],
        "percipitation_summer": [request.percipitation_summer],
        "percipitation_spring": [request.percipitation_spring],
        "percipitation_autumn": [request.percipitation_autumn],
        "Max temperature": [request.Max_temperature],
        "Mean temperature": [request.Mean_temperature],
        "Min temperature": [request.Min_temperature]
    }
    df = pd.DataFrame(data)
    predictions = predict_model(model, data=df)
    first_prediction = predictions["prediction_label"].iloc[0]
    return {"prediction_label": first_prediction}

