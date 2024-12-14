import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from pycaret.regression import load_model, predict_model
from supabase import create_client, Client
supabase_url = "https://ycimfxsnhtqakxqaknmj.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InljaW1meHNuaHRxYWt4cWFrbm1qIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQwMjI2MDgsImV4cCI6MjA0OTU5ODYwOH0.cRPg4-kAkBN_H8R38_fNR8k3sV8o7FL1yafAy0wWISU"

# Initialize the Supabase clien
supabase: Client = create_client(supabase_url, supabase_key)

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
    
    Country_Name: str
    MtCo2: float
    

@app.post('/predict')
def predict(request: PredictionRequest):
    response = supabase.table("yearlydata").select("*").eq("Country Name", request.Country_Name).eq("Year", 2021).execute()

    if not response.data:
        return {"error": "No data found for the given country and year"}

    record = response.data[0]  # Safely get the first record

    
    data = {
        "Year": [record['Year']],
        "Country Name": [request.Country_Name],
        "NDVI": [record['NDVI']],
        "NightLight": [record['NightLight']],
        "MtCo2": [request.MtCo2],
        "Land Use(Tgc)": [record['Land Use(Tgc)']],
        "percipitation_winter": [record['percipitation_winter']],
        "percipitation_summer": [record['percipitation_summer']],
        "percipitation_spring": [record['percipitation_spring']],
        "percipitation_autumn": [record['percipitation_autumn']],
        "Max temperature": [record['Max temperature']],
        "Mean temperature": [record['Mean temperature']],
        "Min temperature": [record['Min temperature']]
    }
    df = pd.DataFrame(data)
    predictions = predict_model(model, data=df)
    first_prediction = predictions["prediction_label"].iloc[0]
    return {"prediction_label": first_prediction}

