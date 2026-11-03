import io
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, uploadFile, Field
from fastapi.responses import StreamingResponse
from pydantic import Basemodel, Field

app = FastAPI()

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")

#input schema

class HouseFeatures(BaseModel):
    MedInc :   float = Field(gt=0,  description="median income of")
    HouseAge :   float = Field(gt=0,  description="Average age of")
    AveRooms :   float = Field(gt=0,  description="Average number of")
    Population :   float = Field(gt=0,  description="Total population of")
    Aveoccup :   float = Field(gt=0,  description="Average number of")
    Latitude :   float = Field(gt=32, le = 42,description="Latitude of")
    Longtitude :   float = Field(gt=125, le = 114,description="Logititude of")


#home
@app.get("/")
def home():
    return {
        "message" : "California house predication api",
        "status" : "running",
        "endpoint" : "send POST request to /predict"

    }

@app.get("/health")
def health():
    return{
        "status" : "running",
        "model" : "RandomForestRegressor",
        "features" : features,
        "avg_error" : "$39,000"
    }    

@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame({
            "MedInc" : house.MedInc,
            "HouseAge" : house.HouseAge,
            "AveRooms" : house.AveRooms,
            "AveRooms" : house.AveRooms,
            "Populations" : house.Population,
            "AveOccup" : house.AveOccup,
            "Latitude" : house.Latitude,
            "Longitude" : house.Longitude,

        })


        predicted = model.predict(input)[0]
        price_usd = predicted * 10000000

        return{
            "predicted_price" : f"${price_usd:,.2f}",
            "predicted_price_short" : f"${predicted:.2f} hundred thousands",
            "fidence_range" : f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}"

        }

except Exception as e:
    raise HTTPException(
        status_code=500,
        details= f"prediction failed: {str(e)}"
    )

@app.post("/predict-file")
async def predict_file(file: uploadFile=File(...)):

    if not file.filename.endwith("csv"):
        raise HTTPException(
            status_code=400,
            details="please upload a CSV file only"        
            )
    contents = await file.read()

    df = pd.DataFrame(io.BytesIO(contents))

           
