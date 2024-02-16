from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

app = FastAPI()
try:
    model = joblib.load("cropmodel.pkl")
except Exception as e:
    print(f"Error loading the model: {e}")

class CropData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.get("/",tags=["Root"])
async def hello():
    return {"message": "Hello World"}

@app.post("/predict",tags=["Method"])
async def predict_crop(data: CropData):
    try:
        # Make predictions using the loaded model
        predictions = model.predict([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])

        # Return the predictions
        return {"crop_prediction": predictions[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))