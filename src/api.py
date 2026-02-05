from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import logging
from src.data_processing import DataProcessor
from src.model import HousePricePredictor

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title = "Bengaluru House Price Prediction API",
    description = "Predict house prices in bangaluru based on property features",
    version = "1.0.0"
)

processor = DataProcessor()
predictor = HousePricePredictor()

class HouseFeatures(BaseModel):
    
    total_sqft: float = Field(..., gt=0, description="Total area in square feet")
    bhk: int = Field(..., ge=1, le=10, description="Number of bedrooms (BHK)")
    bath: int = Field(..., ge=1, le=10, description="Number of bathrooms")
    balcony: Optional[int] = Field(0, ge=0, le=5, description="Number of balconies")
    location: str = Field(..., description="Location in Bengaluru")
    area_type: Optional[str] = Field("Super built-up  Area", description="Type of area")
    availability: Optional[str] = Field("Ready To Move", description="Availability status")
    
    @field_validator('location')
    def validate_location(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Location cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "total_sqft": 1200,
                "bhk": 2,
                "bath": 2,
                "balcony": 1,
                "location": "Whitefield",
                "area_type": "Super built-up  Area",
                "availability": "Ready To Move"
            }
        }
        
class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    predicted_price_lakhs: float = Field(..., description="Predicted price in Lakhs")
    predicted_price_inr: float = Field(..., description="Predicted price in INR")
    confidence_interval: Optional[dict] = Field(None, description="95% confidence interval")
    input_features: dict = Field(..., description="Echo of input features")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    
#  API Endpoints

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Bengaluru House Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "locations": "/locations",
            "docs": "/docs"
        }
    }
    
@app.get("/locations", tags=["Metadata"])
async def get_locations():
    return {
        "locations": processor.get_available_locations(),
        "total": len(processor.get_available_locations())
    }
    
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if API and model are healthy."""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None
    }
    
@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def perdict_price(features: HouseFeatures):
    logger.info(f"Chal to rha hai mamla")
    try:
        # Convert to dict
        input_dict = features.model_dump()
        
        # Preprocess
        logger.info(f"Preprocessing input: {input_dict}")
        processed_features = processor.preprocess_input(input_dict)
        
        # Predict
        logger.info("Making prediction")
        prediction_lakhs = predictor.predict(processed_features)
        prediction_inr = prediction_lakhs * 100000
        
        # Confidence interval (simplified)
        std = prediction_lakhs * 0.1  # 10% uncertainty
        confidence = {
            "lower_lakhs": prediction_lakhs - 1.96 * std,
            "upper_lakhs": prediction_lakhs + 1.96 * std,
            "lower_inr": (prediction_lakhs - 1.96 * std) * 100000,
            "upper_inr": (prediction_lakhs + 1.96 * std) * 100000
        }
        
        logger.info(f"Prediction: {prediction_lakhs:.2f} Lakhs")
        
        return {
            "predicted_price_lakhs": round(prediction_lakhs, 2),
            "predicted_price_inr": round(prediction_inr, 2),
            "confidence_interval": confidence,
            "input_features": input_dict
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
               
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )