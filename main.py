import os
from typing import List
from contextlib import asynccontextmanager

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.recommendation import recommend


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================
# Request Schema
# ==========================

class RecommenderInput(BaseModel):
    track_id: str = Field(
        ...,
        min_length=1,
        description="Spotify track ID"
    )

    top_n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recommendations"
    )

class Recommendation(BaseModel):
    id : str
    name : str
    artist : str
    similarity : float

class RecommendationResponse(BaseModel):
    track_id : str
    cluster : str
    recommendations : List[Recommendation]

# ==========================
# Lifespan Event
# ==========================

@asynccontextmanager
async def lifespan(app: FastAPI):

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "kmeans_pipeline.pkl"
    )

    data_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "tracks_with_clusters.csv"
    )

    try:
        app.state.pipeline = joblib.load(model_path)
        app.state.df = pd.read_csv(data_path)

        print("Artifacts loaded successfully")

    except Exception as e:
        print(f"Failed to load artifacts: {e}")
        raise RuntimeError(
            f"Failed to load artifacts: {e}"
        )

    yield

    print("Application shutting down")


# ==========================
# FastAPI App
# ==========================

app = FastAPI(
    title="Spotify Recommender API",
    description="Music recommendation service using KMeans clustering",
    version="1.0.0",
    lifespan=lifespan
)


# ==========================
# Root Endpoint
# ==========================

@app.get("/")
def home():

    return {
        "message": "Spotify Recommender API is running"
    }


# ==========================
# Health Check
# ==========================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==========================
# Model Metadata
# ==========================

@app.get("/model-info")
def model_info():

    return {
        "model": "Spotify Recommender",
        "algorithm": "KMeans",
        "version": "1.0.0"
    }


# ==========================
# Recommendation Endpoint
# ==========================

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(data: RecommenderInput):

    pipeline = app.state.pipeline
    df = app.state.df
    if pipeline is None or df is None:
        raise HTTPException(
            status_code=500,
            detail="Artifacts not loaded"
        )
    
    if data.track_id not in df['id'].values:
        raise HTTPException(
            status_code = 404,
            detail = "Track Id is not Found"
        )
    
    label, result = recommend(
        track_id=data.track_id,
        df=df,
        pipeline=pipeline,
        top_n=data.top_n
    )

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(
            status_code=404,
            detail=result["error"]
        )

    return {
        "track_id": data.track_id,
        "cluster": label,
        "recommendations": result.to_dict(
            orient="records"
        )
    }