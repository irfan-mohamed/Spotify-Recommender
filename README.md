# Spotify Music Recommender System

A machine learning-based recommendation system that suggests Spotify tracks based on audio features and clustering algorithms. This project uses K-Means clustering with PCA dimensionality reduction to group similar tracks and provides recommendations via a FastAPI REST API.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [API Documentation](#api-documentation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Project Overview

This project builds an intelligent music recommendation engine that:
- Processes Spotify track audio features and metadata
- Groups tracks into meaningful clusters (Energetic Happy, Party, Calm Acoustic, Balanced)
- Recommends similar tracks based on cosine similarity within the same cluster
- Exposes functionality through a REST API for easy integration

The system is designed to understand music characteristics beyond simple genre categorization by analyzing multiple audio features like danceability, energy, acousticness, and more.

---

## Features

- **Smart Clustering**: Groups tracks into 4 meaningful clusters based on audio characteristics
- **Content-Based Recommendations**: Uses cosine similarity to find similar tracks within clusters
- **Feature Engineering**: Creates derived features like energy-to-danceability ratio and acoustic softness
- **Multi-Label Genre Support**: Handles multiple genres per track
- **REST API**: FastAPI endpoint for easy integration with applications
- **Scalable Pipeline**: Sklearn pipeline with preprocessing, feature engineering, and modeling
- **Model Persistence**: Saves trained models for reproducible recommendations

---

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
Raw Data → Preprocessing → Feature Engineering → PCA → K-Means Clustering
                                                           ↓
                                                    Cluster Labels
                                                           ↓
                                                  Recommendation Engine
                                                           ↓
                                                      FastAPI Server
```

**Pipeline Flow:**
1. **Data Cleaning & Parsing** - Handles missing values, duplicates, and genre parsing
2. **Feature Engineering** - Creates derived features from audio characteristics
3. **Preprocessing** - Scales numerical features and encodes categorical features
4. **Dimensionality Reduction** - PCA reduces to 90% variance
5. **Clustering** - K-Means groups similar tracks into 4 clusters
6. **Recommendation** - Uses cosine similarity to find nearest neighbors

---

## Tech Stack

### Core Libraries
- **FastAPI** - Web framework for building the REST API
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-Learn** - Machine learning algorithms and preprocessing
- **Joblib** - Model serialization and persistence

### Key Technologies
- **K-Means Clustering** - Unsupervised learning for track grouping
- **PCA** - Dimensionality reduction
- **Cosine Similarity** - Distance metric for recommendations
- **ColumnTransformer** - Pipeline preprocessing
- **StandardScaler** - Feature normalization
- **OneHotEncoder** - Categorical feature encoding
- **MultiLabelBinarizer** - Multi-genre handling

---

## Project Structure

```
Recommender system/
├── main.py                          # FastAPI application entry point
├── requirement.txt                  # Project dependencies
├── README.md                        # This file
├── data/
│   ├── raw/
│   │   └── Final Dataset.csv       # Raw Spotify tracks data
│   └── processed/
│       └── tracks_with_clusters.csv # Processed data with cluster labels
├── models/
│   └── kmeans_pipeline.pkl         # Trained model pipeline
├── src/
│   ├── preprocessing.py            # Data cleaning & preprocessing
│   ├── feature_engineering.py      # Feature creation and transformation
│   ├── clustering.py               # Model training and evaluation
│   ├── pipeline.py                 # ML pipeline assembly
│   └── recommendation.py           # Recommendation engine
└── notebooks/
    └── poc_eda.ipynb               # Exploratory Data Analysis
```

---

## Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd "Recommender system"
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

4. **Prepare the data and train the model**
   ```bash
   python src/clustering.py
   ```
   This will:
   - Load the raw dataset from `data/raw/Final Dataset.csv`
   - Apply preprocessing and feature engineering
   - Train the K-Means model
   - Log parameters, metrics, artifacts, and the sklearn model to MLflow
   - Register the trained model as `spotify-kmeans-recommender`
   - Save the trained pipeline to `models/kmeans_pipeline.pkl`
   - Save processed data to `data/processed/tracks_with_clusters.csv`

   MLflow uses a local `mlruns` tracking store by default. To view runs and the
   model registry locally, run:
   ```bash
   mlflow ui --backend-store-uri ./mlruns
   ```

   Optional environment variables:
   - `MLFLOW_TRACKING_URI` - tracking server or backend store URI
   - `MLFLOW_EXPERIMENT_NAME` - experiment name, defaults to `spotify-recommender`
   - `MLFLOW_REGISTERED_MODEL_NAME` - registered model name, defaults to `spotify-kmeans-recommender`

### Share the registered model online with DagsHub

DagsHub provides a free hosted MLflow server and model registry for each
repository. To create a shareable model registry URL:

1. Create a free DagsHub account.
2. Create a DagsHub repository, for example `spotify-recommender`.
3. Create a DagsHub access token from your account settings.
4. Register the model to DagsHub:
   ```powershell
   .\scripts\register_model_dagshub.ps1 `
     -DagsHubUsername "your-dagshub-username" `
     -DagsHubRepo "spotify-recommender" `
     -DagsHubToken "your-dagshub-token"
   ```

After the run finishes, share this URL:
```text
https://dagshub.com/your-dagshub-username/spotify-recommender.mlflow/#/models/spotify-kmeans-recommender
```

Do not commit or paste your DagsHub token into source files. Pass it through the
script parameter or set it as a local environment variable only.

5. **Start the API server**
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`

---

## Data Pipeline

### Input Data Format
The raw dataset (`Final Dataset.csv`) contains the following columns:
- **Identifiers**: `id`, `name`, `artist`
- **Audio Features**:
  - `danceability` - How suitable for dancing (0-1)
  - `energy` - Intensity and activity (0-1)
  - `loudness` - Overall loudness (dB)
  - `speechiness` - Presence of spoken words (0-1)
  - `acousticness` - Acoustic vs electronic (0-1)
  - `instrumentalness` - Lack of vocals (0-1)
  - `liveness` - Audience presence (0-1)
  - `valence` - Musical positiveness (0-1)
  - `tempo` - Speed (BPM)
  - `duration_ms` - Track length (milliseconds)
- **Meta Features**: `key`, `mode`, `time_signature`, `genres`

### Processing Steps
1. **Parse Genres** - Convert string representations to lists
2. **Merge Duplicates** - Aggregate multiple versions of the same track
3. **Clean Data** - Remove unnecessary columns, handle missing values
4. **Fill Missing Values** - Use median for numerical features
5. **Data Type Fixes** - Ensure correct data types for categorical features

### Data Cleaning Details
- Drops columns: `Unnamed: 0.1`, `Unnamed: 0`, `type`, `uri`, `track_href`, `analysis_url`, `error`
- Removes duplicate tracks
- Fills null values in audio features with median
- Fills non-critical nulls (`acousticness`, `instrumentalness`) with 0

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
**GET** `/`

Returns the API status.

**Response:**
```json
{
  "message": "Spotify Recommender API is running"
}
```

#### 2. Get Recommendations
**GET** `/recommend/{track_id}`

Get music recommendations for a specific track.

**Parameters:**
- `track_id` (string, required): Spotify track ID
- `top_n` (integer, optional): Number of recommendations to return (default: 10)

**Response Success (200):**
```json
[
  "Party",
  [
    {
      "id": "track_id_1",
      "name": "Track Name",
      "artist": "Artist Name",
      "similarity": 0.95
    },
    {
      "id": "track_id_2",
      "name": "Another Track",
      "artist": "Another Artist",
      "similarity": 0.92
    }
  ]
]
```

**Response - Track Not Found (404):**
```json
{
  "detail": "Track ID not found"
}
```

**Response - Model Not Loaded (500):**
```json
{
  "detail": "Model not loaded"
}
```


## 📝 Notes

- The model is trained on the final dataset once during initialization
- Recommendations are computed on-the-fly using cosine similarity
- The API requires the trained model to be present in `models/kmeans_pipeline.pkl`
- All features are normalized to ensure fair contribution to similarity calculations

---

## 📄 License

This project is open source and available for educational and research purposes.

---

## 👥 Author

irfan-mohamed

---

## Contributing

Feel free to fork, submit issues, and create pull requests for improvements!

---

**Last Updated**: April 2026
