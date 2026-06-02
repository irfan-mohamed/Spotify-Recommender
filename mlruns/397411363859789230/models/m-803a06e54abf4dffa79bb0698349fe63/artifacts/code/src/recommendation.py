import os
import sys
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Loading artifacts

def load_artifacts(BASE_DIR):
    """
    Load trained pipeline and processed dataset
    """
    model_path = os.path.join(BASE_DIR, "models", "kmeans_pipeline.pkl")
    data_path = os.path.join(BASE_DIR, "data", "processed", "tracks_with_clusters.csv")

    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)

    return df, pipeline


# Recommendation using cosine similarity

def recommend(track_id, df, pipeline, top_n=10):

    if track_id not in df["id"].values:
        return {"error": "Track ID not found"}

    # Get target track
    track = df[df["id"] == track_id]

    # Get cluster
    cluster = track["cluster"].values[0]

    # Filter same cluster
    cluster_df = df[df["cluster"] == cluster].copy()

    # Transform features (exclude model step)
    X_cluster = pipeline[:-1].transform(cluster_df)
    X_track = pipeline[:-1].transform(track)

    # Similarity calculation
    similarities = cosine_similarity(X_track, X_cluster)[0]

    # Attach similarity
    cluster_df["similarity"] = similarities

    # Remove the same track
    cluster_df = cluster_df[cluster_df["id"] != track_id]

    # Sort by similarity
    recommendations = cluster_df.sort_values(
        by="similarity", ascending=False
    ).head(top_n)

    return track['cluster_label'].values[0], recommendations[["id", "name", "artist", "similarity"]]
