import sys
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.pipeline import build_pipeline
from src.preprocessing import clean_data

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "spotify-recommender")
REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "spotify-kmeans-recommender",
)
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")


def configure_mlflow():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"file:///{MLRUNS_DIR.replace(os.sep, '/')}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


# Load data
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Final Dataset.csv")
raw_df = pd.read_csv(DATA_PATH)

df = clean_data(raw_df)

# Features
num_features = [
    'danceability','energy','loudness','speechiness',
    'acousticness','instrumentalness','liveness',
    'valence','tempo','duration_ms',
    'energy_dance_ratio','acoustic_softness','vocal_presence'
]

cat_features = ['key','mode','time_signature']

n_clusters = 5
pca_n_components = 0.9
pca_random_state = 42
kmeans_random_state = 42
kmeans_n_init = 10


# Build pipeline
pipeline = build_pipeline(num_features, cat_features, n_clusters=n_clusters)

configure_mlflow()

with mlflow.start_run(run_name="kmeans-pca-training"):
    mlflow.log_params({
        "raw_rows": len(raw_df),
        "training_rows": len(df),
        "num_features": len(num_features),
        "cat_features": len(cat_features),
        "n_clusters": n_clusters,
        "pca_n_components": pca_n_components,
        "pca_random_state": pca_random_state,
        "kmeans_random_state": kmeans_random_state,
        "kmeans_n_init": kmeans_n_init,
    })
    mlflow.log_param("numeric_feature_names", ",".join(num_features))
    mlflow.log_param("categorical_feature_names", ",".join(cat_features))

    pipeline.fit(df)
    clusters = pipeline.predict(df)

    # Evaluation
    X_transformed = pipeline[:-1].transform(df)
    score = silhouette_score(X_transformed, clusters)

    mlflow.log_metric("silhouette_score", score)
    mlflow.log_metric("pca_output_features", X_transformed.shape[1])

    cluster_counts = pd.Series(clusters).value_counts().sort_index().to_dict()
    for cluster, count in cluster_counts.items():
        mlflow.log_metric(f"cluster_{cluster}_count", count)

    print("Clusters:", set(clusters))
    print("Silhouette Score:", score)

    df["cluster"] = clusters

    cluster_labels = {
        0 : 'Energetic Happy',
        1 : 'Dance Party',
        2 : 'Instrumental Calm',
        3 : 'Acoustic Chill',
        4 : 'Rap / Spoken'
    }

    df["cluster_label"] = df['cluster'].map(cluster_labels)

    # Save dataset
    processed_data_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "tracks_with_clusters.csv",
    )
    df.to_csv(processed_data_path, index=False)

    # Save model
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    model_path = os.path.join(BASE_DIR, "models", "kmeans_pipeline.pkl")
    joblib.dump(pipeline, model_path)

    mlflow.log_artifact(processed_data_path, artifact_path="processed_data")
    mlflow.log_artifact(model_path, artifact_path="joblib_model")
    mlflow.log_dict(cluster_labels, "cluster_labels.json")
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
        code_paths=[os.path.join(BASE_DIR, "src")],
    )

    print(f"MLflow run_id: {mlflow.active_run().info.run_id}")
    print(f"Registered model: {REGISTERED_MODEL_NAME}")
