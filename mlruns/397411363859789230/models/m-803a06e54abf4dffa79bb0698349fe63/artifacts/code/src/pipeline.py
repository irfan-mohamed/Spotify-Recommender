from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.preprocessing import FullPreprocessor
from src.feature_engineering import FeatureEngineer

def build_pipeline(num_features, cat_features, n_clusters = 5):

    STEPS = [
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", FullPreprocessor(num_features, cat_features)),
        ("pca", PCA(n_components = 0.9, random_state = 42)),
        ("model", KMeans(n_clusters = n_clusters, random_state = 42, n_init = 10))
    ]

    pipeline = Pipeline(STEPS)

    return pipeline
