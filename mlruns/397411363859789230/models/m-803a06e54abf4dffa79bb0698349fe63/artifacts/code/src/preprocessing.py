import pandas as pd
import numpy as np
import ast

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer

# Parse the genres to convert into list
def parse_genres(df: pd.DataFrame) -> pd.DataFrame:

    def parse_genre(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except:
            return []

    df = df.copy()
    df["genres"] = df["genres"].apply(parse_genre)

    return df

# Merging multiple tracks
def merge_tracks(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = df.groupby("id").agg({
        "name": "first",
        "artist": "first",
        "genres": lambda x: list(set([g for sublist in x for g in sublist])),
        "danceability": "first",
        "energy": "first",
        "key": "first",
        "loudness": "first",
        "mode": "first",
        "speechiness": "first",
        "acousticness": "first",
        "instrumentalness": "first",
        "liveness": "first",
        "valence": "first",
        "tempo": "first",
        "duration_ms": "first",
        "time_signature": "first"
    }).reset_index()

    return df

# Data Cleaning

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Drop unnecessary columns 
    df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'type', 'uri', 'track_href', 'analysis_url', 'error'], inplace = True)

    # Drop duplicates
    df = df.drop_duplicates()

    # Fill nulls with median
    num_cols = [
        'danceability','energy','loudness','speechiness',
        'liveness','valence','tempo','duration_ms'
    ]
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill non-critical nulls
    df["acousticness"] = df["acousticness"].fillna(0)
    df["instrumentalness"] = df["instrumentalness"].fillna(0)

    # Data Type Fixing
    for col in ["key", "mode", "time_signature"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    df = parse_genres(df)
    df = merge_tracks(df)

    return df

# Genre Encodes

class GenreEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X):
        # Passing the pandas series list of genre
        self.mlb.fit(X)
        return self

    def transform(self, X):
        genre_encoded = self.mlb.transform(X)
        genre_df = pd.DataFrame(
            genre_encoded,
            columns=self.mlb.classes_,
            index=X.index
        )
        return genre_df


# Column transformer builder.

def build_preprocessor(num_features, cat_features):

    num_pipeline = StandardScaler()

    cat_pipeline = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
        ],
        remainder="drop"
    )

    return preprocessor


# Full preprocessor

class FullPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_features, cat_features):
        self.genre_encoder = GenreEncoder()
        self.preprocessor = build_preprocessor(num_features, cat_features)

    def fit(self,X):
        self.genre_encoder.fit(X["genres"])
        self.preprocessor.fit(X)
        return self
    
    def transform(self, X):
        genre_encoded = self.genre_encoder.transform(X["genres"])
        other_features = self.preprocessor.transform(X)

        return  np.hstack([other_features, genre_encoded])