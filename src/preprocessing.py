"""Construccion del Pipeline completo: limpieza -> preprocesamiento -> modelo."""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE
from .data import clean_features


def build_preprocessor(max_tfidf_features: int = 300) -> ColumnTransformer:
    """Imputacion, escalado, codificacion y TF-IDF. Se ajusta solo con datos de entrenamiento."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Categorias no vistas en entrenamiento: se codifican como todo ceros (se ignoran).
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    text = TfidfVectorizer(
        max_features=max_tfidf_features,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("text", text, TEXT_FEATURE),
        ]
    )


def build_pipeline(estimator, max_tfidf_features: int = 300) -> Pipeline:
    """Pipeline unico: el artefacto guardado contiene limpieza, preprocesamiento y modelo."""
    return Pipeline(
        [
            ("cleaner", FunctionTransformer(clean_features, validate=False)),
            ("preprocessor", build_preprocessor(max_tfidf_features)),
            ("model", estimator),
        ]
    )
