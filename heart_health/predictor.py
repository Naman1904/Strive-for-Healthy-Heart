import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "heart-disease-prediction-knn-model.pkl"
DATA_PATH = PROJECT_ROOT / "heart_cleveland_upload.csv"
FEATURE_COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)


class HeartDiseasePredictor:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        data_path: Path = DATA_PATH,
    ) -> None:
        with model_path.open("rb") as model_file:
            self.model = pickle.load(model_file)

        dataset = pd.read_csv(data_path)
        features = dataset.loc[:, list(FEATURE_COLUMNS)]
        target = dataset["target"]
        training_features, _ = train_test_split(
            features,
            target,
            test_size=0.25,
            random_state=42,
        )
        self.scaler = StandardScaler().fit(training_features)

    def predict(self, values: Sequence[float]) -> int:
        if len(values) != len(FEATURE_COLUMNS):
            raise ValueError(
                f"Expected {len(FEATURE_COLUMNS)} input values; got {len(values)}."
            )

        features = np.asarray(values, dtype=float).reshape(1, -1)
        scaled_features = self.scaler.transform(features)
        return int(self.model.predict(scaled_features)[0])
