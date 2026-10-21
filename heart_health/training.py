import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from heart_health.predictor import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH


def train_model() -> None:
    dataset = pd.read_csv(DATA_PATH)
    features = dataset.loc[:, list(FEATURE_COLUMNS)]
    target = dataset["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)

    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("Classification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)


if __name__ == "__main__":
    train_model()
