import math

from flask import Flask, render_template, request

from heart_health.predictor import FEATURE_COLUMNS, HeartDiseasePredictor

VALUE_RANGES = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (50, 250),
    "chol": (50, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (50, 250),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (0, 2),
    "ca": (0, 3),
    "thal": (0, 3),
}
INTEGER_FIELDS = set(FEATURE_COLUMNS) - {"oldpeak"}


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    predictor = HeartDiseasePredictor()

    @app.route("/")
    def home():
        return render_template("main.html", error=None)

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            values = [float(request.form[name]) for name in FEATURE_COLUMNS]
        except (KeyError, ValueError):
            return (
                render_template(
                    "main.html",
                    error="Enter a valid value for every field.",
                ),
                400,
            )

        for name, value in zip(FEATURE_COLUMNS, values):
            minimum, maximum = VALUE_RANGES[name]
            if (
                not math.isfinite(value)
                or not minimum <= value <= maximum
                or (name in INTEGER_FIELDS and not value.is_integer())
            ):
                return (
                    render_template(
                        "main.html",
                        error=f"Enter a valid value for {name}.",
                    ),
                    400,
                )

        result = predictor.predict(values)
        return render_template("result.html", has_disease=bool(result))

    return app
