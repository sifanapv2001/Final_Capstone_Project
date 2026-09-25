# ==========================================
# NASA EXOPLANET DETECTION APPLICATION
# ==========================================

from flask import Flask, render_template, request

from src.predictor import (
    predict_binary,
    predict_multiclass
)

from src.config import FEATURES


# ------------------------------------------
# Create Flask application
# ------------------------------------------

app = Flask(__name__)


# ------------------------------------------
# Home page
# ------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        features=FEATURES
    )


# ------------------------------------------
# Prediction
# ------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get prediction type
        prediction_type = request.form["prediction_type"]

        # Collect input values
        input_data = []

        for feature in FEATURES:

            value = float(
                request.form[feature]
            )

            input_data.append(value)

        # ----------------------------------
        # Binary classification
        # ----------------------------------

        if prediction_type == "binary":

            result = predict_binary(
                input_data
            )

            title = "Binary Classification Result"

        # ----------------------------------
        # Multiclass classification
        # ----------------------------------

        elif prediction_type == "multiclass":

            result = predict_multiclass(
                input_data
            )

            title = "Multiclass Classification Result"

        else:

            result = "Invalid prediction type."
            title = "Prediction Error"

        return render_template(
            "index.html",
            features=FEATURES,
            result=result,
            title=title
        )

    except ValueError:

        return render_template(
            "index.html",
            features=FEATURES,
            result="Please enter valid numerical values.",
            title="Input Error"
        )

    except Exception as e:

        return render_template(
            "index.html",
            features=FEATURES,
            result=f"Prediction error: {str(e)}",
            title="Error"
        )


# ------------------------------------------
# Run application
# ------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )