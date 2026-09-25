# ==========================================
# NASA EXOPLANET PREDICTOR
# ==========================================

import os
import joblib
import pandas as pd

from src.config import FEATURES


# ------------------------------------------
# Load trained models
# ------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "exoplanet_best_models.pkl"
)

artifacts = joblib.load(MODEL_PATH)


binary_model = artifacts["binary_model"]
multiclass_model = artifacts["multiclass_model"]
multiclass_imputer = artifacts["multiclass_imputer"]


# ------------------------------------------
# Prepare input data
# ------------------------------------------

def prepare_input(input_data):

    data = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    return data


# ------------------------------------------
# Binary prediction
# ------------------------------------------

def predict_binary(input_data):

    data = prepare_input(input_data)

    prediction = binary_model.predict(data)[0]

    if prediction == 1:
        result = "Planet"
    else:
        result = "False Signal"

    return result


# ------------------------------------------
# Multiclass prediction
# ------------------------------------------

def predict_multiclass(input_data):

    data = prepare_input(input_data)

    # Apply the same median imputation
    # used during multiclass model training
    data_imputed = multiclass_imputer.transform(data)

    prediction = multiclass_model.predict(
        data_imputed
    )[0]

    return prediction

