import json
import joblib
import numpy as np
from pathlib import Path

def model_fn(model_dir):
    """Load the model from disk."""
    model_path = Path(model_dir) / "model.joblib"
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    """Parse input data."""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return np.array(input_data)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions."""
    predictions = model.predict(input_data)
    return predictions.tolist()


def output_fn(prediction, accept):
    """Format output response."""
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")