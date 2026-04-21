import json
import joblib
import numpy as np
import os

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.joblib")
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)
    data = np.array(data)
    preds = model.predict(data)
    return preds.tolist()