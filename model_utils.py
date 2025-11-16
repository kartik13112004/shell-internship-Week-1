# model_utils.py
import os
import joblib
import numpy as np

# Cache model
_loaded = None

def load_model():
    """
    Loads the EV range prediction model from the project folder.
    Looks for any .joblib or .pkl file automatically.
    """
    global _loaded

    if _loaded is not None:
        return _loaded

    script_dir = os.path.dirname(__file__)

    # Try common model names
    candidates = [
        "ev_range_model.joblib",
        "ev_range_model (1).joblib",
        "model.joblib",
        "model.pkl"
    ]

    for c in candidates:
        model_path = os.path.join(script_dir, c)
        if os.path.exists(model_path):
            _loaded = joblib.load(model_path)
            print(f"Loaded model: {c}")
            return _loaded

    # Search for any model file in folder
    for f in os.listdir(script_dir):
        if f.endswith(".joblib") or f.endswith(".pkl"):
            model_path = os.path.join(script_dir, f)
            _loaded = joblib.load(model_path)
            print(f"Loaded model: {f}")
            return _loaded

    raise FileNotFoundError("❌ No model file found in project folder.")

def predict_range(battery_kwh: float, eff_km_per_kwh: float) -> float:
    """
    Predicts EV range based on battery size (kWh) and efficiency (km/kWh).
    """
    model = load_model()

    # Create a dummy dataframe with all required features
    # The model expects 13 features, but we only have battery and efficiency
    # We'll use reasonable defaults for the missing features
    import pandas as pd

    data = {
        'AccelSec': [5.0],  # Default acceleration time
        'TopSpeed_KmH': [140.0],  # Default top speed
        'Efficiency_WhKm': [float(battery_kwh) / float(eff_km_per_kwh) * 1000],  # Convert to Wh/km
        'Seats': [5],  # Default seats
        'PriceEuro': [30000],  # Default price
        'Brand': ['Generic'],
        'Model': ['EV'],
        'FastCharge_KmH': ['No'],  # Default fast charge
        'RapidCharge': ['No'],
        'PowerTrain': ['Electric'],
        'PlugType': ['Type 2 CCS'],
        'BodyStyle': ['Hatchback'],
        'Segment': ['B']
    }

    X = pd.DataFrame(data)
    pred = model.predict(X)
    return float(pred[0])
