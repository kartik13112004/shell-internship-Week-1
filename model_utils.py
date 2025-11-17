# model_utils.py
import importlib
import os
import sys
import traceback
from joblib import load as joblib_load

# --- Compatibility monkeypatches for known missing sklearn private names ---
# This must run BEFORE joblib.load() is called.
def _ensure_sklearn_compatibility():
    module_name = "sklearn.compose._column_transformer"
    try:
        ct = importlib.import_module(module_name)
    except Exception:
        ct = None

    if ct is not None:
        # Fallback: if the old private name is missing, provide a simple shim.
        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                """Compatibility fallback for old pickles that referenced _RemainderColsList"""
                pass
            setattr(ct, "_RemainderColsList", _RemainderColsList)

    # add any other compatibility fallbacks here if needed:
    # e.g. for very old pipelines you might need to create aliases for other names.

# --- Model loader with helpful logging ---
def load_model(model_path="ev_range_model.joblib"):
    """
    Load the joblib model at model_path. If unpickling fails because of missing
    classes from sklearn, this function attempts compatibility fallbacks first.
    """
    # Ensure compatibility shims are present
    _ensure_sklearn_compatibility()

    # Resolve absolute path (if the model sits in a folder, adjust accordingly)
    abs_path = os.path.join(os.getcwd(), model_path) if not os.path.isabs(model_path) else model_path

    if not os.path.exists(abs_path):
        # helpful error for debugging in Streamlit logs
        raise FileNotFoundError(f"Model file not found at: {abs_path}")

    try:
        # Use joblib to load the model
        model = joblib_load(abs_path)
        return model
    except Exception as ex:
        # Print stacktrace to the logs so Streamlit's log viewer shows the full detail
        tb = traceback.format_exc()
        # Re-raise a clear error so you see something in the app logs
        raise RuntimeError(
            "Failed to load model via joblib.load(). See nested exception and logs.\n\n"
            f"Original exception:\n{ex}\n\nTraceback:\n{tb}"
        ) from ex
# ---------- Add this to model_utils.py ----------
import os
import joblib
import pandas as pd
import numpy as np

def _load_feature_list():
    """Load feature order if present (feature_list.pkl or feature_list.joblib)."""
    base = os.path.dirname(__file__)
    for name in ("feature_list.pkl", "feature_list.joblib", "feature_list.pkl.joblib"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                # fallback to pickle if needed
                import pickle
                with open(path, "rb") as f:
                    return pickle.load(f)
    return None

def _prepare_input_df(input_dict=None, input_df=None):
    """Return a DataFrame ready to feed the model."""
    if input_df is not None:
        df = input_df.copy()
    elif input_dict is not None:
        df = pd.DataFrame([input_dict])
    else:
        raise ValueError("Provide input_dict or input_df")

    # reorder columns to the feature list if available
    feature_list = _load_feature_list()
    if feature_list:
        # ensure all features are present; missing set to NaN
        for feat in feature_list:
            if feat not in df.columns:
                df[feat] = np.nan
        df = df.loc[:, feature_list]
    return df

def predict_range(model=None, input_dict=None, input_df=None):
    """
    Predict EV range using model.
    - model: a loaded model object. If None, tries to call load_model() from this module.
    - input_dict: dict of feature_name: value for single sample.
    - input_df: pandas.DataFrame for multiple samples.
    Returns: numpy array of predictions.
    """
    # lazy-load model if not provided
    if model is None:
        # avoid circular import; import local load_model function
        try:
            from .model_utils import load_model as _lm  # for package-style import
        except Exception:
            try:
                from model_utils import load_model as _lm  # fallback for direct script import
            except Exception:
                _lm = None
        if _lm is None:
            raise RuntimeError("No model provided and load_model() not found in model_utils.")
        model = _lm()

    df = _prepare_input_df(input_dict=input_dict, input_df=input_df)

    # If model is a sklearn pipeline or has .predict
    if hasattr(model, "predict"):
        preds = model.predict(df)
    else:
        # try 'predict_proba' fallback or attribute 'predict_range' if custom
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(df)[:, 1]  # this is guessy; adapt if needed
        elif hasattr(model, "predict_range"):
            preds = model.predict_range(df)
        else:
            raise AttributeError("Model has no predict/predict_proba/predict_range method.")
    return np.array(preds)
    import pandas as pd
import numpy as np

def predict_range(model, input_dict):
    """
    Takes a dictionary of user inputs and returns model prediction.
    """
    # Convert the dictionary into a DataFrame
    df = pd.DataFrame([input_dict])

    # Predict using the model (sklearn pipeline)
    prediction = model.predict(df)

    # Return single value
    return prediction[0]

# ---------- End snippet ----------
