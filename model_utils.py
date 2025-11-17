# model_utils.py
import importlib
import os
import sys
import traceback
from joblib import load as joblib_load
import joblib
import pandas as pd
import numpy as np

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

# ---------- model loading ----------
def load_model(model_path="ev_range_model.joblib"):
    """
    Load the joblib model at model_path. If unpickling fails because of missing
    classes from sklearn, this function attempts compatibility fallbacks first.
    Returns the loaded model object.
    """
    # Ensure compatibility shims are present before importing/loading
    _ensure_sklearn_compatibility()

    # Resolve absolute path (relative to repository root)
    base_dir = os.path.dirname(__file__)
    abs_path = os.path.join(base_dir, model_path) if not os.path.isabs(model_path) else model_path

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model file not found at: {abs_path}")

    try:
        model = joblib_load(abs_path)
        return model
    except Exception as ex:
        tb = traceback.format_exc()
        # Provide a helpful error for logs
        raise RuntimeError(
            "Failed to load model via joblib.load(). See nested exception and logs.\n\n"
            f"Original exception:\n{ex}\n\nTraceback:\n{tb}"
        ) from ex

# ---------- feature-list helpers ----------
def _load_feature_list():
    """Load feature order if present (feature_list.pkl or .joblib). Returns list or None."""
    base = os.path.dirname(__file__)
    for name in ("feature_list.pkl", "feature_list.joblib", "feature_list.pkl.joblib"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
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

    # If a feature order list exists, ensure columns match that order
    feature_list = _load_feature_list()
    if feature_list:
        # Add missing features as NaN and drop extra columns
        for feat in feature_list:
            if feat not in df.columns:
                df[feat] = np.nan
        df = df.loc[:, feature_list]

    return df

# ---------- prediction API ----------
def predict_range(model=None, input_dict=None, input_df=None):
    """
    Predict EV range.
    - model: optional; sklearn pipeline or estimator. If None, will call load_model().
    - input_dict: dict of feature_name: value for single sample.
    - input_df: pandas.DataFrame for multiple samples.
    Returns: numpy array of predictions (shape: (n_samples,)).
    """
    # Lazy-load model if not provided
    if model is None:
        model = load_model()

    # Prepare DataFrame
    df = _prepare_input_df(input_dict=input_dict, input_df=input_df)

    # Use common prediction methods
    if hasattr(model, "predict"):
        preds = model.predict(df)
    elif hasattr(model, "predict_proba"):
        # fallback: if classification, choose probability of positive class
        probs = model.predict_proba(df)
        # If binary, take column 1; else take argmax expected use-case is regression so this is rare.
        preds = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    else:
        raise AttributeError("Model has no predict/predict_proba method.")

    return np.asarray(preds)
