from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from scipy import sparse
from sklearn.pipeline import Pipeline


def _to_dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else matrix


def extract_positive_class_shap(shap_values, n_features: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    if arr.ndim == 2 and arr.shape[1] == n_features:
        return arr
    raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")


def generate_shap_summary_plot(
    fitted_pipeline: Pipeline,
    X_test,
    output_path: str = "reports/figures/shap_summary.png",
) -> Path:
    """
    Generate SHAP summary plot for the test set and save as PNG.
    """
    preprocess = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    transformed = preprocess.transform(X_test)
    dense = _to_dense(transformed)

    col_transform = preprocess.named_steps["preprocess"]
    feature_names = list(col_transform.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dense)
    shap_values_pos = extract_positive_class_shap(shap_values, len(feature_names))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values_pos, dense, feature_names=feature_names, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def save_shap_explainer_bundle(
    fitted_pipeline: Pipeline,
    output_path: str = "models/shap_explainer.joblib",
) -> Path:
    """
    Persist SHAP explainer metadata bundle for downstream explainability workflows.
    """
    preprocess = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    col_transform = preprocess.named_steps["preprocess"]
    feature_names = list(col_transform.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    bundle = {
        "explainer": explainer,
        "feature_names": feature_names,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    return out
