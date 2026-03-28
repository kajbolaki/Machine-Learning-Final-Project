import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pipeline_config import MODEL_DIR


MODEL_PATH = MODEL_DIR / "crash_severity_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {MODEL_PATH}. Run training mode first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing metadata artifact: {METADATA_PATH}. Run training mode first."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _risk_band(probability: float, metadata: dict) -> str:
    low_max = metadata["risk_bands"]["low_max"]
    medium_max = metadata["risk_bands"]["medium_max"]
    if probability <= low_max:
        return "Low"
    if probability <= medium_max:
        return "Medium"
    return "High"


def _safe_dense(array_like):
    return array_like.toarray() if hasattr(array_like, "toarray") else np.asarray(array_like)


def top_contributors(model, input_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed = _safe_dense(preprocessor.transform(input_df))
    values = transformed[0]
    feature_names = preprocessor.get_feature_names_out()

    contributions = None
    if hasattr(estimator, "coef_"):
        contributions = values * estimator.coef_[0]
    elif hasattr(estimator, "feature_importances_"):
        contributions = values * estimator.feature_importances_

    if contributions is None:
        return []

    indices = np.argsort(np.abs(contributions))[::-1]
    top_items = []
    for idx in indices:
        score = float(contributions[idx])
        if abs(score) < 1e-9:
            continue
        top_items.append(
            {
                "feature": feature_names[idx],
                "impact_score": round(score, 5),
                "direction": "Risk Up" if score > 0 else "Risk Down",
            }
        )
        if len(top_items) >= top_n:
            break
    return top_items


def _pretty_feature_name(feature_name: str) -> str:
    pretty = feature_name.replace("num__", "").replace("cat__", "")
    return pretty.replace("_", " ").title()


def main() -> None:
    st.set_page_config(page_title="Crash Severity Predictor", layout="wide")
    st.title("Chicago Crash Severity Predictor")
    st.caption(
        "Predicts probability of severe injury/fatal outcome from crash conditions."
    )

    metadata = load_metadata()
    model = load_model()

    with st.form("prediction_form"):
        st.subheader("Crash Context Inputs")
        form_values = {}

        for feature in metadata["numeric_features"]:
            info = metadata["numeric_ranges"][feature]
            form_values[feature] = st.slider(
                feature.replace("_", " ").title(),
                min_value=float(info["min"]),
                max_value=float(info["max"]),
                value=float(info["default"]),
            )

        for feature in metadata["categorical_features"]:
            options = metadata["category_options"].get(feature) or ["UNKNOWN"]
            form_values[feature] = st.selectbox(
                feature.replace("_", " ").title(),
                options=options,
            )

        submitted = st.form_submit_button("Predict Severity Risk")

    if submitted:
        try:
            input_df = pd.DataFrame([form_values])
            probability = float(model.predict_proba(input_df)[0, 1])
            band = _risk_band(probability, metadata)
            threshold = float(metadata["selected_model"]["threshold"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Severe Crash Probability", f"{probability:.2%}")
            col2.metric("Risk Band", band)
            col3.metric("Decision Threshold", f"{threshold:.0%}")

            st.progress(int(probability * 100), text="Predicted severe-risk probability")

            contribs = top_contributors(model, input_df=input_df, top_n=5)
            st.subheader("Top Contributing Features")
            if contribs:
                contrib_df = pd.DataFrame(contribs)
                contrib_df["feature"] = contrib_df["feature"].map(_pretty_feature_name)

                impact_chart_df = contrib_df[["feature", "impact_score", "direction"]].sort_values(
                    "impact_score", ascending=True
                )
                impact_chart = (
                    alt.Chart(impact_chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("impact_score:Q", title="Impact Score"),
                        y=alt.Y("feature:N", sort=None, title="Feature"),
                        color=alt.Color(
                            "direction:N",
                            scale=alt.Scale(domain=["Risk Up", "Risk Down"], range=["#d62728", "#2ca02c"]),
                        ),
                        tooltip=["feature", "impact_score", "direction"],
                    )
                ).properties(height=280)
                st.altair_chart(impact_chart, use_container_width=True)
                st.dataframe(contrib_df, use_container_width=True)
            else:
                st.info("No contribution details available for this model.")
        except Exception as error:  # pragma: no cover
            st.error(f"Prediction failed: {error}")

    st.divider()
    st.subheader("Model Snapshot")
    metrics = metadata["test_metrics"]
    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    m2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    m3.metric("Recall (Severe)", f"{metrics['recall_severe']:.1%}")

    st.json(
        {
            "selected_model": metadata["selected_model"]["model_name"],
            "threshold": metadata["selected_model"]["threshold"],
            "test_metrics": metadata["test_metrics"],
        }
    )


if __name__ == "__main__":
    main()
