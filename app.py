from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "weather_model.joblib"

INPUTS = {
    "temperature_c": {
        "label": "Temperature (°C)",
        "min": -10.0,
        "max": 55.0,
        "default": 28.0,
        "step": 0.1,
    },
    "humidity_pct": {
        "label": "Humidity (%)",
        "min": 0.0,
        "max": 100.0,
        "default": 75.0,
        "step": 0.1,
    },
    "pressure_hpa": {
        "label": "Atmospheric pressure (hPa)",
        "min": 850.0,
        "max": 1100.0,
        "default": 1012.0,
        "step": 0.1,
    },
    "wind_speed_kmh": {
        "label": "Wind speed (km/h)",
        "min": 0.0,
        "max": 200.0,
        "default": 10.0,
        "step": 0.1,
    },
}


@st.cache_resource
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model is missing. Run python train_model.py first."
        )
    return joblib.load(MODEL_PATH)


def predict_rainfall(bundle: dict, values: dict[str, float]) -> float:
    sample = pd.DataFrame([values], columns=bundle["features"])
    return max(0.0, float(bundle["model"].predict(sample)[0]))


def main() -> None:
    st.set_page_config(
        page_title="Rainfall Prediction",
        page_icon="🌦️",
        layout="wide",
    )

    st.title("🌦️ Weather Prediction with Machine Learning")
    st.write(
        "Adjust the weather conditions below to estimate rainfall using a "
        "Random Forest regression model."
    )

    try:
        bundle = load_model_bundle()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    values: dict[str, float] = {}
    left, right = st.columns(2)
    for index, (feature, config) in enumerate(INPUTS.items()):
        column = left if index % 2 == 0 else right
        with column:
            values[feature] = st.slider(
                config["label"],
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
            )

    if st.button("Predict rainfall", type="primary", use_container_width=True):
        rainfall = predict_rainfall(bundle, values)
        st.metric("Estimated rainfall", f"{rainfall:.2f} mm")
        if rainfall < 0.1:
            st.success("The model estimates little or no rainfall for these inputs.")
        elif rainfall < 10:
            st.info("The model estimates light to moderate rainfall.")
        else:
            st.warning("The model estimates heavier rainfall.")

    st.divider()
    st.subheader("Model information")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Algorithm", "Random Forest")
    metric_2.metric("Test MAE", "3.14 mm")
    metric_3.metric("Test R²", "0.442")

    st.warning(
        "Educational demonstration only. This model was trained on a small "
        "240-row sample and must not be used for safety-critical forecasts."
    )
    st.caption("See MODEL_CARD.md for intended use, limitations and next steps.")


if __name__ == "__main__":
    main()

