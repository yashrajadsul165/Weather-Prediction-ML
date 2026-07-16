from pathlib import Path
import argparse
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "weather_model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict rainfall in millimetres.")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--pressure", type=float, required=True)
    parser.add_argument("--wind-speed", type=float, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file is missing. Run `python train_model.py` first."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]

    sample = pd.DataFrame([{
        "temperature_c": args.temperature,
        "humidity_pct": args.humidity,
        "pressure_hpa": args.pressure,
        "wind_speed_kmh": args.wind_speed,
    }], columns=features)

    rainfall = max(0.0, float(model.predict(sample)[0]))
    print(f"Predicted rainfall: {rainfall:.2f} mm")

if __name__ == "__main__":
    main()
