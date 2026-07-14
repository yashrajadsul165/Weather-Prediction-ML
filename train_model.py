from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/sample_weather.csv")
MODEL_PATH = Path("models/weather_model.joblib")

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    features = ["temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_kmh"]
    target = "rainfall_mm"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f} mm")
    print(f"R² score: {r2_score(y_test, predictions):.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
