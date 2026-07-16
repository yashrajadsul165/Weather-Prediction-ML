from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "sample_weather.csv"
MODEL_PATH = PROJECT_ROOT / "weather_model.joblib"
FEATURES = ["temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_kmh"]
TARGET = "rainfall_mm"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    clean_df = df.dropna(subset=required_columns)
    if len(clean_df) < 10:
        raise ValueError("At least 10 complete rows are required to train the model.")

    X = clean_df[FEATURES]
    y = clean_df[TARGET]

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
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
