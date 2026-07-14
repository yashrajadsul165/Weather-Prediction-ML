from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/sample_weather.csv")
OUTPUT_PATH = Path("screenshots/weather_trends.png")

def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["temperature_c"], label="Temperature (°C)")
    plt.plot(df["date"], df["humidity_pct"], label="Humidity (%)", alpha=0.75)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Weather trends")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    print(f"Chart saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
