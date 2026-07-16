# Weather Prediction with Machine Learning

A reproducible machine-learning project that predicts rainfall in millimetres from temperature, humidity, atmospheric pressure and wind speed. The project includes data exploration, model training, command-line prediction and visualisation.

## What the project does

- Loads and validates an included weather dataset
- Trains a `RandomForestRegressor`
- Evaluates the model using MAE and R²
- Saves the trained model for later predictions
- Accepts new weather values from the command line
- Produces a weather-trends chart

## Model results

Results from the included 240-row sample dataset using `random_state=42`:

| Metric | Result |
|---|---:|
| Mean Absolute Error | 3.14 mm |
| R² score | 0.442 |

These results are a learning baseline on a small demonstration dataset, not a production weather forecast.

## Screenshots

![Weather trends chart](weather_trends.png)

![Example rainfall prediction](prediction_result.png)

## Project structure

```text
Weather-Prediction-ML/
├── sample_weather.csv          # Demonstration dataset
├── train_model.py              # Training and evaluation pipeline
├── predict.py                  # Command-line rainfall prediction
├── visualize_data.py           # Dataset visualisation
├── weather_prediction.ipynb    # Exploration notebook
├── weather_model.joblib        # Trained model bundle
├── weather_trends.png          # Generated chart
├── prediction_result.png       # Example output
└── requirements.txt            # Python dependencies
```

## Installation

```bash
git clone https://github.com/yashrajadsul165/Weather-Prediction-ML.git
cd Weather-Prediction-ML
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Train and evaluate the model:

```bash
python train_model.py
```

Make a prediction:

```bash
python predict.py --temperature 28 --humidity 75 --pressure 1012 --wind-speed 10
```

Create the trends chart:

```bash
python visualize_data.py
```

Open the notebook:

```bash
jupyter notebook weather_prediction.ipynb
```

## Dataset fields

| Column | Meaning |
|---|---|
| `date` | Observation date |
| `temperature_c` | Temperature in °C |
| `humidity_pct` | Relative humidity in percent |
| `pressure_hpa` | Atmospheric pressure in hPa |
| `wind_speed_kmh` | Wind speed in km/h |
| `rainfall_mm` | Rainfall target in millimetres |

## Future improvements

- Train on a larger verified meteorological dataset
- Compare multiple regression models
- Use time-aware validation and feature engineering
- Add live weather API data
- Deploy an interactive Streamlit application

## Author

**Yashraj Adsul**

[LinkedIn](https://www.linkedin.com/in/yashraj-adsul-b69a8b2b4) · [Email](mailto:yashrajadsul165@gmail.com)

## License

This project is available under the [MIT License](LICENSE).
