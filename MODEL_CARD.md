# Model Card: Rainfall Prediction Baseline

## Overview

This repository contains an educational rainfall-regression baseline built with a RandomForestRegressor. It estimates rainfall in millimetres from temperature, humidity, atmospheric pressure and wind speed.

## Intended use

- Learning the end-to-end machine-learning workflow
- Demonstrating data validation, model training and evaluation
- Comparing command-line and interactive prediction interfaces
- Supporting portfolio discussion and further experimentation

## Not intended for

- Official meteorological forecasting
- Emergency, agricultural, travel or public-safety decisions
- Predictions outside the demonstrated feature ranges
- Production deployment without verified real-world data and monitoring

## Training data

The included dataset contains 240 demonstration observations. It is intentionally small and is provided only so the complete workflow can run locally.

## Evaluation

| Metric | Test result |
|---|---:|
| Mean Absolute Error | 3.14 mm |
| R² score | 0.442 |

The fixed train/test split uses random_state=42. The results represent a learning baseline, not a claim of operational forecasting accuracy.

## Limitations

- Small demonstration dataset
- No geographic location or seasonal lag features
- Random train/test split rather than time-aware validation
- Limited feature set and no uncertainty interval
- No comparison against established forecasting baselines

## Recommended improvements

1. Replace the demonstration data with a verified meteorological dataset.
2. Add lagged rainfall, seasonality and location features.
3. Use time-series cross-validation and compare multiple algorithms.
4. Report uncertainty, monitor drift and document data provenance.
5. Deploy only after domain review and robust validation.

