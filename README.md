# Forest Fire Prediction System

This project implements a machine learning-based Forest Fire Prediction System using various meteorological parameters. The system uses multiple machine learning algorithms to predict the likelihood and severity of forest fires.

## Features

- Multiple machine learning models:
  - Random Forest Regressor
  - Decision Tree Regressor
  - ANN-GBM (Artificial Neural Network with Gradient Boosting)
  - Extra Tree Regressor
- Meteorological parameter analysis
- Model performance comparison
- Feature importance visualization

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python forest_fire_prediction.py
```

## Project Structure

- `forest_fire_prediction.py`: Main script containing the implementation
- `requirements.txt`: List of Python dependencies
- `data/`: Directory for storing the dataset
- `README.md`: Project documentation

## Model Details

The system uses the following features for prediction:
- Temperature
- Relative Humidity
- Wind Speed
- Rain
- Fine Fuel Moisture Code (FFMC)
- Duff Moisture Code (DMC)
- Drought Code (DC)
- Initial Spread Index (ISI)
- Buildup Index (BUI)
- Fire Weather Index (FWI)

## Performance Metrics

The models are evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared Score
- Mean Absolute Error (MAE) 