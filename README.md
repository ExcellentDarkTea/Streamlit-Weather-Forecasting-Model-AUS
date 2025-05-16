# [Streamlit Weather Prediction App](https://excellentdarktea-streamlit-weather-forecasting-model-app-kngdng.streamlit.app/)



This project is a Streamlit web application for predicting the likelihood of rainfall in Australia for the next day, based on weather data. The app uses a trained machine learning model and provides an interactive interface for users to input weather features and visualize predictions.

## Features

- **Interactive Input:** Enter weather data for today.
- **Location Selection:** Choose from Australian locations; see your selection on a map.
- **Rain Prediction:** Get a prediction for tomorrow's rainfall and the probability.
- **Preprocessing:** The app automatically handles missing values, scaling, and categorical encoding to match the model's requirements.

## Project Structure

```
streamlit-weather/
│
├── app.py                  # Main Streamlit app
├── model/
│   └── aussie_rain.joblib  # Trained model and preprocessors
├── data/
│   ├── weatherAUS.csv      # Weather data
│   └── locations.csv       # Location coordinates
├── images/                 # Weather icons for predictions
└── README.md
```

## Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/yourusername/streamlit-weather.git
cd streamlit-weather
```

### 2. Install dependencies

It is recommended to use a virtual environment.

```sh
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the main dependencies:

```sh
pip install streamlit pandas numpy scikit-learn pydeck joblib
```

### 3. Run the app

```sh
streamlit run app.py
```

### 4. Open in your browser

You can use the [link](https://excellentdarktea-streamlit-weather-forecasting-model-app-kngdng.streamlit.app) to use the app.

## Usage

1. Enter today's weather data in the provided fields.
2. Select your location from the dropdown.
3. Click the **Predict** button.
4. View the prediction and probability.

## Model & Data

- The model is stored in `model/aussie_rain.joblib` and includes the trained estimator, imputer, scaler, and encoder.
- Weather data and location coordinates are in the `data/` folder.

## Customization

- To use your own model, retrain and save it as a joblib dictionary with the same keys as used in the app.

