import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import pydeck as pdk


def generate_inputs(cols, title=None):
    """Generate Streamlit inputs for a list of columns."""
    if title:
        st.write(title)
    for col in cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                input_data[col] = st.number_input(
                    f"{col} (min: {min_val}, max: {max_val})",
                    min_value=min_val,
                    max_value=max_val,
                    value=None if col in empty_cols else min_val
                )
            else:
                options = df[col].dropna().unique()
                input_data[col] = st.selectbox(f"{col}", options)
        else:
            st.warning(f"{col} is not a valid feature. Please check the feature list.")



st.image("images/header.png", use_container_width=True)
st.write("Please provide today's weather data to get the prediction for tomorrow's rainfall.")
# st.header("APP.py")

@st.cache_data
def load_data(file):
    path = "data/"
    df = pd.read_csv(path + file)
    return df

# Load  data
df_locations = load_data("locations.csv")  
df = load_data( "weatherAUS.csv")
weather_model = joblib.load("model/aussie_rain.joblib")

full_cols = weather_model['input_cols']
input_cols = weather_model['input_cols']
target_cols = weather_model['target_col']


# include all columns which include "9am"
input_9am = [col for col in df.columns if "9am" in col]
# include all columns which include "3pm"
input_3pm = [col for col in df.columns if "3pm" in col]
# combine all spsific cols together
drop_cols = ["RainToday", "Location"] + input_9am + input_3pm
# save the rest of the columns
input_rest = [col for col in full_cols if col not in drop_cols]


# create number input fields for each feature

st.write("Please enter the weather data for today:")
input_data = {}

# Make part of the value optional by default
# select the columns which have more than 10 000 missing values
empty_cols = df.isna().sum().sort_values(ascending=False) 
empty_cols = list(empty_cols[empty_cols > 10000].index)

#create the checkbox for RainToday
rain_today = st.checkbox("RainToday")
if rain_today:
    input_data["RainToday"] = input_data["RainToday"] = "Yes"
else:
    input_data["RainToday"] = "No"


# Generate input widgets
generate_inputs(input_rest)

# Create columns for time-based inputs
col1, col2 = st.columns(2)

with col1:
    generate_inputs(input_9am, "Please enter the weather data for 9am:")

with col2:
    generate_inputs(input_3pm, "Please enter the weather data for 3pm:")


# # select box for Locatoin colum 
st.write("Please select the location:")
locations = df["Location"].unique()
location = st.selectbox("Location", locations)   
input_data["Location"] = location


# Create map
base_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_locations,
    get_position="[longitude, latitude]",
    get_fill_color="[128, 128, 128, 140]",
    get_radius=30000,
    pickable=True
)

# selected location (bright yellow)
highlight_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_locations[df_locations["location"] == location],
    get_position="[longitude, latitude]",
    get_fill_color="[255, 234, 0, 255]",
    get_radius=50000,
    pickable=True
)

# Set the initial view of the map to center on selected location
if not df_locations[df_locations["location"] == location].empty:
    lat = df_locations[df_locations["location"] == location]["latitude"].values[0]
    lon = df_locations[df_locations["location"] == location]["longitude"].values[0]
else:
    lat, lon = -25.0, 135.0  # fallback center (Australia)

view_state = pdk.ViewState(
    latitude=lat,
    longitude=lon,
    zoom=3,
    pitch=0
)

# Combine the layers
st.pydeck_chart(pdk.Deck(
    initial_view_state=view_state,
    layers=[base_layer, highlight_layer],
    tooltip={"text": "{location}"}
))


col3, col4 = st.columns(2)

with col3:
    st.write("Click the button to predict tomorrow's rainfall:")
    if st.button("Predict"):
        # st.write("Data process started...")
        
        input_df = pd.DataFrame([input_data])

        # Reorder the columns to match the original order
        input_df = input_df.reindex(columns=full_cols)

        st.write("Input data loaded successfully")

        # 1. Impute missing values
        imputer = weather_model['imputer']
        input_df[weather_model['numeric_cols']] = imputer.transform(input_df[weather_model['numeric_cols']])
        st.write("Imputed missing values successfully")

        # 2. Scale numeric features
        scaler = weather_model['scaler']
        input_df[weather_model['numeric_cols']] = scaler.transform(input_df[weather_model['numeric_cols']])
        st.write("Scaled numeric features successfully")

        # 3. Encode categorical features
        encoder = weather_model['encoder']
        encoded_cats = encoder.transform(input_df[weather_model['categorical_cols']])
        encoded_cat_df = pd.DataFrame(encoded_cats, columns=weather_model['encoded_cols'])
        st.write("Encoded categorical features successfully")

        # 4. Combine numeric and encoded categorical features
        X = pd.concat([input_df[weather_model['numeric_cols']].reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)


        # 5. Predict
        model = weather_model['model']
        prediction = model.predict(X)
        prediction_proba = model.predict_proba(X)

        st.header(f"Tomorrow's rainfall prediction: {prediction[0]}" )
        st.write("Rainfall probability:", round(prediction_proba[0][1]* 100, 2), "%")

        # Show image and message in col4
        with col4:
            if prediction_proba[0][1] < 0.1:
                st.image("images/sun100.png", caption="No Rain", use_container_width=True)
                # st.write("No Rain")
            elif prediction_proba[0][1] >= 0.1 and prediction_proba[0][1] < 0.4:
                st.image("images/sunny.png", caption="Low Chance of Rain", use_container_width=True)
                # st.write("Low Chance of Rain")
            elif prediction_proba[0][1] >= 0.4 and prediction_proba[0][1] < 0.6: 
                st.image("images/rain50.png", caption="Medium Chance of Rain", use_container_width=True)
                # st.write("Medium Chance of Rain")
            elif prediction_proba[0][1] >= 0.6 and prediction_proba[0][1] < 0.8:
                st.image("images/rainy.png", caption="High Chance of Rain", use_container_width=True)
                # st.write("High Chance of Rain")
            elif prediction_proba[0][1] >= 0.8:
                st.image("images/thunder.png", caption="Very High Chance of Rain", use_container_width=True)
                # st.write("Very High Chance of Rain")

