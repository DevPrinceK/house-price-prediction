from fastapi.middleware.cors import CORSMiddleware
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

app = FastAPI()

# open api to all origins and platforms
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class House(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    yr_built: float
    yr_renovated: float
    city: str


def load_models():
    '''this loads the model from the dir and return it to the predict function'''
    with open('models/lr_model.pkl', 'rb') as file:
        lr_model = pickle.load(file)

    with open('models/knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)

    return lr_model, knn_model


def load_scaler():
    '''This loads the scaler used to scale the model - for scaling the input features'''
    with open('scalers/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler


def scale_input_features(input_features: House):
    '''this function scales the input features'''
    # Convert the dictionary to a Pandas DataFrame
    df = pd.DataFrame([input_features])

    # Load X_pickle.pkl
    with open('models/X_pickle.pkl', 'rb') as f:
        X = pickle.load(f)

    # One-hot encode the input features
    df = pd.get_dummies(df)

    # Add missing columns with 0 as the default value
    for col in X.columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure the order of columns in df matches X
    df = df[X.columns]

    # Scale the input features using the trained scaler
    scaler = load_scaler()
    scaled_df = scaler.transform(df)
    return scaled_df


def predict(features: House):
    '''this function takes the features and returns the prediction'''
    lr_model, knn_model = load_models()
    scaled_features = scale_input_features(features)
    lr_prediction = lr_model.predict(scaled_features)
    knn_prediction = knn_model.predict(scaled_features)
    return lr_prediction, knn_prediction


@app.get('/')
def preview():
    '''this function is just to preview the api - ensure that api is working'''
    return {'Hello World': 'House Price Prediction System API [v1].'}


@app.get('/cities')
def get_cities():
    '''this function returns the unique values of the city column'''
    try:
        # provide the full path to the data file
        df = pd.read_csv('data/data.csv')

        # check if the DataFrame is empty
        if df.empty:
            return {'Error': 'Data file is empty.'}

        # remove all rows with missing values
        df = df.dropna(axis=1)

        # get unique values from column 'column_name'
        cities = df['city'].unique()
    except Exception as err:
        import traceback
        traceback.print_exc()
        return {'Error': str(err)}
    else:
        return {"cities": cities.tolist()}



@app.post('/predict')
def predict_house_price(house_features: House):
    '''this function takes the features and returns the prediction'''
    input_features = {
        'bedrooms': house_features.bedrooms,
        'bathrooms': house_features.bathrooms,
        'sqft_living': house_features.sqft_living,
        'sqft_lot': house_features.sqft_lot,
        'floors': house_features.floors,
        'yr_built': house_features.yr_built,
        'yr_renovated': house_features.yr_renovated,
        'city': house_features.city
    }

    lr_prediction, knn_prediction = predict(input_features)

    return {
        'prediction': [
            {"LR": lr_prediction[0]},
            {"KNN": knn_prediction[0]},
            {"AVG": (lr_prediction[0]+knn_prediction[0])/2},
        ]
    }
