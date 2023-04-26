from fastapi.middleware.cors import CORSMiddleware
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# open api to all origins and platforms
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HouseFeatures(BaseModel):
    features: list


def load_model():
    '''this loads the model from the dir and return it to the predict function'''
    with open('models/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


def predict(features):
    '''this function takes the features and returns the prediction'''
    model = load_model()
    prediction = model.predict(features)
    return prediction


@app.get('/')
def preview():
    return {'hello world': 'House Price Prediction System API [v1].'}


@app.post('/predict')
def predict_house_price(house_features: HouseFeatures):
    features = np.array(house_features.features).reshape(1, -1)
    prediction = predict(features)
    return {'prediction': prediction[0]}
