import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()


class HouseFeatures(BaseModel):
    features: list

# this loads the model from the dir and return it to the predict function
def load_model():
    with open('models/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


# this function takes the features and returns the prediction
def predict(features):
    model = load_model()
    prediction = model.predict(features)
    return prediction


@app.post('/predict')
def predict_house_price(house_features: HouseFeatures):
    features = np.array(house_features.features).reshape(1, -1)
    prediction = predict(features)
    return {'prediction': prediction[0]}
