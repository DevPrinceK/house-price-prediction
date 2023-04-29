from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = FastAPI()


class House(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    yr_built: float
    yr_renovated: float
    city: str


@app.post("/predict")
async def predict_price(house: House):
    # Load the dataset
    df = pd.read_csv("housing_data.csv")

    # Drop the unnecessary columns
    df = df.drop(['date', 'price', 'waterfront', 'view', 'condition',
                 'sqft_above', 'sqft_basement', 'street', 'statezip', 'country'], axis=1)

    # Split the dataset into features and target variable
    X = df.drop('sqft_living', axis=1)
    y = df['sqft_living']

    # One-hot encode the categorical variable 'city'
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
    X = ct.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # Scale the features using StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train the KNN Regressor model
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Train the Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predict the price using both models
    knn_price = knn.predict(sc.transform(ct.transform(house.dict())))
    lr_price = lr.predict(sc.transform(ct.transform(house.dict())))

    return {"knn_price": knn_price[0], "lr_price": lr_price[0]}
