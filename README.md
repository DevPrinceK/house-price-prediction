# House Price Prediction Model
This is a machine learning model that predicts the price of a house based on various features such as the number of bedrooms, bathrooms, square footage, and location. <br>
The two models used are: <br>
* KNN Regressor
* Linear Regression


# Table of Contents
**Getting Started** <br>
**Training** <br>
**Running** <br>
**EndPoints** <br>
**Contributing** <br>
**Built With** <br>
**Links**

# Getting Started
To get started with the house price prediction model, you will need to have Python 3 installed on your system. Once you have Python 3 installed, you can clone this repository and install the required dependencies:
>git clone https://github.com/DevPrinceK/house-price-prediction-model.git <br>
>cd house-price-prediction-model <br>
>pip install -r requirements.txt 


# Training
Run the following command to train and save the model. 
You can also train the model on your own data by modifying the knn_lr.py script and provide your own dataset. The dataset should be in CSV format with the following columns: bedrooms, bathrooms, sqft_living, sqft_lot, floors, yr_built, yr_renovated, and price. Once you have modified the script to load your dataset, you can run it to train the model:
>python knn_lr.py

# Run 
Now you can run the fastapi app to expose the model via an api.
> uvicorn main:app --reload

# EndPoints
These are the endpoints you can use.
## > /
The home endpoint. This is a dummy endpoint that just tells you that the api is functional.
** No data required
## > cities
The cities endpoint fetches all the unique cities in our dataset. This is useful for implementing something like a dropdownlist of all cities supported by your model.
**It is a post request**
### Request Data
{
    "bedrooms": float,
    "bathrooms": float,
    "sqft_living": float,
    "sqft_lot": float,
    "floors": float,
    "yr_built": float,
    "yr_renovated": float,
    "city": str
}

# Contribution
Contributions to the house price prediction model are welcome. If you find a bug or have a suggestion for improvement, please open an issue or submit a pull request.


# Built With
**Python 3** <br>
**scikit-learn** <br>
**pandas** <br>
**numpy** <br>

# Links
You can find [The React Frontend here](https://github.com/Desmondgoldsmith/House_price_prediction_project) <br>
You can find [The Datasets here](https://www.kaggle.com/datasets/shree1992/housedata/versions/2?resource=download) <br>
You can find [Scikit-learn documentation here](https://scikit-learn.org/0.18/_downloads/scikit-learn-docs.pdf) <br>
You can find [scikit-learn linear regression here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) <br>
You can find [scikit-learn knn here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) <br>


