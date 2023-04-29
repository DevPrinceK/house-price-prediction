# House Price Prediction Model
This is a machine learning model that predicts the price of a house based on various features such as the number of bedrooms, bathrooms, square footage, and location

# Table of Contents
-**Getting Started**
-**Usage**
-**Training**
-**Built With**
-**Contributing**
-**License**
-**Acknowledgments**

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


# Built With
-**Python 3**
-**scikit-learn**
-**pandas**
-**numpy**


