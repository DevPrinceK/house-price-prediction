import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# Replace this with the actual path to your downloaded CSV file
data = pd.read_csv('data/data.csv')

# Preprocess the dataset
data = data.dropna(axis=1)

X = data.drop(columns=['date', 'price', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'street', 'statezip', 'country'])  # noqa

# One-hot encode categorical variables
X = pd.get_dummies(X)

y = data['price']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.values)
X_test = scaler.transform(X_test.values)

# Train a K-Nearest Neighbors Regressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Save the trained models to files
with open('models/knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)

with open('models/lr_model.pkl', 'wb') as file:
    pickle.dump(lr_model, file)

# save the scaler
with open('scalers/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# save X
with open('models/X_pickle.pkl', 'wb') as f:
    pickle.dump(X, f)

# Evaluate the models
knn_predictions = knn_model.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_predictions)
knn_score = knn_model.score(X_test, y_test)

lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)

# done
print("Model Training Completed!")
print("KNN MSE:", knn_mse)
print("KNN Score:", knn_score)
print("LR MSE:", lr_mse)

