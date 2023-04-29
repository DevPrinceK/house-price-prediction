import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# Replace this with the actual path to your downloaded CSV file
data = pd.read_csv('data/data.csv')

# Preprocess the dataset
data = data.dropna(axis=1)
replacement_values = ['kent', 'seattle', 'redmond', 'kirkland']

for i in range(len(data['city'])):
    data['city'] = random.choice(replacement_values)

X = data.drop(columns=['date', 'price', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'street', 'statezip', 'country'])  # noqa
y = data['price']

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a K-Nearest Neighbors Regressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Train a Gaussian Naïve Bayes model
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

# Save the trained models to files
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)

with open('gnb_model.pkl', 'wb') as file:
    pickle.dump(gnb_model, file)

# save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# save x to a file
with open('x.pkl', 'wb') as file:
    pickle.dump(X, file)

# Evaluate the models
knn_predictions = knn_model.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_predictions)
knn_score = knn_model.score(X_test, y_test)

gnb_predictions = gnb_model.predict(X_test)
gnb_mse = mean_squared_error(y_test, gnb_predictions)

# Sample input feature as a dictionary)
sample_input = {
    'bedrooms': 3,
    'bathrooms': 2.0,
    'sqft_living': 1800,
    'sqft_lot': 7200,
    'floors': 1,
    'yr_built': 1980,
    'yr_renovated': 0,
    'city': 'kirkland'
}

# Convert the dictionary to a Pandas DataFrame
sample_df = pd.DataFrame([sample_input])

# One-hot encode the city feature
sample_df = pd.get_dummies(sample_df).reindex(columns=X.columns, fill_value=0)

# Scale the input feature using the trained scaler
scaled_sample = scaler.transform(sample_df)

print("Scaled Sample Input feature:")
print(scaled_sample)

# Print the input features and the performance of the models
print(X_test)
print("KNN Score:", knn_score)
print("KNN Mean Squared Error:", knn_mse)
print("Gaussian Naïve Bayes Mean Squared Error:", gnb_mse)

# Print the length of the input features
print(len(X_test))
