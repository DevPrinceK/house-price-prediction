import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# Replace this with the actual path to your downloaded CSV file
data = pd.read_csv('data/data.csv')

# Preprocess the dataset
data = data.dropna(axis=1)
# override the city series randomly with one of the following in each dataframe: kent, seattle, redmond, kirkland
replacement_values = ['kent', 'seattle', 'redmond', 'kirkland']

# Replace the values randomly
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


# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# print the input features and the accuracy of the model
print(X_test)
print(model.score(X_test, y_test))

# print the lengh of the input features
print(len(X_test))
