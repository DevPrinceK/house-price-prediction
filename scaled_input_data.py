import pandas as pd

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

print("Scaled sample input feature:")
print(scaled_sample)
