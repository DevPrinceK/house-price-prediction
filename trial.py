import pandas as pd

# read csv file
df = pd.read_csv('data/data.csv')

# get unique values from column 'column_name'
unique_values = df['city'].unique()

# print unique values and length
print("Unique values:", unique_values)
print("Number of unique values:", len(unique_values))
