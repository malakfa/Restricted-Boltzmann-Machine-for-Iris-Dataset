import pandas as pd

# Read the dataset
data = pd.read_csv('discretized_data.csv')

# One-hot encode the 'Species' column
one_hot_encoded_species = pd.get_dummies(data['Species'], prefix='Species')

# One-hot encode the 'Petal width', 'Petal length', 'Sepal width', and 'Sepal length' columns
one_hot_encoded_petal_width = pd.get_dummies(data['Petal width'], prefix='Petal width')
one_hot_encoded_petal_length = pd.get_dummies(data['Petal length'], prefix='Petal length')
one_hot_encoded_sepal_width = pd.get_dummies(data['Sepal width'], prefix='Sepal width')
one_hot_encoded_sepal_length = pd.get_dummies(data['Sepal length'], prefix='Sepal Length')

# Concatenate the one-hot encoded columns with the original dataframe
data_encoded = pd.concat([data, one_hot_encoded_species, one_hot_encoded_petal_width, 
                         one_hot_encoded_petal_length, one_hot_encoded_sepal_width, 
                         one_hot_encoded_sepal_length], axis=1)

# Drop the original 'Species', 'Petal width', 'Petal length', 'Sepal width', and 'Sepal length' columns
data_encoded = data_encoded.drop(['Species', 'Petal width', 'Petal length', 'Sepal width', 'Sepal length'], axis=1)

# Replace False with 0 and True with 1
data_encoded.replace({False: 0, True: 1}, inplace=True)

# Save the resulting dataframe to a new CSV file
data_encoded.to_csv('binarize_data.csv', index=False)


