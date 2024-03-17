import pandas as pd

file_path = 'data.csv'

# Read the data from the CSV file
df = pd.read_csv(file_path)

# Discretize the four columns into short, medium, and long categories
df['Sepal length'] = pd.qcut(df['Sepal length'], q=[0, 0.33, 0.67, 1], labels=['Short', 'Medium', 'Long'])
df['Sepal width'] = pd.qcut(df['Sepal width'], q=[0, 0.33, 0.67, 1], labels=['Narrow', 'Medium', 'Wide'])
df['Petal length'] = pd.qcut(df['Petal length'], q=[0, 0.33, 0.67, 1], labels=['Short', 'Medium', 'Long'])
df['Petal width'] = pd.qcut(df['Petal width'], q=[0, 0.33, 0.67, 1], labels=['Thin', 'Medium', 'Thick'])

# Save the updated DataFrame to a new CSV file
output_file_path = 'discretized_data.csv'
df.to_csv(output_file_path, index=False)

# Display a message indicating the successful save
print(f"Discretized data saved to {output_file_path}")
