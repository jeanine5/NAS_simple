import os

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

from nsga2 import *

def download_iris_dataset(url, filename):
    response = requests.get(url, verify=False)  # Bypass SSL verification
    with open(filename, 'wb') as file:
        file.write(response.content)

if not os.path.exists('iris.csv'):
    print("Downloading and saving the Iris dataset...")
    # Define the URL of the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    download_iris_dataset(url, 'iris.csv')
else:
    print("Iris dataset found, loading...")

# Step 1: Load the CSV file
df = pd.read_csv('iris.csv', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Step 2: Preprocess the dataset
# Convert class labels to integers
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Step 3: Split features and labels
X = df.drop('class', axis=1).values
y = df['class'].values

# Step 4: Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 5: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Step 7: Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Example usage
population_size = 100
generations = 5
crossover_factor = 0.9
mutation_factor = 0.1
max_hidden_layers = 5
max_hidden_size = 100

# Initialize NSG2 instance
nsga2 = NSG2(population_size, generations, crossover_factor, mutation_factor)

# Evolve the population
best_population = nsga2.evolve(train_loader, test_loader, max_hidden_layers, max_hidden_size)

# Print results
for i, arch in enumerate(best_population):
    print(f"Individual {i} - Non-dominated Rank: {arch.nondominated_rank}, Crowding Distance: {arch.crowding_distance}, Test Accuracy: {arch.acc_objective}")