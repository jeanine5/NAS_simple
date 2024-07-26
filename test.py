import pandas as pd
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from nsga2 import NSG2
import os

# Load the dataset
curr_dir = os.getcwd()
data = pd.read_csv(curr_dir + '/Prevalences.csv')

# Convert the 'Culture' column to categorical type
data.Culture = data.Culture.astype('category')

# Extract features and target variables
X = data.iloc[:, 7:].values
y = data.iloc[:, 3].values

# Convert target to categorical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the datasets to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create PyTorch DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Example usage
population_size = 100
generations = 10
crossover_factor = 0.9
mutation_factor = 0.1
max_hidden_layers = 5
max_hidden_size = 100

# Initialize NSG2 instance (assuming nsga2.py defines this class)
nsga2 = NSG2(population_size, generations, crossover_factor, mutation_factor)

# Evolve the population
best_population = nsga2.evolve(train_loader, test_loader, max_hidden_layers, max_hidden_size)

# Print results
for i, arch in enumerate(best_population):
    train_acc = arch.train_model(train_loader)
    test_acc = arch.evaluate_accuracy(test_loader)
    print(f"Individual {i} - Non-dominated Rank: {arch.nondominated_rank}, Crowding Distance: {arch.crowding_distance}, Test Accuracy: {test_acc}")
    print(f"Hidden Size: {arch.hidden_sizes}, Activation: {arch.activation}")
    print("")
