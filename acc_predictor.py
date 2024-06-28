"""

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from architectures import NeuralArchitecture


class RegressionPredictor:
    def __init__(self, csv_filename='Training_CSV'):
        self.df = pd.read_csv(csv_filename)

        # Convert activation_function to categorical codes
        if self.df['activation_function'].dtype == 'object':
            self.df['activation_function'] = self.df['activation_function'].astype('category').cat.codes

        self.X = self.df[['num_hidden_layers', 'max_hidden_size', 'min_hidden_size', 'activation_function',
                          'train_accuracy']]

        self.y_test_acc = self.df['test_accuracy']

        self.X_train, self.X_test, self.y_acc_train, self.y_acc_test = (
            train_test_split(self.X, self.y_test_acc, test_size=0.2, random_state=42))

        self.model_acc = DecisionTreeRegressor()

    def train_models(self):
        """
        Train the linear regression model after preprocessing the data
        """
        self.model_acc.fit(self.X_train, self.y_acc_train)

    def evaluate_models(self):
        """
        Evaluate the performance of the regression model on the test set
        """
        y_acc_pred = self.model_acc.predict(self.X_test)
        mean_acc_pred = y_acc_pred.mean()
        return mean_acc_pred

    def predict_performance(self, new_architecture: NeuralArchitecture):
        """
        Make predictions for the performance of a new architecture
        """

        # Convert activation_function to categorical code if necessary
        activation_function = str(new_architecture.activation).split("'")[1].split(".")[-1]
        activation_code = pd.Series([activation_function]).astype('category').cat.codes[0]

        # Prepare input data for prediction
        new_data = pd.DataFrame([{
            'num_hidden_layers': len(new_architecture.hidden_layers),
            'max_hidden_size': max(new_architecture.hidden_sizes),
            'min_hidden_size': min(new_architecture.hidden_sizes),
            'activation_function': activation_code,
            'train_accuracy': new_architecture.train_acc
        }])

        # Use trained regression models to predict performance metrics
        acc_pred = self.model_acc.predict(new_data)

        # return acc_pred[0]

        return acc_pred.mean()
