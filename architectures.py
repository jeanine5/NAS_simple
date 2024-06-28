import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from thop import profile

device = "cpu" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)
print(f"Using device: {device}")


class NeuralNetwork(nn.Module):
    """
    A fully connected neural network with a variable number of hidden layers
    """

    def __init__(self, input_size, hidden_sizes, activation):
        super().__init__()

        self.input_size = input_size
        self.activation = activation
        self.hidden_sizes = hidden_sizes

        self.hidden_layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            if issubclass(activation, nn.Module):
                self.hidden_layers.append(activation())
            else:
                self.hidden_layers.append(activation)
            input_size = hidden_size

        self.output_layer = nn.Sequential(
            nn.Linear(input_size, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # batch_size = len(x)
        # x = x.view(batch_size, self.input_size).to(device)
        # for layer in self.hidden_layers:
        #     x = layer(x)
        # x = self.output_layer(x)
        # return x

        batch_size = len(x)
        x = x.view(batch_size, self.input_size).to(device)

        for layer in self.hidden_layers:
            h = layer(x)
            x = F.relu(h)

        output = self.output_layer(x)
        return output


class NeuralArchitecture(NeuralNetwork):
    """
    A wrapper class for the NeuralNetwork class. This class is used for the evolutionary search algorithms
    """

    def __init__(self, input_size, hidden_sizes, activation):
        super().__init__(input_size, hidden_sizes, activation)
        self.train_acc = 0
        self.acc_objective = 0
        self.nondominated_rank = 0
        self.crowding_distance = 0.0

    def accuracy(self, outputs, labels):
        """
        This function calculates the accuracy of the neural network. It is used for training and evaluating the
        neural network.
        :param outputs: The outputs of the neural network. Data type is torch.Tensor
        :param labels: The labels of the data. Data type is torch.Tensor
        :return: Returns the accuracy of the neural network
        """
        predictions = outputs.argmax(-1)
        correct = accuracy_score(predictions.cpu().detach().numpy(), labels.cpu().detach().numpy())
        return correct

    def evaluate_accuracy(self, loader):
        """
        Evaluates the accuracy of the neural network
        :param loader: Data loader for the dataset
        :return: Returns the accuracy of the neural network
        """
        criterion = nn.CrossEntropyLoss()

        self.eval()
        acc = 0
        loss = 0
        n_samples = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss += criterion(outputs, targets).item() * len(targets)
                acc += self.accuracy(outputs, targets) * len(targets)
                n_samples += len(targets)

        self.acc_objective = acc / n_samples

        return acc / n_samples

    def train_model(self, train_loader, epochs=20):
        """
        Trains the neural network. Optimizer is Adam, learning rate is 1e-4, and loss function is CrossEntropyLoss
        (can change if needed)
        :param epochs: Number of epochs (rounds) to train the neural network for
        :param train_loader: Data loader for the training dataset
        :return: Returns the loss and accuracy of the neural network
        """
        criterion = nn.CrossEntropyLoss()
        lr = 1e-3
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            train_loss = 0
            train_acc = 0
            n_samples = 0
            self.train()

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.detach().item() * len(targets)
                train_acc += self.accuracy(outputs, targets) * len(targets)
                n_samples += len(targets)

        final_train_accuracy = train_acc / n_samples
        self.train_acc = final_train_accuracy
        return final_train_accuracy


def clone(self):
        return NeuralArchitecture(self.input_size, self.hidden_sizes.copy())


def Dataloading(x, y, batch_size, train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.Tensor(x).to(device)
    y = torch.LongTensor(y.values).to(device)

    dataset = torch.utils.data.TensorDataset(x, y)
    if train:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
