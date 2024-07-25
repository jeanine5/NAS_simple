import csv

from matplotlib import pyplot as plt

from genetic_functions import *
from pareto_functions import *
import functools
import numpy as np

from acc_predictor import *


class NSG2:
    def __init__(self, population_size, generations, crossover_factor, mutation_factor):
        self.population_size = population_size
        self.generations = generations
        self.crossover_factor = crossover_factor
        self.mutation_factor = mutation_factor

    def initial_population(self, train_loader, test_loader, max_hidden_layers, max_hidden_size):
        """
        Initialize the population pool with random deep neural architectures
        :param train_loader:
        :param test_loader:
        :param max_hidden_layers: Maximum number of hidden layers
        :param max_hidden_size: Maximum number of hidden units per layer
        :return: Returns a list of NeuralArchitecture objects
        """

        input_size = next(iter(train_loader))[0].shape[1]
        archs = []
        for _ in range(self.population_size):
            num_hidden_layers = random.randint(3, max_hidden_layers)
            hidden_sizes = [random.randint(10, max_hidden_size) for _ in range(num_hidden_layers)]
            activation = random.choice([nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU])

            arch = NeuralArchitecture(input_size, hidden_sizes, activation)

            train_acc = arch.train_model(train_loader)
            test_acc = arch.evaluate_accuracy(test_loader)
            print(f'trained arch {_}')

            max_hs = max(hidden_sizes)
            min_hs = min(hidden_sizes)

            activation_str = str(activation).split("'")[1].split(".")[-1]
            results = store_results(num_hidden_layers, max_hs, min_hs, activation_str, train_acc, test_acc)
            store_results_to_csv(results, 'Training_CSV')

            archs.append(arch)

        return archs

    def evolve(self, train_loader, test_loader, max_hidden_layers, max_hidden_size):
        """
        The NSGA-2 algorithm. It evolves the population for a given number of generations, however
        there is quite a bit of excessive training going on here.
        :param max_hidden_size:
        :param max_hidden_layers:
        :param test_loader:
        :param train_loader:
        :return: List of the best performing NeuralArchitecture objects of size of at most population_size
        """
        P = self.initial_population(train_loader, test_loader, max_hidden_layers, max_hidden_size)

        regression_trainer = RegressionPredictor('Training_CSV')

        regression_trainer.train_models()
        regression_trainer.evaluate_models()

        # step 4: create an offspring population Q0 of size N
        Q = generate_offspring(P, self.crossover_factor, self.mutation_factor)

        for child in Q:
            child.train_model(train_loader)
            child.evaluate_accuracy(test_loader)
            activation_str = str(child.activation).split("'")[1].split(".")[-1]
            results = store_results(len(child.hidden_sizes), max(child.hidden_sizes), min(child.hidden_sizes),
                                    activation_str, child.train_acc, child.acc_objective)
            store_results_to_csv(results, 'Training_CSV')

        # Main Loop
        for generation in range(self.generations):
            print(f'Generation: {generation + 1}')
            combined_population = P + Q

            F = fast_non_dominating_sort(combined_population)
            set_non_dominated_ranks(combined_population, F)

            P = []
            i = 0
            while len(P) + len(F[i]) <= self.population_size:
                crowding_distance_assignment(combined_population, F[i])
                P.extend([combined_population[idx] for idx in F[i]])
                i += 1

            # sorted_front = sorted(F[i], key=lambda idx: combined_population[idx].crowding_distance, reverse=True)
            best = [combined_population[idx] for idx in F[i]]
            sorted_front = sorted(best, key=functools.cmp_to_key(crowded_comparison_operator), reverse=True)

            P.extend(sorted_front[:(self.population_size - len(P))])

            Q = generate_offspring(P, self.crossover_factor, self.mutation_factor)

            for offspring in Q:
                predicted_performance = regression_trainer.predict_performance(offspring)
                offspring.acc_objective = predicted_performance
                # offspring.train_model(train_loader)

            Q.sort(key=lambda arch: arch.acc_objective, reverse=True)

            # train best N/2
            num_to_train = len(Q) // 2
            for i in range(num_to_train):
                train_acc = Q[i].train_model(train_loader)
                test_acc = Q[i].evaluate_accuracy(test_loader)

                max_hs = max(Q[i].hidden_sizes)
                min_hs = min(Q[i].hidden_sizes)

                activation_str = str(Q[i].activation).split("'")[1].split(".")[-1]
                results = store_results(len(Q[i].hidden_sizes), max_hs, min_hs, activation_str, train_acc, test_acc)
                store_results_to_csv(results, 'Training_CSV')

        return P


def store_results(num_hidden_layers, max_hidden_size, min_hidden_size, activation_function, train_accuracy,
                  test_accuracy):
    result = {
        'num_hidden_layers': num_hidden_layers,
        'max_hidden_size': max_hidden_size,
        'min_hidden_size': min_hidden_size,
        'activation_function': activation_function,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }

    return result


def store_results_to_csv(results, filename):
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['num_hidden_layers', 'max_hidden_size', 'min_hidden_size', 'activation_function',
                      'train_accuracy', 'test_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        csvfile.seek(0, 2)
        is_empty = csvfile.tell() == 0

        if is_empty:
            writer.writeheader()

        writer.writerow(results)
