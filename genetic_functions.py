import random
from architectures import *


def binary_tournament_selection(population: list[NeuralArchitecture], tournament_size=2):
    """
    Perform binary tournament selection on the population
    :param population: list of NeuralArchitecture objects
    :param tournament_size: size of the tournament, default is 2 for binary tournament selection
    :return: the NeuralArchitecture object with the best fitness value
    """

    # select 2 parent models from population P
    selected_parents = random.sample(population, tournament_size)

    # determine two parents w/ best fitness val
    return min(selected_parents, key=lambda arch: arch.nondominated_rank)


def crossover(parent1: NeuralArchitecture, parent2: NeuralArchitecture, crossover_rate: float):
    """
    Perform crossover on two NeuralArchitecture objects
    :param parent1: NeuralArchitecture object
    :param parent2: NeuralArchitecture object
    :param crossover_rate: probability of crossover
    :return:
    """
    if random.uniform(0, 1) < crossover_rate:
        if len(parent1.hidden_sizes) <= 3 or len(parent2.hidden_sizes) <= 3:
            return one_point_crossover(parent1, parent2)
        return two_point_crossover(parent1, parent2)
    else:
        return one_point_crossover(parent1, parent2)


def one_point_crossover(parent1: NeuralArchitecture, parent2: NeuralArchitecture):
    """
    Perform one point crossover on two NeuralArchitecture objects
    :param parent1: NeuralArchitecture object
    :param parent2: NeuralArchitecture object
    :return: offspring NeuralArchitecture object
    """
    # sanity check
    assert parent1.input_size == parent2.input_size

    # pick a crossover point
    crossover_point = random.randint(1, len(parent1.hidden_sizes) - 1)

    # combine hidden layer sizes from both parents at the crossover point
    offspring_hidden_sizes = parent1.hidden_sizes[:crossover_point] + parent2.hidden_sizes[crossover_point:]

    offspring_activation = random.choice([parent1.activation, parent2.activation])

    offspring = NeuralArchitecture(parent1.input_size, offspring_hidden_sizes, offspring_activation)

    # Combine parameters from both parents
    parent1_params = list(parent1.parameters())
    parent2_params = list(parent2.parameters())
    offspring_params = list(offspring.parameters())

    # for i in range(len(offspring_params)):
    #     if i < len(parent1_params) // 2:
    #         offspring_params[i].data.copy_(parent1_params[i].data)
    #     else:
    #         offspring_params[i].data.copy_(parent2_params[i].data)

    return offspring

def two_point_crossover(parent1: NeuralArchitecture, parent2: NeuralArchitecture):
    """
    Perform two point crossover on two NeuralArchitecture objects
    :param parent1: NeuralArchitecture object
    :param parent2: NeuralArchitecture object
    :return: offspring NeuralArchitecture object
    """
    # sanity checks
    assert parent1.input_size == parent2.input_size
    assert len(parent1.hidden_sizes) >= 2 and len(parent2.hidden_sizes) >= 2

    # choose two crossover points
    crossover_point1 = random.randint(1, len(parent1.hidden_sizes) - 2)
    crossover_point2 = random.randint(crossover_point1 + 1, len(parent1.hidden_sizes) - 1)

    offspring_hidden_sizes = (
            parent1.hidden_sizes[:crossover_point1] +
            parent2.hidden_sizes[crossover_point1:crossover_point2] +
            parent1.hidden_sizes[crossover_point2:]
    )

    offspring_activation = random.choice([parent1.activation, parent2.activation])

    offspring = NeuralArchitecture(parent1.input_size, offspring_hidden_sizes, offspring_activation)

    # Combine parameters from both parents
    parent1_params = list(parent1.parameters())
    parent2_params = list(parent2.parameters())
    offspring_params = list(offspring.parameters())

    # for i in range(len(offspring_params)):
    #     if i < crossover_point1 or i >= crossover_point2:
    #         offspring_params[i].data.copy_(parent1_params[i].data)
    #     else:
    #         if parent1_params[i].size() == parent2_params[i].size():
    #             offspring_params[i].data.copy_(parent2_params[i].data)
    #         else:
    #             print(f"Skipping crossover for layer {i} due to size mismatch")
    #             offspring_params[i].data.copy_(parent1_params[i].data)

    return offspring


def mutate(offspring: NeuralArchitecture, mutation_factor: float):
    """
    Perform mutation on a NeuralArchitecture object. Multiple mutations may occur
    :param offspring: NeuralArchitecture object
    :param mutation_factor: probability of mutation
    :return: mutated NeuralArchitecture object
    """
    # Mutate hidden layer sizes
    if random.uniform(0, 1) < mutation_factor:
        layer_to_mutate = random.randint(0, len(offspring.hidden_sizes) - 1)
        change = random.choice([-1, 1]) * random.randint(1, 5)  # Change by a random number of neurons
        new_size = max(1, offspring.hidden_sizes[layer_to_mutate] + change)  # Ensure size is at least 1
        offspring.hidden_sizes[layer_to_mutate] = new_size

    # Mutate by adding a new layer
    if random.uniform(0, 1) < mutation_factor:
        new_layer_size = random.randint(1, 50)  # Random size for the new layer
        position = random.randint(0, len(offspring.hidden_sizes))
        offspring.hidden_sizes.insert(position, new_layer_size)

    # Mutate by removing a layer
    if len(offspring.hidden_sizes) > 1 and random.uniform(0, 1) < mutation_factor:
        layer_to_remove = random.randint(0, len(offspring.hidden_sizes) - 1)
        offspring.hidden_sizes.pop(layer_to_remove)

    # Mutate activation function
    if random.uniform(0, 1) < mutation_factor:
        offspring.activation = random.choice([nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.Tanh])
            # nn.Tanh

    # Reinitialize the model with the new hidden sizes
    offspring = NeuralArchitecture(offspring.input_size, offspring.hidden_sizes, offspring.activation).to(device)

    # Mutate weights and biases
    for param in offspring.parameters():
        if random.uniform(0, 1) < mutation_factor:
            mutation = torch.randn_like(param) * mutation_factor
            param.data += mutation

    return offspring


def generate_offspring(population: list[NeuralArchitecture], crossover_rate: float, mutation_rate: float):
    """
    Generate offspring population using binary tournament selection, crossover, and mutation
    :param population: list of NeuralArchitecture objects
    :param crossover_rate: probability of crossover
    :param mutation_rate: probability of mutation
    :return: list of NeuralArchitecture objects
    """
    offspring_pop = []

    for _ in range(len(population)):
        parent_1 = binary_tournament_selection(population)
        parent_2 = binary_tournament_selection(population)

        offspring = crossover(parent_1, parent_2, crossover_rate)

        mutated_offspring = mutate(offspring, mutation_rate)

        offspring_pop.append(mutated_offspring)

    return offspring_pop
