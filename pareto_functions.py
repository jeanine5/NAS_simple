import numpy as np
from architectures import *


def get_corr_archs(front, architectures: list[NeuralArchitecture]):
    """
    Get the architectures corresponding to the indices in the front
    :param front: list of indices
    :param architectures: list of NeuralArchitecture objects
    :return: list of NeuralArchitecture objects
    """
    corr_archs = []
    for idx in front:
        corr_archs.append(architectures[idx])

    return corr_archs


def crowded_comparison_operator(ind1: NeuralArchitecture, ind2: NeuralArchitecture):
    """
    Crowded comparison operator defined from Deb et al. (2002). https://ieeexplore.ieee.org/document/996017
    :param ind1: NeuralArchitecture object
    :param ind2: NeuralArchitecture object
    :return: True if ind1 is better than ind2, False otherwise
    """
    if (ind1.nondominated_rank < ind2.nondominated_rank) or (ind1.nondominated_rank == ind2.nondominated_rank and
                                                             ind1.crowding_distance > ind2.crowding_distance):
        return True
    else:
        return False


def set_non_dominated_ranks(population: list[NeuralArchitecture], fronts: list[np.ndarray]) -> None:
    """
    Set the non-dominated rank for each architecture in the population based on the given fronts.
    :param population: List of NeuralArchitecture objects.
    :param fronts: List of Pareto fronts, where each front is represented by a numpy array of indices.
    """
    for rank, front in enumerate(fronts):
        for idx in front:
            population[idx].nondominated_rank = rank


def is_pareto_dominant(p: NeuralArchitecture, q: NeuralArchitecture):
    """
    Check if p dominates q. In other words, is p a better architecture than q, by objective values.
    :param p: list of fitness values
    :param q: list of fitness values
    :return: True if p dominates q, False otherwise
    """
    return p.acc_objective > q.acc_objective


def fast_non_dominating_sort(population):
    """
    Fast non-dominated sort algorithm from Deb et al. (2002). https://ieeexplore.ieee.org/document/996017
    Code from: https://github.com/adam-katona/NSGA_2_tutorial/blob/master/NSGA_2_tutorial.ipynb
    :param population:  list of fitness values
    :return: list of Pareto fronts
    """

    domination_sets = []
    domination_counts = []

    for arch_1 in population:
        current_domination_set = set()
        domination_counts.append(0)
        for i, arch_2 in enumerate(population):
            if is_pareto_dominant(arch_1, arch_2):
                current_domination_set.add(i)
            elif is_pareto_dominant(arch_2, arch_1):
                domination_counts[-1] += 1

        domination_sets.append(current_domination_set)

    domination_counts = np.array(domination_counts)
    fronts = []
    while True:
        current_front = np.where(domination_counts == 0)[0]
        if len(current_front) == 0:
            break
        fronts.append(current_front)

        for individual in current_front:
            domination_counts[
                individual] = -1
            dominated_by_current_set = domination_sets[individual]
            for dominated_by_current in dominated_by_current_set:
                domination_counts[dominated_by_current] -= 1

    return fronts


def crowding_distance_assignment(population, front: list[np.ndarray]):
    """
    Calculate the crowding distance for each individual in the population and set it as an attribute.

    Crowding distance assignment from Deb et al. (2002). https://ieeexplore.ieee.org/document/996017
    Code from: https://github.com/adam-katona/NSGA_2_tutorial/blob/master/NSGA_2_tutorial.ipynb
        :param front:
        :param population: List of NeuralArchitecture objects.
    """
    if len(front) == 1:
        population[front[0]].crowding_distance = np.inf
        return

    num_individuals = len(population)
    fitnesses = np.array([ind.acc_objective for ind in population])

    # Normalize fitness values
    min_fitness = np.min(fitnesses)
    max_fitness = np.max(fitnesses)
    normalized_fitnesses = (fitnesses - min_fitness) / (
                max_fitness - min_fitness) if max_fitness > min_fitness else fitnesses

    sorted_front = sorted(front, key=lambda x: normalized_fitnesses[x])
    population[sorted_front[0]].crowding_distance = np.inf
    population[sorted_front[-1]].crowding_distance = np.inf

    for i in range(1, len(sorted_front) - 1):
        distance = normalized_fitnesses[sorted_front[i + 1]] - normalized_fitnesses[sorted_front[i - 1]]
        population[sorted_front[i]].crowding_distance += distance



def fronts_to_nondomination_rank(fronts):
    """
    :param fronts:
    :return:
    """
    non_domination_rank_dict = {}
    for i, front in enumerate(fronts):
        for x in front:
            non_domination_rank_dict[x] = i
    return non_domination_rank_dict
