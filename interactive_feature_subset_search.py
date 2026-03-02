import numpy as np
import math
import random
from evaluation import evaluation


def initialize(popsize, u, C, mm):
    """
    Initialize the population for the GA using integer encoding.
    Employs a rank-based non-linear transformation method to make differences in
    NMI values more distinct.
    :param popsize: int, size of the population.
    :param u: list of lists, each element is a feature cluster containing global feature indices.
    :param C: list of lists, NMI values corresponding to each feature cluster in u, representing feature importance.
    :param mm: int, number of feature clusters.
    :return:
        - population_array (numpy.ndarray): Initialized population, where each individual is a list of feature indices (1-based).
        - n_select (list): Number of features to be selected for each feature cluster.
    """
    n_select = [math.floor(math.sqrt(len(u[j]))) for j in range(mm)]

    # Pre-calculate probability distributions for all feature clusters
    all_probabilities = []

    for j in range(mm):
        if len(u[j]) > 0:
            if np.sum(C[j]) > 0:
                mi_values = np.array(C[j])

                # Rank-based transformation
                # Sort NMI values and obtain ranks (larger NMI values get smaller ranks)
                ranks = np.argsort(np.argsort(-mi_values)) + 1  # 1-based ranking

                # Calculate weights using the reciprocal of the square of the ranks
                transformed_mi = 1.0 / (ranks ** 2)

                # Normalize into a probability distribution
                probabilities = transformed_mi / np.sum(transformed_mi)
            else:
                probabilities = np.ones(len(u[j])) / len(u[j])
        else:
            probabilities = np.array([])

        all_probabilities.append(probabilities)

    population = []

    for i in range(popsize):
        individual = []
        for j in range(mm):
            if len(u[j]) > 0:
                num_to_select = np.random.randint(1, n_select[j] + 1)

                # Use the pre-calculated probability distribution
                probabilities = all_probabilities[j]

                selected_features = np.random.choice(
                    np.arange(1, len(u[j]) + 1),  # 1-based
                    size=num_to_select,
                    replace=False,
                    p=probabilities
                ).tolist()
            else:
                selected_features = []

            # Fill the remaining positions with 0 if selected features are fewer than n_select[j]
            if len(selected_features) < n_select[j]:
                selected_features.extend([0] * (n_select[j] - len(selected_features)))

            individual.extend(selected_features)

        population.append(individual)

    population_array = np.array(population)
    return population_array, n_select


def initialize_with_hsinfo(popsize, u, C, mm, feature_cluster_map):
    """
    Initialize the population for the GA.
    :param popsize: int, size of the population.
    :param u: list of lists, each element is a feature cluster containing global indices of features.
    :param C: list of lists, NMI values corresponding to each feature cluster in u, representing feature importance.
    :param mm: int, number of feature clusters.
    :param feature_cluster_map: dict, mapping between feature clusters and features, formatted as {cluster_index: {feature_indices}}.
    :return:
        - population_array (numpy.ndarray): Initialized population, where each individual is a feature index list (1-based).
        - n_select (list): Number of features to be selected from each feature cluster.
    """
    # Calculate the number of features to select for each cluster (floor(sqrt(len(u[j]))))
    n_select = [math.floor(math.sqrt(len(u[j]))) if len(u[j]) > 0 else 0 for j in range(mm)]

    # Pre-calculate probability distributions for all feature clusters
    all_probabilities = []

    for j in range(mm):
        if len(u[j]) > 0:  # Ensure the feature cluster is not empty
            if np.sum(C[j]) > 0:
                mi_values = np.array(C[j])

                # Rank-based transformation
                ranks = np.argsort(np.argsort(-mi_values)) + 1  # 1-based rank (larger NMI gets smaller rank)
                transformed_mi = 1.0 / (ranks ** 2)  # Reciprocal of the square of the rank

                # Normalize into a probability distribution
                probabilities = transformed_mi / np.sum(transformed_mi)
            else:
                probabilities = np.ones(len(u[j])) / len(u[j])
        else:
            probabilities = np.array([])
        all_probabilities.append(probabilities)

    population = []

    for i in range(popsize):
        individual = []
        for j in range(mm):
            num_to_select = np.random.randint(1, n_select[j] + 1) if n_select[j] > 0 else 0
            selected_features = []

            for _ in range(num_to_select):
                if random.random() <= 0.5:
                    # Strategy 1: Select from feature_cluster_map, ensuring no duplicates
                    if j + 1 in feature_cluster_map:
                        cluster_features = list(feature_cluster_map[j + 1])
                        u_features = u[j]
                        remaining_selected_features = []

                        for feature in cluster_features:
                            if feature in u_features:
                                index_in_u = u_features.index(feature) + 1  # 1-based
                                if index_in_u not in selected_features:
                                    remaining_selected_features.append(index_in_u)

                        # Check if remaining_selected_features is not empty
                        if remaining_selected_features:
                            selected_features.append(np.random.choice(remaining_selected_features))
                        else:
                            # If empty, switch to Strategy 2
                            use_strategy_2 = True
                else:
                    use_strategy_2 = True

                # Strategy 2: Use pre-calculated enhanced probability distribution for roulette wheel selection
                if 'use_strategy_2' in locals() and use_strategy_2:
                    if len(u[j]) > 0:  # Ensure feature cluster is not empty
                        remaining_indices = list(set(np.arange(1, len(u[j]) + 1)) - set(selected_features))

                        if remaining_indices:  # Ensure there are still selectable features
                            # Use pre-calculated probabilities
                            probabilities = all_probabilities[j]
                            remaining_probabilities = np.array([probabilities[idx - 1] for idx in remaining_indices])
                            remaining_probabilities /= np.sum(remaining_probabilities)  # Ensure probabilities sum to 1

                            selected_feature = np.random.choice(
                                remaining_indices,
                                size=1,
                                replace=False,
                                p=remaining_probabilities
                            )
                            selected_features.append(selected_feature[0])

                    # Reset flag
                    if 'use_strategy_2' in locals():
                        del use_strategy_2

            # Fill with 0 to match n_select[j]
            if len(selected_features) < n_select[j]:
                selected_features.extend([0] * (n_select[j] - len(selected_features)))

            individual.extend(selected_features)

        population.append(individual)

    population_array = np.array(population)
    return population_array, n_select


def selection(pop, fitnesses, num_parents, tournament_size=3):
    """
    Tournament selection operation, choosing individuals with lower fitness as parents.
    :param pop: ndarray, population matrix.
    :param fitnesses: ndarray, fitness array.
    :param num_parents: int, number of parents to select.
    :param tournament_size: int, tournament size.
    :return: parents: ndarray, parent population.
    """
    parents = []
    popsize = pop.shape[0]
    selected_indices = set()  # Store indices of already selected parents

    for _ in range(num_parents):
        # Randomly select individuals for the tournament
        participants = np.random.choice(popsize, tournament_size, replace=False)
        # Choose the individual with the minimum fitness
        best_participant = participants[np.argmin(fitnesses[participants])]

        # Ensure that selected parents are unique
        while best_participant in selected_indices:
            participants = np.random.choice(popsize, tournament_size, replace=False)
            best_participant = participants[np.argmin(fitnesses[participants])]

        parents.append(pop[best_participant])
        selected_indices.add(best_participant)

    return np.array(parents)


def crossover(parents, offspring_size):
    """
    Perform single-point crossover to generate offspring.

    Args:
        parents (np.ndarray): The parent population.
        offspring_size (tuple): The size of the offspring population (num_offspring, num_genes).

    Returns:
        np.ndarray: The generated offspring population.
    """
    num_offspring, num_genes = offspring_size
    offspring = np.empty(offspring_size, dtype=parents.dtype)

    for k in range(num_offspring):
        # Randomly select two distinct parent individuals
        parent1_idx, parent2_idx = np.random.choice(parents.shape[0], size=2, replace=False)

        # Randomly select a crossover point
        crossover_point = np.random.randint(1, num_genes)

        # Generate offspring
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring


def mutation(offspring, parents, u, n_select, mutation_rate):
    """
    Mutation operation to ensure no duplicate individuals between offspring and offspring,
    or between offspring and parents.

    :param offspring: ndarray, offspring population, shape (num_offspring, num_genes)
    :param parents: ndarray, parent population, shape (num_parents, num_genes)
    :param u: list of lists, feature group indices, each sublist contains feature IDs within a cluster
    :param n_select: list of int, number of features to select for each feature cluster
    :param mutation_rate: float, mutation probability (between 0 and 1)
    :return: ndarray, mutated offspring population
    """
    num_offspring, num_genes = offspring.shape

    # Ensure the sum of n_select matches the number of genes
    if sum(n_select) != num_genes:
        raise ValueError("The sum of n_select does not match the number of genes.")

    # Calculate the gene index range for each feature cluster
    cluster_ranges = []
    start = 0
    for select in n_select:
        end = start + select
        cluster_ranges.append((start, end))
        start = end

    # Perform mutation on each offspring
    for idx in range(num_offspring):
        # Process each feature cluster
        for cluster_id, (start, end) in enumerate(cluster_ranges):
            selected_positions = offspring[idx, start:end]
            unique_positions, counts = np.unique(selected_positions, return_counts=True)

            # For duplicate genes within a cluster, mutate the second duplicate gene
            # (assumes a gene only repeats twice at most)
            for unique_position, count in zip(unique_positions, counts):
                if count > 1:
                    duplicate_indices = np.where(selected_positions == unique_position)[0]
                    dup_idx = duplicate_indices[1]
                    total_positions = len(u[cluster_id])
                    available_positions = list(set(np.arange(1, total_positions + 1)) - set(selected_positions))

                    if available_positions:
                        new_position = np.random.choice(available_positions)
                        offspring[idx, start + dup_idx] = new_position
                        selected_positions[dup_idx] = new_position

        # Randomly mutate other positions based on mutation_rate
        mutation_matrix = np.random.rand(num_genes) < mutation_rate
        # Process mutation by cluster
        for cluster_id, (start, end) in enumerate(cluster_ranges):
            # Get indices within the current cluster that need mutation
            cluster_mutation_indices = np.where(mutation_matrix[start:end])[0]

            if len(cluster_mutation_indices) > 0:
                # Get all genes in the current cluster
                cluster_genes = offspring[idx, start:end].copy()
                total_positions = len(u[cluster_id])

                # Process each position requiring mutation within the cluster
                for mut_idx in cluster_mutation_indices:
                    # Get all gene values in current cluster except for the one being mutated
                    current_values = set(cluster_genes)
                    current_values.remove(cluster_genes[mut_idx])

                    # Calculate available new positions
                    available_positions = list(set(range(1, total_positions + 1)) - current_values)

                    if available_positions:
                        # Randomly select a new position
                        new_position = np.random.choice(available_positions)
                        # Update the gene value
                        cluster_genes[mut_idx] = new_position
                        offspring[idx, start + mut_idx] = new_position

    # Check for duplicate individuals across the population
    for idx in range(num_offspring):
        while True:
            current_individual = tuple(offspring[idx])
            # Count occurrences of the current individual in parents and offspring
            count = 0
            # Check parents
            for parent in parents:
                if tuple(parent) == current_individual:
                    count += 1
            # Check offspring
            for i in range(num_offspring):
                if tuple(offspring[i]) == current_individual:
                    count += 1

            # If the individual appears only once (itself), there is no duplicate
            if count == 1:
                break

            # If duplicates exist, re-mutate the current individual and re-check
            print(f"Duplicate individual found, re-mutating offspring {idx}...")
            # Use a higher mutation rate to increase diversity
            offspring[idx] = mutation_single(offspring[idx], u, 0.3, cluster_ranges)

    return offspring


def mutation_single(individual, u, mutation_rate, cluster_ranges):
    """
    Mutate a single individual.
    :param individual: ndarray, genes of a single individual
    :param u: list of lists, feature group indices
    :param mutation_rate: float, mutation probability
    :param cluster_ranges: list of tuples, gene index ranges for each feature cluster
    :return: ndarray, the mutated individual
    """
    mutated = individual.copy()

    # Process each cluster
    for cluster_id, (start, end) in enumerate(cluster_ranges):
        cluster_size = end - start
        cluster_genes = mutated[start:end]

        # Generate mutation mask
        mutation_mask = np.random.rand(cluster_size) < mutation_rate
        mutation_indices = np.where(mutation_mask)[0]

        if len(mutation_indices) > 0:
            # Process each position requiring mutation
            total_positions = len(u[cluster_id])

            for mut_idx in mutation_indices:
                # Get all gene values in current cluster except for the one being mutated
                current_values = set(cluster_genes)
                current_values.remove(cluster_genes[mut_idx])

                # Calculate available new positions
                available_positions = list(set(range(1, total_positions + 1)) - current_values)

                if available_positions:
                    # Randomly select a new position
                    new_position = np.random.choice(available_positions)
                    # Update gene value
                    cluster_genes[mut_idx] = new_position
                    mutated[start + mut_idx] = new_position

    return mutated


def variable_length_integer_ga(pop, popsize, mm, n_select, feature_cluster_map, X2, Y0, u, tt, M, m_acc, m1, alpha, mutation_rate):
    """
    Perform feature selection using a GA.
    :param pop: numpy.ndarray, initialized population, where each row represents an individual and each column represents the selection status of a feature.
    :param popsize: int, population size.
    :param mm: int, number of feature groups.
    :param n_select: list, number of features to select from each feature group.
    :param feature_cluster_map: dict, mapping between feature clusters and features selected in sel_f.
    :param X2: numpy.ndarray, feature matrix of the input data.
    :param Y0: numpy.ndarray, class labels of the data.
    :param u: list of lists, list of indices for feature sets, where each element is a feature group.
    :param tt: int, maximum number of iterations.
    :param M: list, records the global optimal feature set for each iteration.
    :param m_acc: list, records the classification accuracy of the optimal solution in each generation.
    :param m1: int, number of times the optimal solution is recorded.
    :param alpha: float, weight parameter to balance the impact of classification accuracy and the number of features.
    :param mutation_rate: float, mutation probability.
    :return: tuple, containing the following elements:
        - gbest (ndarray): Global optimal solution, including feature selection status, Balanced Accuracy, and Fitness.
        - AC (list): Global optimal solution for each generation.
        - sel_f (list): Final selected feature set.
        - feature_cluster_map (dict): Mapping between feature clusters and features selected in sel_f.
        - train_acc (list): Updated record of classification accuracy for the optimal solution in each generation (Note: maps to m_acc).
        - M (list): Updated global optimal feature set.
        - m1 (int): Updated number of times the optimal solution is recorded.
    """
    # Calculate the fitness matrix, including Balanced Accuracy and Fitness
    popeff = evaluation(pop, popsize, mm, n_select, X2, Y0, u, alpha)

    # Initialize individual optimal solutions, with the last two columns being Balanced Accuracy and Fitness
    LBEST = np.copy(popeff)

    # Find the individual with the minimum Fitness as the global optimal solution
    fitness_column = sum(n_select) + 1
    gbest_idx = np.argmin(LBEST[:, fitness_column])
    gbest = LBEST[gbest_idx, :sum(n_select) + 2]  # Contains gene, Balanced Accuracy, and Fitness

    AC = []  # Used to store the global optimal solution for each generation
    t = 0
    while t < tt:
        # Extract Fitness
        fitnesses = popeff[:, fitness_column]

        # Select parents
        num_parents = popsize // 2  # Select half as parents
        parents = selection(pop, fitnesses, num_parents)

        # Generate offspring
        offspring_size = (popsize - parents.shape[0], sum(n_select))
        offspring = crossover(parents, offspring_size)

        # Mutate offspring
        offspring = mutation(offspring, parents, u, n_select, mutation_rate)

        # Combine parents and offspring to form a new population
        population_new = np.vstack((parents, offspring))

        # Evaluate the fitness of the new population
        popeff_new = evaluation(population_new, popsize, mm, n_select, X2, Y0, u, alpha)

        # Update population
        pop = population_new.copy()
        popeff = popeff_new.copy()

        # Update individual optimal solution LBEST, comparing only the Fitness column
        update_mask = popeff_new[:, fitness_column] < LBEST[:, fitness_column]
        LBEST[update_mask, :sum(n_select) + 2] = popeff_new[update_mask, :sum(n_select) + 2]

        # Find the individual with the minimum Fitness as the current global optimal solution
        current_gbest_idx = np.argmin(LBEST[:, fitness_column])
        current_gbest = LBEST[current_gbest_idx, :sum(n_select) + 2]

        # If the current global optimal solution is better than the previous one, update gbest
        if current_gbest[fitness_column] < gbest[fitness_column]:
            gbest = current_gbest.copy()
            AC.append(gbest.copy())  # Record the global optimal solution of the current generation

        t += 1
        print(f"Iteration {t}: Current global optimal Fitness = {gbest[fitness_column]:.6f}, Balanced Accuracy = {gbest[sum(n_select)]:.2f}%")

    # Select the final feature set and generate feature_cluster_map
    sel_f = []
    current_gene = 0
    for j in range(mm):
        cluster_features = []  # List of features selected by the current feature cluster
        for s in range(n_select[j]):
            gene_val = gbest[current_gene]
            if gene_val != 0:
                selected_feature = u[j][int(gene_val) - 1]  # 1-based to 0-based
                sel_f.append(selected_feature)
                cluster_features.append(selected_feature)
            current_gene += 1
        feature_cluster_map[j + 1] = cluster_features  # Feature cluster numbering starts from 1

    # Update historical information
    m_acc.append(gbest[sum(n_select)])
    M.append(sel_f)
    m1 += 1

    return gbest, AC, sel_f, feature_cluster_map, m_acc, M, m1


def variable_length_integer_ga_with_hsinfo(pop, popsize, mm, n_select, feature_cluster_map, X2, Y0, u, tt, M, m_acc, m1, alpha, mutation_rate, sel_f,
           ga_call_count, ga_accuracy_history):
    """
    Perform feature selection using a GA.
    :param pop: numpy.ndarray, initialized population, where each row represents an individual and each column represents the selection status of a feature.
    :param popsize: int, population size.
    :param mm: int, number of feature groups.
    :param n_select: list, number of features to select from each feature group.
    :param feature_cluster_map: dict, mapping between feature clusters and features selected in sel_f.
    :param X2: numpy.ndarray, feature matrix of the input data.
    :param Y0: numpy.ndarray, class labels of the data.
    :param u: list of lists, list of indices for feature sets, where each element is a feature group.
    :param tt: int, maximum number of iterations.
    :param M: list, records the global optimal feature set for each iteration.
    :param m_acc: list, records the classification accuracy of the optimal solution in each generation.
    :param m1: int, number of times the optimal solution is recorded.
    :param alpha: float, weight parameter to balance the impact of classification accuracy and the number of features.
    :param mutation_rate: float, mutation probability.
    :param sel_f: list, selected feature set.
    :param ga_call_count: int, records the number of times the GA algorithm is called.
    :param ga_accuracy_history: list, records the accuracy changes of the GA algorithm.
    :return: tuple, containing the following elements:
        - gbest (ndarray): Global optimal solution, including feature selection status, Balanced Accuracy, and Fitness.
        - AC (list): Global optimal solution for each generation.
        - sel_f (list): Final selected feature set.
        - feature_cluster_map (dict): Mapping between feature clusters and features selected in sel_f.
        - train_acc (list): Updated record of classification accuracy for the optimal solution in each generation (Note: corresponds to m_acc).
        - M (list): Updated global optimal feature set.
        - m1 (int): Updated number of times the optimal solution is recorded.
        - ga_call_count (int): Updated number of times the GA algorithm is called.
        - ga_accuracy_history (list): Updated accuracy changes of the GA algorithm.
    """
    AC = []  # Used to store the global optimal solution for each generation
    # Calculate the fitness matrix, including Balanced Accuracy and Fitness
    popeff = evaluation(pop, popsize, mm, n_select, X2, Y0, u, alpha)

    # Initialize individual optimal solutions, with the last two columns being Balanced Accuracy and Fitness
    LBEST = np.copy(popeff)

    # Find the individual with the minimum Fitness as the global optimal solution
    fitness_column = sum(n_select) + 1
    gbest_idx = np.argmin(LBEST[:, fitness_column])
    acc = LBEST[gbest_idx, sum(n_select)]
    gbest = LBEST[gbest_idx, :sum(n_select) + 2]  # Contains gene, Balanced Accuracy, and Fitness

    if acc > m_acc[-1]:
        AC.append(gbest.copy())  # Record the global optimal solution of the current generation
        sel_f = []
        current_gene = 0
        for j in range(mm):
            cluster_features = []  # List of features selected by the current feature cluster
            for s in range(n_select[j]):
                gene_val = gbest[current_gene]
                if gene_val != 0:
                    selected_feature = u[j][int(gene_val) - 1]  # 1-based to 0-based
                    sel_f.append(selected_feature)
                    cluster_features.append(selected_feature)
                current_gene += 1
            feature_cluster_map[j + 1] = cluster_features  # Feature cluster numbering starts from 1
        m_acc.append(gbest[sum(n_select)])
        M.append(sel_f)
        m1 += 1
    else:
        t = 0
        while t < tt:
            # Extract Fitness
            fitnesses = popeff[:, fitness_column]

            # Select parents
            num_parents = popsize // 2  # Select half as parents
            parents = selection(pop, fitnesses, num_parents)

            # Generate offspring
            offspring_size = (popsize - parents.shape[0], sum(n_select))
            offspring = crossover(parents, offspring_size)

            # Mutate offspring
            offspring = mutation(offspring, parents, u, n_select, mutation_rate)

            # Combine parents and offspring to form a new population
            population_new = np.vstack((parents, offspring))

            # Evaluate the fitness of the new population
            popeff_new = evaluation(population_new, popsize, mm, n_select, X2, Y0, u, alpha)

            # Update population
            pop = population_new.copy()
            popeff = popeff_new.copy()

            # Update individual optimal solution LBEST, comparing only the Fitness column
            update_mask = popeff_new[:, fitness_column] < LBEST[:, fitness_column]
            LBEST[update_mask, :sum(n_select) + 2] = popeff_new[update_mask, :sum(n_select) + 2]

            # Find the individual with the minimum Fitness as the current global optimal solution
            current_gbest_idx = np.argmin(LBEST[:, fitness_column])
            current_gbest = LBEST[current_gbest_idx, :sum(n_select) + 2]

            # If the current global optimal solution is better than the previous one, update gbest
            if current_gbest[fitness_column] < gbest[fitness_column]:
                gbest = current_gbest.copy()
                AC.append(gbest.copy())  # Record the global optimal solution of the current generation

            t += 1
            print(
                f"Iteration {t}: Current global optimal Fitness = {gbest[fitness_column]:.6f}, Balanced Accuracy = {gbest[sum(n_select)]:.2f}%")

        acc = gbest[sum(n_select)]
        ga_call_count += 1
        ga_accuracy_history.append(acc)

        if acc > m_acc[-1]:
            # Select the final feature set and generate feature_cluster_map
            sel_f = []
            current_gene = 0
            for j in range(mm):
                cluster_features = []  # List of features selected by the current feature cluster
                for s in range(n_select[j]):
                    gene_val = gbest[current_gene]
                    if gene_val != 0:
                        selected_feature = u[j][int(gene_val) - 1]  # 1-based to 0-based
                        sel_f.append(selected_feature)
                        cluster_features.append(selected_feature)
                    current_gene += 1
                feature_cluster_map[j + 1] = cluster_features  # Feature cluster numbering starts from 1

            # Update historical information
            m_acc.append(gbest[sum(n_select)])
            M.append(sel_f)
            m1 += 1
        else:
            m_acc.append(m_acc[-1])
            M.append(sel_f)
            m1 += 1

    return gbest, AC, sel_f, feature_cluster_map, m_acc, M, m1, ga_call_count, ga_accuracy_history