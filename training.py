import time
import numpy as np
from irrelevant_feature_filtering import irrelevant_feature_filtering
from incremental_redundant_feature_clustering import affinity_propagation, incremental_clustering
from interactive_feature_subset_search import initialize, initialize_with_hsinfo, variable_length_integer_ga, variable_length_integer_ga_with_hsinfo


def training(X_train, dataset_name):
    """
    Main execution function for the streaming feature selection pipeline.
    :param X_train: ndarray, training dataset containing features and labels.
    :param dataset_name: str, name of the dataset.
    :return: Tuple containing:
        - gbest (ndarray): Global best solution representing the selected feature set.
        - mm (int): Number of feature clusters.
        - AC (list): Accuracy history, recording accuracy at each iteration.
        - sel_f (list): Final selected feature subset.
        - M (list): Archive/Reserve set.
        - first_acc (float): Training accuracy after the first iteration.
        - m_acc (list): History of training accuracies.
        - Firstfeatures (int): Number of features after the first iteration.
    """
    start_time = time.time()

    # Separate features and labels
    Y0 = X_train[:, 0]  # Class labels
    X1 = X_train[:, 1:]  # Features

    # Initialize variables
    mm = 0
    c0_all = []  # Stores the metric values for each feature in X1
    m1 = 0  # Number of optimal solutions in the archive
    M = []  # Archive/Reserve set
    sel_f = []  # Temporarily store the best feature subset of the current window
    m_acc = []  # Store accuracy history
    X2 = None  # Store processed features
    gbest = []  # Global best solution
    AC = []  # Store accuracies
    feature_cluster_map = {}  # Initialize mapping dictionary
    ga_call_count = 0  # Record the number of GA calls
    ga_accuracy_history = []  # Store accuracy history after each GA call
    kmedoids_model = None  # Initialize K-Medoids model

    # Total number of feature columns
    n_cols = X1.shape[1]

    # Divide features into 10 chunks for streaming simulation
    chunk_size = n_cols // 10

    Firstfeatures = 0  # Store the number of features after the first iteration

    for i in range(10):
        iter_start_time = time.time()

        # Calculate start and end indices for the current chunk
        start_col = i * chunk_size
        if i < 9:
            end_col = start_col + chunk_size
        else:
            end_col = n_cols  # Take all remaining columns in the 10th iteration

        # Load current chunk data
        X0 = X1[:, start_col:end_col]

        # GA parameters
        popsize = 20
        tt = 20

        # Remove irrelevant features
        c, f, X, c0_all, kmedoids_model = irrelevant_feature_filtering(X0, Y0, c0_all, kmedoids_model)
        k = X.shape[1]

        # Incremental redundant feature clustering
        if X2 is None:
            X2 = X0
        else:
            X2 = np.concatenate([X2, X0], axis=1)

        if i == 0:
            # Processing for the first iteration
            u, C, mm = affinity_propagation(k, X0, c, f)
            pop, n_select = initialize(popsize, u, C, mm)
            # Use variable length integer GA for feature selection and optimization
            gbest, AC, sel_f, feature_cluster_map, m_acc, M, m1 = variable_length_integer_ga(pop, popsize, mm, n_select, feature_cluster_map,
                                                                     X2, Y0, u, tt, M, m_acc, m1, alpha=0.0,
                                                                     mutation_rate=0.1)
            Firstfeatures = len(sel_f)

        else:
            # Processing for subsequent iterations
            feature_cluster_map_new = feature_cluster_map.copy()
            u, C, mm, feature_cluster_map_new = incremental_clustering(k, X1, X, c, c0_all, f, start_col, feature_cluster_map_new)

            pop, n_select = initialize_with_hsinfo(popsize, u, C, mm, feature_cluster_map_new)
            gbest, AC, sel_f, feature_cluster_map, m_acc, M, m1, ga_call_count, ga_accuracy_history = variable_length_integer_ga_with_hsinfo(pop, popsize, mm, n_select,
                                                                     feature_cluster_map_new, X2, Y0, u, tt, M, m_acc,
                                                                     m1, alpha=0.0, mutation_rate=0.1, sel_f=sel_f,
                                                                     ga_call_count=ga_call_count, ga_accuracy_history=ga_accuracy_history)

        # Print iteration information
        print(f"Iteration time: {time.time() - iter_start_time:.2f} seconds")
        print(f"Current global best classification accuracy: {m_acc[-1]:.2f}%")
        print(f"Current global best feature set: {sel_f}")
        print(f"Number of features in the current global best set: {len(sel_f)}")
        print(f"Number of optimal solutions in the archive: {m1}")

    total_time = time.time() - start_time
    # Print final results
    print("*" * 50)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Final global best classification accuracy: {m_acc[-1]:.2f}%")
    print(f"Final number of selected features: {len(sel_f)}")
    print("*" * 50)

    return gbest, mm, AC, sel_f, M, round(m_acc[0], 2), m_acc, Firstfeatures, ga_call_count