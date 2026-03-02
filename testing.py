import time
from evaluation import Svmevaluation, evaluation_last
from training import training

def testing(train_data, test_data, dataset_name):
    """
    Function for feature selection and testing.
    :param train_data: ndarray, training dataset containing features and labels for model training.
    :param test_data: ndarray, testing dataset containing features and labels for model validation.
    :param dataset_name: str, name of the dataset.
    :return: Tuple containing:
        - SVMAccuracy (float): SVM accuracy on the test set.
        - KNNAccuracy (float): KNN accuracy on the test set.
        - AUC (float): Area Under the Curve, representing classifier performance.
        - d_feature (float): Average length of all feature subsets in the archive M.
        - tt1 (float): Total execution time from start to finish.
        - gbest (ndarray): Global best solution found by the optimization algorithm.
        - ggbest (list or ndarray): Evaluation results from `evaluation_last`, including various metrics.
        - AC (float): Additional performance metrics output by the algorithm.
        - first_iter_acc (float): Accuracy of the first iteration.
        - m_acc (list): Accuracy history across all iterations.
        - svm_accuracies (list): SVM accuracy recorded at each iteration.
        - Firstfeatures (int): Number of features selected in the first iteration.
    """
    X_train = train_data
    Y_test = test_data

    # Record the start time
    start_time = time.time()

    # Streaming feature selection
    gbest, mm, AC, sel_f, M, first_iter_acc, m_acc, Firstfeatures, ga_call_count = training(X_train, dataset_name)

    # Calculate execution time
    tt1 = time.time() - start_time

    # Evaluate SVM accuracy for all feature subsets stored in the archive M
    svm_accuracies = []
    selected_10_features = []
    for subset in M:
        result = Svmevaluation(X_train, Y_test, subset)
        svm_accuracies.append(result * 100)
        selected_10_features.append(len(subset))

    # Call evaluationlast to compute the final performance metrics (ggbest)
    ggbest = evaluation_last(X_train, Y_test, sel_f)

    # Extract corresponding metrics from ggbest
    SVMAccuracy = ggbest[-4]
    KNNAccuracy = ggbest[-3]
    AUC = ggbest[-1]

    # Calculate d_feature as the average length of each feature subset in M
    d_feature = sum(len(subset) for subset in M) / len(M)

    return SVMAccuracy, KNNAccuracy, AUC, d_feature, tt1, gbest, ggbest, AC, first_iter_acc, m_acc, round(
        svm_accuracies[0], 2), svm_accuracies, Firstfeatures, ga_call_count, selected_10_features