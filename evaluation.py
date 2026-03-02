import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def evaluate_individual(X2, Y0, mm, pop, u, i, n_select):
    """
    Calculate the classification accuracy (Balanced Accuracy) and the number of selected features for individual i.
    :param X2: numpy.ndarray, feature matrix (excluding labels)
    :param Y0: numpy.ndarray, label vector
    :param mm: int, number of feature clusters
    :param pop: numpy.ndarray, population matrix, where each row represents an individual and each column represents a feature selected by that individual
    :param u: list of lists, list of feature clusters, where each element is a list containing feature indices
    :param i: int, index of the individual (0-based)
    :param n_select: list, number of features selected for each feature cluster
    :return: tuple, (balanced_accuracy, num_selected_features)
        - balanced_accuracy: float, average classification accuracy (Balanced Accuracy) of individual i
        - num_selected_features: int, number of features selected by individual i
    """
    sel_f = []
    current_gene = 0  # Used to track the gene position of the current feature cluster

    # Iterate through each feature cluster
    for j in range(mm):
        # Iterate through all selection positions of the current feature cluster
        for s in range(n_select[j]):
            if current_gene >= pop.shape[1]:
                # Prevent index out of bounds
                break
            gene_val = pop[i, current_gene]
            if gene_val != 0:
                # Select the corresponding feature, note the 1-based indexing
                selected_feature = u[j][int(gene_val) - 1]
                sel_f.append(selected_feature)
            current_gene += 1

    num_selected_features = len(sel_f)  # Number of selected features

    if num_selected_features == 0:
        return 0, 0

    # Extract data for the selected features
    X_selected = X2[:, sel_f]  # Excluding labels

    # Data preprocessing: Min-Max normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_selected)

    # Initialize the SVM classifier
    svm_classifier = svm.SVC(kernel='rbf')

    # Perform 5-fold cross-validation and calculate Balanced Accuracy
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    balanced_accuracies = []

    for train_index, test_index in kf.split(X_scaled, Y0):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = Y0[train_index], Y0[test_index]

        # Train the model
        svm_classifier.fit(X_train, y_train)

        # Predict
        y_pred = svm_classifier.predict(X_test)

        # Calculate Balanced Accuracy
        ba = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracies.append(ba)

    balanced_accuracy = np.mean(balanced_accuracies)

    return balanced_accuracy, num_selected_features


def evaluation(pop, popsize, mm, n_select, X2, Y0, u, alpha):
    """
    Calculate the fitness value (Balanced Accuracy and Fitness) for each individual in the population, and return the result matrix.
    :param pop: numpy.ndarray, population matrix, where each row represents an individual and each column represents a feature selected by that individual (1-based, 0 means not selected).
    :param popsize: int, size of the population, i.e., the number of individuals.
    :param mm: int, number of feature groups.
    :param X2: numpy.ndarray, dataset feature matrix, containing the features of all samples.
    :param Y0: numpy.ndarray, dataset labels, containing the labels of all samples.
    :param u: list of lists, list of feature set indices, where each sublist represents a feature group and contains the indices of all features in that group.
    :param alpha: float, weight parameter used to balance the impact of classification accuracy and the number of features.
    :return: numpy.ndarray
        Fitness matrix, where the first `sum(n_select)` columns represent the feature selection,
        the `sum(n_select)`-th column is the Balanced Accuracy of the individual,
        and the `sum(n_select) + 1`-th column is the Fitness of the individual.
    """
    total_feats = sum(n_select)

    # Initialize the fitness matrix, adding two columns to store Balanced Accuracy and Fitness respectively
    popeff = np.zeros((popsize, total_feats + 2))

    # Copy the population's feature selection to the fitness matrix
    popeff[:, :total_feats] = pop

    # Iterate through each individual to calculate Balanced Accuracy and Fitness
    for i in range(popsize):
        balanced_acc, num_selected_features = evaluate_individual(X2, Y0, mm, pop, u, i, n_select)
        norm_feat = num_selected_features / total_feats
        fitness = (1 - alpha) * (1-balanced_acc) + alpha * norm_feat
        popeff[i, total_feats] = balanced_acc * 100
        popeff[i, total_feats + 1] = fitness

    return popeff


def Svmevaluation(X_train, Y_test, sel_f):
    """
    SVM Classifier: Train and calculate the balanced accuracy on the test set.
    :param X_train: Training dataset, containing class labels in the first column.
    :param Y_test: Testing dataset, containing class labels in the first column.
    :param sel_f: Indices of the selected features.
    :return: balanced_accuracy: Balanced accuracy score on the test set.
    """
    # Extract class labels
    train_labels = X_train[:, 0]
    test_labels = Y_test[:, 0]

    # Extract feature data
    X_train_features = X_train[:, 1:]
    X_test_features = Y_test[:, 1:]

    # Select specific features using indices in sel_f
    X_train_selected = X_train_features[:, sel_f]
    X_test_selected = X_test_features[:, sel_f]

    # Normalize data to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_selected = scaler.fit_transform(X_train_selected)
    X_test_selected = scaler.transform(X_test_selected)

    # Define the SVM model with an RBF kernel
    model = svm.SVC(kernel='rbf')
    model.fit(X_train_selected, train_labels)

    # Predict and calculate Balanced Accuracy
    predictions = model.predict(X_test_selected)
    balanced_accuracy = balanced_accuracy_score(test_labels, predictions)

    return balanced_accuracy


def knn_jz(X_train, Y_test, sel_f):
    """
    KNN Classifier: Calculate the classification accuracy on the test set.
    :param X_train: Training dataset, containing class labels in the first column.
    :param Y_test: Testing dataset, containing class labels in the first column.
    :param sel_f: Indices of the selected features.
    :return: Tuple containing balanced classification accuracy and the number of selected features.
    """
    # Extract data for selected features
    train_labels = X_train[:, 0]
    test_labels = Y_test[:, 0]
    train_data = X_train[:, 1:]
    test_data = Y_test[:, 1:]
    train_data_selected = train_data[:, sel_f]
    test_data_selected = test_data[:, sel_f]

    # Normalize data to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_norm = scaler.fit_transform(train_data_selected)
    test_data_norm = scaler.transform(test_data_selected)

    # Define and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_data_norm, train_labels)

    # Predict and calculate classification accuracy
    predictions = knn.predict(test_data_norm)
    balanced_accuracy = balanced_accuracy_score(test_labels, predictions)

    # Return balanced accuracy and the count of selected features
    return balanced_accuracy, len(sel_f)


def lastkmean_res(X_train, Y_test, sel_f):
    """
    SVM Classifier: Train and calculate test set accuracy (Balanced Accuracy) and AUC.
    AUC is calculated using Logistic Regression fitting for probability distribution.

    :param X_train: Training dataset, including class labels in the first column.
    :param Y_test: Testing dataset, including class labels in the first column.
    :param sel_f: Indices of the selected features.
    :return: Balanced accuracy and mean AUC.
    """
    # Extract class labels
    train_labels = X_train[:, 0]
    test_labels = Y_test[:, 0]

    # Extract feature data
    X_train_features = X_train[:, 1:]
    X_test_features = Y_test[:, 1:]

    # Select specific features using sel_f
    X_train_selected = X_train_features[:, sel_f]
    X_test_selected = X_test_features[:, sel_f]

    # Normalize data to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_selected = scaler.fit_transform(X_train_selected)
    X_test_selected = scaler.transform(X_test_selected)

    # Define SVM model and its parameters
    model = svm.SVC(kernel='rbf', probability=True)
    model.fit(X_train_selected, train_labels)

    # Predict and calculate Balanced Accuracy
    predictions = model.predict(X_test_selected)
    balanced_accuracy = balanced_accuracy_score(test_labels, predictions)

    # Get the list of unique classes
    unique_classes = np.unique(train_labels)
    auc_scores = []

    # Calculate AUC per class (One-vs-All)
    for class_label in unique_classes:
        # Binarize current class labels: current class is 1, others are 0
        binarized_test_labels = (test_labels == class_label).astype(int)

        # Get SVM probability predictions for the test set
        probabilities = model.predict_proba(X_test_selected)[:, np.where(model.classes_ == class_label)[0][0]]

        # Skip category if the test set contains only one label (to avoid roc_auc_score error)
        if len(np.unique(binarized_test_labels)) < 2:
            auc = np.nan
        else:
            # Fit probability distribution using Logistic Regression
            lr_model = LogisticRegression(solver='lbfgs')
            lr_model.fit(probabilities.reshape(-1, 1), binarized_test_labels)

            # Calculate AUC using Logistic Regression probability output
            probabilities_lr = lr_model.predict_proba(probabilities.reshape(-1, 1))[:, 1]
            auc = roc_auc_score(binarized_test_labels, probabilities_lr)

        auc_scores.append(auc)

    # Calculate mean AUC (skipping NaN values)
    auc_mean = np.nanmean(auc_scores)

    return balanced_accuracy, auc_mean


def evaluation_last(X_train, Y_test, sel_f):
    """
    Calculate the classification accuracy, AUC, and the number of selected features for the selected feature subset using SVM and KNN models.
    :param X_train: Feature matrix of the training dataset, shape (n_samples, n_features), excluding labels.
    :param Y_test: Label vector of the test set, shape (n_samples,), containing the class labels for each sample.
    :param sel_f: Selected feature subset, containing the indices of the selected features.
    :return: ggbest - Array containing the selected features, SVM accuracy, KNN accuracy, number of selected features, and AUC.
    """
    ggbest = sel_f

    # SVM calculation
    SVMary, AUC = lastkmean_res(X_train, Y_test, sel_f)

    # KNN calculation
    KNNary, jie_sum = knn_jz(X_train, Y_test, sel_f)

    # Store the results in ggbest
    ggbest = np.append(ggbest, [100 * SVMary, 100 * KNNary, jie_sum, 100 * AUC])

    return ggbest