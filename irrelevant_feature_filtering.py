import numpy as np
from sklearn_extra.cluster import KMedoids
from mutual_information import mutual_information


def irrelevant_feature_filtering(X0, Y0, c0_all, kmedoids_model):
    """
    Filter and remove irrelevant features using KMedoids, incorporating min-max normalization.
    :param X0: ndarray, a batch of features read at each step
    :param Y0: ndarray, class labels
    :param c0_all: list, stores the metric values (mutual information) of all processed features
    :param kmedoids_model: KMedoids, the current KMedoids model, used for training or updating
    :return: Tuple containing the following elements:
        - c (list): Mutual information values of the selected features
        - f (list): Indices of the selected features
        - X (ndarray): Filtered feature matrix
        - c0_all (list): Updated metric values of all processed features
        - kmedoids_model (KMedoids): Updated clustering model
    """
    # Initialize variables
    kk, k = X0.shape
    c_0 = []
    f_0 = []
    c = []
    f = []

    # Calculate the NMI value between each feature and the class labels
    for i in range(k):
        mi_value = mutual_information(X0[:, i], Y0)
        f_0.append(i)  # Append feature index to f_0
        c_0.append(mi_value)  # Append MI value to c_0

    # Update the metric values of all processed features
    c0_all = np.concatenate((c0_all, c_0), axis=None)

    # Convert to numpy array and reshape into a column vector for subsequent processing
    c_0 = np.array(c_0).reshape(-1, 1)

    c_0_normalized = c_0

    # Determine if this is the initial clustering or subsequent processing
    if kmedoids_model is None:
        # === Initial clustering: perform clustering directly on all features ===
        kmedoids_model = KMedoids(
            n_clusters=3,
            init='k-medoids++',
            random_state=42,
        )
        # Train the KMedoids model
        kmedoids_model.fit(c_0_normalized)

        # Obtain cluster labels directly
        labels = kmedoids_model.labels_

        # Get cluster centers and find the most irrelevant cluster
        cluster_centers_normalized = kmedoids_model.cluster_centers_.flatten()
        min_center_idx = np.argmin(cluster_centers_normalized)

        # Filter features directly based on cluster labels
        for i in range(len(f_0)):
            if labels[i] != min_center_idx:  # Retain if not belonging to the most irrelevant cluster
                f.append(f_0[i])
                c.append(c_0[i][0])

    else:
        # === Subsequent processing: determine the cluster of new features by distance ===
        # Get existing cluster centers
        cluster_centers_normalized = kmedoids_model.cluster_centers_.flatten()

        # Find the most irrelevant cluster center (the cluster with the minimum center value)
        min_center_idx = np.argmin(cluster_centers_normalized)

        # Determine which cluster the new feature belongs to based on the distance to cluster centers
        for i in range(len(f_0)):
            # Get the normalized MI value of the current feature
            mi_value_normalized = c_0_normalized[i][0]

            # Calculate the absolute distance to all cluster centers
            distances = np.abs(cluster_centers_normalized - mi_value_normalized)

            # Find the cluster with the minimum distance
            min_distance_idx = np.argmin(distances)

            if min_distance_idx == min_center_idx:  # Discard if it belongs to the irrelevant cluster
                continue
            else:  # Otherwise, it belongs to a relevant cluster; retain the feature
                f.append(f_0[i])  # Retain relevant feature index
                c.append(c_0[i][0])  # Retain NMI value of the relevant feature

    # Construct the relevant feature matrix X based on the filtered features
    X = X0[:, f]

    return c, f, X, c0_all, kmedoids_model