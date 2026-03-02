import numpy as np
import math
from mutual_information import mutual_information
from sklearn.cluster import AffinityPropagation


def affinity_propagation(k, X0, c, f, damping=0.7):
    """
    Use Affinity Propagation (AP) clustering to group strongly correlated features based on NMI.

    :param k: int, number of features
    :param X0: ndarray, feature matrix where each column corresponds to a feature
    :param c: list, list of NMI values between each feature and the label
    :param f: list, list of feature indices used to select corresponding columns from X0
    :param damping: float, damping factor for Affinity Propagation
    :return: Tuple containing:
        - u (list of lists): Feature groups, each containing original indices (from the f list) of highly correlated features
        - C (list of lists): List of NMI values (feature-to-label) for each group
        - mm (int): Number of groups
    """
    # Calculate the NMI matrix between features
    mi_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            mi_val = mutual_information(X0[:, f[i]], X0[:, f[j]])
            mi_matrix[i, j] = mi_val
            mi_matrix[j, i] = mi_val

    # Convert to distance matrix: 1 - NMI represents dissimilarity; negated to act as similarity
    distance_matrix = -(1 - mi_matrix)

    # Set the diagonal to the minimum value of the distance matrix (off-diagonal elements)
    off_diagonal_values = distance_matrix[np.triu_indices(k, 1)]
    min_val = np.min(off_diagonal_values)
    np.fill_diagonal(distance_matrix, min_val)
    # np.fill_diagonal(distance_matrix, 1)

    # Perform Affinity Propagation clustering using the precomputed similarity matrix
    ap = AffinityPropagation(affinity='precomputed', damping=damping, random_state=42)
    ap.fit(distance_matrix)

    # Get the category each feature belongs to based on ap.labels_
    n_clusters = len(ap.cluster_centers_indices_)
    clusters = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(ap.labels_):
        clusters[label].append(idx)

    # Map clustering results back to original feature indices and collect corresponding NMI values
    u = []
    C = []
    for cluster in clusters:
        group_features = [f[i] for i in cluster]
        group_c = [c[i] for i in cluster]
        u.append(group_features)
        C.append(group_c)

    mm = len(u)
    return u, C, mm


def incremental_clustering(k, X1, X, c, c0_all, f, start_col, feature_cluster_map, beta=0.7):

    # Initialize the number of feature clusters
    mm = len(feature_cluster_map)

    # Initialize the feature cluster list u and the corresponding NMI value list C
    u = [[] for _ in range(mm)]
    C = [[] for _ in range(mm)]

    # Populate initial feature clusters
    for cluster_id, rep_features in feature_cluster_map.items():
        for feature in rep_features:
            u[cluster_id - 1].append(feature)
            C[cluster_id - 1].append(c0_all[feature])  # NMI with class label y

    # Track indices of newly added clusters
    m = []

    # Cache for inter-cluster distance calculation results
    cluster_distances = {}

    # Pre-compute and cache distances between all pairs of existing clusters (only calculated once at the beginning)
    def compute_cluster_distance(cluster_id1, cluster_id2):
        """Calculate the distance between two clusters and cache the result."""
        if cluster_id1 > cluster_id2:
            cluster_id1, cluster_id2 = cluster_id2, cluster_id1

        # Return the cached result if already calculated
        key = (cluster_id1, cluster_id2)
        if key in cluster_distances:
            return cluster_distances[key]

        rep_features1 = feature_cluster_map[cluster_id1]
        rep_features2 = feature_cluster_map[cluster_id2]

        # Calculate MI between representative features of different clusters, converting it to distance
        distances = []
        for f1 in rep_features1:
            for f2 in rep_features2:
                distances.append(1 - mutual_information(X1[:, f1], X1[:, f2]))

        # Calculate and cache the mean distance
        mean_distance = np.mean(distances)
        cluster_distances[key] = mean_distance
        return mean_distance

    # Initialize and calculate distances between all pairs of clusters
    cluster_ids = list(feature_cluster_map.keys())
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            compute_cluster_distance(cluster_ids[i], cluster_ids[j])

    # Process each new feature
    for i in range(k):
        # Calculate NMI between the new feature and the representative features of each cluster, convert to distance, and find the minimum
        min_distance = np.inf
        closest_cluster = -1

        # Calculate the distance between the new feature and each cluster
        for cluster_id, rep_features in feature_cluster_map.items():
            # Calculate NMI between the new feature and the cluster's representative features, convert to distance
            distances = [1 - mutual_information(X[:, i], X1[:, feature]) for feature in rep_features]
            mean_cluster_distance = np.mean(distances)

            # Update the minimum distance and the closest cluster
            if mean_cluster_distance < min_distance:
                min_distance = mean_cluster_distance
                closest_cluster = cluster_id

        # Retrieve the maximum inter-cluster distance from the cache to use as a threshold
        max_cluster_distance = max(cluster_distances.values())

        threshold = max_cluster_distance

        # If the minimum distance is greater than the threshold, create a new cluster
        if min_distance > threshold:
            new_feature = f[i] + start_col
            u.append([new_feature])
            C.append([c[i]])
            mm += 1
            feature_cluster_map[mm] = [new_feature]
            m.append(mm)

        else:
            # Assign the new feature to the closest cluster
            new_feature = f[i] + start_col
            u[closest_cluster - 1].append(new_feature)
            C[closest_cluster - 1].append(c[i])

        # Update representative features for newly added feature clusters
        for cluster_id in m:
            features = u[cluster_id - 1]
            mi_with_y = C[cluster_id - 1]
            n = int(math.floor(math.sqrt(len(features))))
            # Sort features in descending order of NMI value
            sorted_indices = np.argsort(-np.array(mi_with_y))  # Indices for descending sort
            # Select the top n features with the largest NMI values as representative features
            top_n_features = [features[idx] for idx in sorted_indices[:max(1, n)]]
            feature_cluster_map[cluster_id] = top_n_features

            # When representative features change:
            for other_cluster in feature_cluster_map.keys():
                if other_cluster != cluster_id:
                    key = (min(cluster_id, other_cluster), max(cluster_id, other_cluster))
                    if key in cluster_distances:
                        del cluster_distances[key]
                    # Recalculate new inter-cluster distances
                    compute_cluster_distance(cluster_id, other_cluster)

    # Cluster merging phase
    cluster_ids = sorted(feature_cluster_map.keys())
    all_distances = []
    pair_list = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            cid1, cid2 = cluster_ids[i], cluster_ids[j]
            key = (min(cid1, cid2), max(cid1, cid2))
            dist_val = cluster_distances[key]
            all_distances.append(dist_val)
            pair_list.append((cid1, cid2, dist_val))

    if len(all_distances) == 0:
        return u, C, mm, feature_cluster_map

    mean_distance = np.mean(all_distances)
    # std_distance = np.std(all_distances)
    merge_threshold = mean_distance * beta

    # Graph construction: connect edges only when distance is less than the threshold
    adjacency = {cid: set() for cid in cluster_ids}
    for cid1, cid2, dist_val in pair_list:
        if dist_val < merge_threshold:
            adjacency[cid1].add(cid2)
            adjacency[cid2].add(cid1)

    # DFS to find connected components (clusters)
    visited = set()
    components = []

    def dfs(start, comp):
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

    for cid in cluster_ids:
        if cid not in visited:
            comp = []
            dfs(cid, comp)
            components.append(comp)

    if len(components) == len(cluster_ids):
        # No merging occurred
        return u, C, mm, feature_cluster_map

    # Merge clusters
    new_u = []
    new_C = []
    new_feature_cluster_map = {}
    new_cluster_id = 1

    print("Performing cluster merging")
    for comp in components:
        merged_features = []
        merged_mi = []
        merged_rep_features = []  # Used to store the merged representative features

        # Check if it is a single cluster (no merging)
        if len(comp) == 1:
            # Single cluster case: keep as is
            old_cid = comp[0]
            merged_features = u[old_cid - 1]
            merged_mi = C[old_cid - 1]
            merged_rep_features = feature_cluster_map[old_cid]
        else:
            # Multi-cluster merge case
            for old_cid in comp:
                merged_features.extend(u[old_cid - 1])
                merged_mi.extend(C[old_cid - 1])
                # Merge all original representative features
                merged_rep_features.extend(feature_cluster_map[old_cid])

        new_u.append(merged_features)
        new_C.append(merged_mi)
        new_feature_cluster_map[new_cluster_id] = merged_rep_features
        new_cluster_id += 1

    new_mm = len(components)
    return new_u, new_C, new_mm, new_feature_cluster_map