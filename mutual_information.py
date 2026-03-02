import numpy as np

def mutual_information(u1, u2):
    """
    Calculate the Normalized Mutual Information (NMI) between two arrays.
    Args:
        u1 (np.ndarray): The first input array.
        u2 (np.ndarray): The second input array.

    Returns:
        float: The Normalized Mutual Information (NMI) score.
    """
    wind_size = u1.shape[0]
    x = np.column_stack((u1, u2))
    n = wind_size
    bin_indices = np.zeros((n, 2), dtype=int)
    pmf = np.zeros((n, 2))

    for i in range(2):
        minx = np.min(x[:, i])
        maxx = np.max(x[:, i])
        binwidth = (maxx - minx) / n
        edges = minx + binwidth * np.arange(n + 1)
        histc_edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        bin_indices[:, i] = np.digitize(x[:, i], histc_edges) - 1  # Convert to 0-based indices
        occur = np.histogram(x[:, i], bins=histc_edges)[0]
        pmf[:, i] = occur[:n] / wind_size

    # Calculate the joint Probability Mass Function (PMF)
    joint_occur = np.zeros((n, n))
    for b in bin_indices:
        joint_occur[b[0], b[1]] += 1
    joint_pmf = joint_occur / wind_size

    # Calculate the individual information entropy
    Hx = -np.sum(pmf[:, 0] * np.log2(pmf[:, 0] + np.finfo(float).eps))
    Hy = -np.sum(pmf[:, 1] * np.log2(pmf[:, 1] + np.finfo(float).eps))
    Hxy = -np.sum(joint_pmf * np.log2(joint_pmf + np.finfo(float).eps))

    # Calculate Mutual Information (MI)
    MI = Hx + Hy - Hxy

    # Calculate Normalized Mutual Information (NMI)
    NMI = 2 * MI / (Hx + Hy + np.finfo(float).eps)

    return NMI