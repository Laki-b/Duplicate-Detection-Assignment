import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform



def jaccard_similarity(binary_matrix, candidate_pairs):
    """Compute Jaccard dissimilarity for candidate pairs."""
    if not candidate_pairs:
        print("No candidate pairs found. Skipping clustering.")
        return []
    
    jaccard_distances = {}
    for pair in candidate_pairs:
        i, j = pair
        intersection = np.sum(np.logical_and(binary_matrix[:, i], binary_matrix[:, j]))
        union = np.sum(np.logical_or(binary_matrix[:, i], binary_matrix[:, j]))
        if union == 0:  # Handle division by zero
            jaccard_distances[pair] = 1  # Maximum dissimilarity
        else:
            jaccard_distances[pair] = 1 - (intersection / union)  # Dissimilarity
    return jaccard_distances



def perform_clustering(jaccard_distances, threshold):
    # Early exit if no candidate pairs
    if not jaccard_distances:
        print("No candidate pairs found. Skipping clustering.")
        return []
    
    items = list({i for pair in jaccard_distances.keys() for i in pair})
    item_index = {item: idx for idx, item in enumerate(items)}
    n = len(items)

    distance_matrix = np.full((n, n), np.inf)
    for (i, j), dist in jaccard_distances.items():
        if not np.isfinite(dist):  # Check for non-finite distances
            print(f"Non-finite Jaccard distance found for pair ({i}, {j}). Skipping.")
            continue
        idx_i, idx_j = item_index[i], item_index[j]
        distance_matrix[idx_i, idx_j] = dist
        distance_matrix[idx_j, idx_i] = dist

    # Ensure diagonal is zero
    np.fill_diagonal(distance_matrix, 0)

    # Check for any non-finite values in the matrix
    if not np.all(np.isfinite(distance_matrix)):
        print("Distance matrix contains non-finite values. Aborting clustering.")
        return []

    # Convert to condensed distance matrix
    condensed_matrix = squareform(distance_matrix)

    # Perform clustering
    clusters = linkage(condensed_matrix, method='complete')
    cluster_labels = fcluster(clusters, threshold, criterion='distance')

    # Group items into clusters
    predicted_clusters = {}
    for item, label in zip(items, cluster_labels):
        if label not in predicted_clusters:
            predicted_clusters[label] = []
        predicted_clusters[label].append(item)

    return list(predicted_clusters.values())
