import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from lsh import lsh
from clustering import jaccard_similarity, perform_clustering
from evaluation_lsh import evaluate_lsh

def evaluate_final_clusters(predicted_clusters, ground_truth_pairs):
    """
    Evaluate the performance of clustering using precision, recall, and F1 score.

    Parameters:
        predicted_clusters (list of lists): Each sublist contains indices of items in a cluster.
        ground_truth_pairs (set of tuples): The ground truth duplicate pairs.

    Returns:
        final_f1: The F1 score for the clustering step.
    """
    predicted_pairs = set()

    # Generate pairs from clusters
    for cluster in predicted_clusters:
        cluster = list(map(int, cluster))  # Ensure cluster elements are integers
        if len(cluster) > 1:
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    predicted_pairs.add(tuple(sorted((cluster[i], cluster[j]))))

    true_positives = len(predicted_pairs & ground_truth_pairs)
    false_positives = len(predicted_pairs - ground_truth_pairs)
    false_negatives = len(ground_truth_pairs - predicted_pairs)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    final_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return final_f1

def generate_ground_truth_pairs(products):
    """Generate ground truth pairs based on duplicate modelIDs."""
    ground_truth_pairs = set()
    model_to_indices = {}

    for index, product in enumerate(products):
        model_id = product.get("modelID", None)
        if model_id:
            if model_id not in model_to_indices:
                model_to_indices[model_id] = []
            model_to_indices[model_id].append(index)

    for indices in model_to_indices.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    ground_truth_pairs.add((indices[i], indices[j]))

    return ground_truth_pairs

def bootstrap_and_evaluate(signature_matrix, ground_truth_pairs, r_values, b_values, num_bootstraps, thresholds):
    """Perform bootstrapping to evaluate LSH and clustering, tuning r, b, and threshold."""
    results = []
    if signature_matrix.shape[1] == 0:
        print("Signature matrix has no columns. Skipping evaluation.")
        return results  # Skip if there are no products in the signature matrix

    num_rows = signature_matrix.shape[0]
    total_possible_comparisons = signature_matrix.shape[1] * (signature_matrix.shape[1] - 1) / 2  # Total comparisons

    for r in r_values:
        for b in b_values:
            if r * b > signature_matrix.shape[0]:
                print(f"Skipping r={r}, b={b} due to invalid dimensions.")
                continue

            for threshold in thresholds:
                bootstrap_results = {
                    "r": r, "b": b, "threshold": threshold,
                    "pair_quality": [], "pair_completeness": [], "f1_star": [], "fraction_comparisons": [], "final_f1": []
                }

                for _ in tqdm(range(num_bootstraps), desc=f"Bootstrap r={r}, b={b}, threshold={threshold:.2f}", leave=False):
                    indices = list(range(signature_matrix.shape[1]))
                    train_indices = resample(
                        indices, 
                        replace=True, 
                        n_samples=max(1, int(0.63 * len(indices)))  # Ensure n_samples is at least 1
                    )
                    test_indices = [i for i in indices if i not in train_indices]
                    if not test_indices:
                        test_indices = [train_indices.pop()]

                    train_matrix = signature_matrix[:, train_indices]
                    test_matrix = signature_matrix[:, test_indices]

                    train_ground_truth = {(i, j) for i, j in ground_truth_pairs if i in train_indices and j in train_indices}
                    test_ground_truth = {(i, j) for i, j in ground_truth_pairs if i in test_indices and j in test_indices}

                    if not train_ground_truth or not test_ground_truth:
                        print("Empty ground truth in training or testing. Skipping.")
                        continue

                    candidate_pairs = lsh(train_matrix, r, b)
                    if not candidate_pairs:
                        print(f"No candidate pairs generated for r={r}, b={b}. Skipping.")
                        continue

                    pair_quality, pair_completeness, f1_star, fraction_comparisons = evaluate_lsh(
                        candidate_pairs, test_ground_truth, total_possible_comparisons
                    )

                    jaccard_distances = jaccard_similarity(train_matrix, candidate_pairs)
                    predicted_clusters = perform_clustering(jaccard_distances, threshold)

                    if not predicted_clusters:
                        print(f"No predicted clusters for r={r}, b={b}, threshold={threshold}. Skipping.")
                        continue

                    final_f1 = evaluate_final_clusters(predicted_clusters, train_ground_truth)

                    bootstrap_results["pair_quality"].append(pair_quality)
                    bootstrap_results["pair_completeness"].append(pair_completeness)
                    bootstrap_results["f1_star"].append(f1_star)
                    bootstrap_results["fraction_comparisons"].append(fraction_comparisons)
                    bootstrap_results["final_f1"].append(final_f1)

                # Store average results for this combination
                if bootstrap_results["pair_quality"]:
                    results.append({
                        "r": r,
                        "b": b,
                        "threshold": threshold,
                        "fraction_of_comparisons": np.mean(bootstrap_results["fraction_comparisons"]),
                        "avg_pair_quality": np.mean(bootstrap_results["pair_quality"]),
                        "avg_pair_completeness": np.mean(bootstrap_results["pair_completeness"]),
                        "avg_f1_star": np.mean(bootstrap_results["f1_star"]),
                        "avg_final_f1": np.mean(bootstrap_results["final_f1"]),
                    })

    return results

if __name__ == "__main__":
    with open("cleaned_data.json", "r") as f:
        products = json.load(f)

    ground_truth_pairs = generate_ground_truth_pairs(products)
    print(f"Generated {len(ground_truth_pairs)} ground truth pairs.")

    r_values = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 200]
    b_values = [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 80, 100, 200]
    thresholds = [0.5, 0.7, 0.6]
    num_bootstraps = 10

    print("\nProcessing Enhanced Case...")
    enhanced_results = []

    for block_type in ["primary", "secondary"]:
        block_dir = f"signature_matrices/{block_type}"
        for file_name in tqdm(os.listdir(block_dir), desc=f"Processing {block_type.capitalize()} Blocks"):
            if file_name.endswith(".npy"):
                signature_matrix = np.load(os.path.join(block_dir, file_name))
                if signature_matrix.shape[1] == 0:
                    print(f"Signature matrix {file_name} has no columns. Skipping.")
                    continue
                results = bootstrap_and_evaluate(signature_matrix, ground_truth_pairs, r_values, b_values, num_bootstraps, thresholds)
                enhanced_results.extend(results)

    print("\nAll Results for Enhanced Case:")
    for res in enhanced_results:
        print(res)

    if enhanced_results:
        best_enhanced_result = max(enhanced_results, key=lambda x: x["avg_f1_star"])
        print("\nBest Results for Enhanced Case:")
        print(best_enhanced_result)
