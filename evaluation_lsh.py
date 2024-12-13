def evaluate_lsh(candidate_pairs, ground_truth_pairs, total_possible_comparisons):
    """Evaluate LSH performance using Pair Quality, Pair Completeness, F1*, and Fraction of Comparisons."""
    candidate_pairs_set = set(candidate_pairs)
    true_positives = len(candidate_pairs_set & ground_truth_pairs)
    false_positives = len(candidate_pairs_set - ground_truth_pairs)
    false_negatives = len(ground_truth_pairs - candidate_pairs_set)

    pair_quality = true_positives / len(candidate_pairs) if candidate_pairs else 0
    pair_completeness = true_positives / len(ground_truth_pairs) if ground_truth_pairs else 0
    f1_star = (2 * pair_quality * pair_completeness) / (pair_quality + pair_completeness) if (pair_quality + pair_completeness) > 0 else 0

    # Fraction of comparisons (scalability metric)
    fraction_comparisons = len(candidate_pairs) / total_possible_comparisons if total_possible_comparisons > 0 else 0

    return pair_quality, pair_completeness, f1_star, fraction_comparisons
