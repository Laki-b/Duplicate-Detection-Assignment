import numpy as np
from itertools import combinations
import hashlib

def robust_hash(value):
    """Generate a robust hash for a given value."""
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16)

def split_signature_into_bands(signature_matrix, r, b):
    """
    Split the signature matrix into bands. Dynamically adjust r and b to fit the number of rows.
    """
    num_rows, num_cols = signature_matrix.shape

    if num_rows != r * b:
        print(f"Adjusting signature matrix: num_rows ({num_rows}) != r * b ({r * b}).")
        target_rows = r * b
        if num_rows < target_rows:
            # Pad with zero rows
            padding = np.zeros((target_rows - num_rows, num_cols))
            signature_matrix = np.vstack([signature_matrix, padding])
        else:
            # Truncate extra rows
            signature_matrix = signature_matrix[:target_rows, :]

    return [signature_matrix[band_idx * r:(band_idx + 1) * r, :] for band_idx in range(b)]

def lsh(signature_matrix, r, b):
    """
    Perform LSH and generate candidate pairs.
    """
    try:
        bands = split_signature_into_bands(signature_matrix, r, b)
    except Exception as e:
        print(f"Error during band splitting: {e}")
        return set()  # Return empty set if an error occurs

    num_cols = signature_matrix.shape[1]
    buckets = [{} for _ in range(b)]
    candidate_pairs = set()

    for band_idx, band in enumerate(bands):
        for col_idx in range(num_cols):
            # Use a robust hash of the tuple
            band_signature = robust_hash(tuple(band[:, col_idx]))
            if band_signature not in buckets[band_idx]:
                buckets[band_idx][band_signature] = []
            buckets[band_idx][band_signature].append(col_idx)

        # Log bucket distribution
        print(f"Band {band_idx}: {len(buckets[band_idx])} unique buckets.")

    for bucket in buckets:
        for products in bucket.values():
            if len(products) > 1:
                candidate_pairs.update(combinations(products, 2))

    print(f"Generated {len(candidate_pairs)} candidate pairs.")
    return candidate_pairs
