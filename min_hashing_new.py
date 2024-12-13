import numpy as np
import random
import sympy
import json
import re
import os

# Function to calculate hash values
def compute_hash(a, b, row, prime):
    return (a * row + b) % prime

# Generate MinHash signature matrix
def generate_signature_matrix(binary_matrix, num_hashes):
    num_rows, num_cols = binary_matrix.shape
    prime = sympy.nextprime(num_rows)

    random.seed(42)
    a_values = np.random.randint(1, prime, size=num_hashes)
    b_values = np.random.randint(0, prime, size=num_hashes)

    signature_matrix = np.full((num_hashes, num_cols), np.iinfo(np.int32).max, dtype=np.int32)

    for row in range(num_rows):
        hash_values = [compute_hash(a_values[i], b_values[i], row, prime) for i in range(num_hashes)]
        for col in range(num_cols):
            if binary_matrix[row, col] == 1:
                signature_matrix[:, col] = np.minimum(signature_matrix[:, col], hash_values)

    return signature_matrix

if __name__ == "__main__":
    enhanced_case = True

    if enhanced_case:
        primary_dir = "blocked_binary_matrices/primary"
        secondary_dir = "blocked_binary_matrices/secondary"
        output_dir_primary = "signature_matrices/primary"
        output_dir_secondary = "signature_matrices/secondary"

        os.makedirs(output_dir_primary, exist_ok=True)
        os.makedirs(output_dir_secondary, exist_ok=True)

        # Process primary blocks
        print("\nProcessing Primary Blocks...")
        for file_name in os.listdir(primary_dir):
            block_path = os.path.join(primary_dir, file_name)
            if block_path.endswith(".npy"):
                binary_matrix = np.load(block_path)
                num_rows = binary_matrix.shape[0]
                num_hashes = max(1, num_rows // 2)  # Ensure at least 1 hash
                print(f"Processing primary block '{file_name}' with {num_rows} rows and {num_hashes} MinHashes.")

                signature_matrix = generate_signature_matrix(binary_matrix, num_hashes)
                output_file = os.path.join(output_dir_primary, f"signature_{file_name}")
                np.save(output_file, signature_matrix)
                print(f"Signature matrix saved for primary block '{file_name}'.")

        # Process secondary blocks
        print("\nProcessing Secondary Blocks...")
        for file_name in os.listdir(secondary_dir):
            block_path = os.path.join(secondary_dir, file_name)
            if block_path.endswith(".npy"):
                binary_matrix = np.load(block_path)
                num_rows = binary_matrix.shape[0]
                num_hashes = max(1, num_rows // 2)  # Ensure at least 1 hash
                print(f"Processing secondary block '{file_name}' with {num_rows} rows and {num_hashes} MinHashes.")

                signature_matrix = generate_signature_matrix(binary_matrix, num_hashes)
                output_file = os.path.join(output_dir_secondary, f"signature_{file_name}")
                np.save(output_file, signature_matrix)
                print(f"Signature matrix saved for secondary block '{file_name}'.")

    else:
        print("\nProcessing Base Case...")
        binary_matrix = np.load("binary_matrix.npy")
        num_rows = binary_matrix.shape[0]
        num_hashes = max(1, num_rows // 2)  # Ensure at least 1 hash
        print(f"Processing full matrix with {num_rows} rows and {num_hashes} MinHashes.")

        signature_matrix = generate_signature_matrix(binary_matrix, num_hashes)
        os.makedirs("signature_matrices", exist_ok=True)
        np.save("signature_matrices/signature_matrix.npy", signature_matrix)
        print("Signature matrix saved for full binary matrix.")
