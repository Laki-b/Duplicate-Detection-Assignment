import numpy as np
import hashlib
import os
import re
import json
from collections import defaultdict, Counter


# Extract words from title
def extract_title_words(title):
    predefined = {"led", "hdtv", "supersonic", "smart", "plasma", "lcd", "ledlcd", "hd tv"}
    model_words = set()
    pattern = r'([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)'
    matches = re.findall(pattern, title)
    model_words.update([match[0] for match in matches])
    title_words = set(title.split())
    model_words.update(predefined.intersection(title_words))
    return model_words

# Extract unique brand names
def extract_brands(products):
    return {prod.get("featuresMap", {}).get("Brand", "").lower() for prod in products if prod.get("featuresMap", {}).get("Brand", "").lower()}

# Extract feature words
def extract_feature_words(features):
    model_words = set()
    pattern = r'(^\d+(\.\d+)?[a-zA-Z]*$|^\d+(\.\d+)?$)'
    for value in features:
        matches = re.findall(pattern, value.lower())
        model_words.update([match[0] for match in matches])
    return model_words

# Merge small blocks
def merge_small_blocks(blocks, products, min_block_size=3):
    # Extract product attributes
    product_attributes = {
        prod["modelID"]: {
            "bigrams": set(zip(prod["title"].split()[-5:], prod["title"].split()[-4:])),
            "brand": prod.get("brand", "").lower(),
            "resolution": prod.get("featuresMap", {}).get("resolution", "").lower()
        }
        for prod in products
    }

    merged_blocks = defaultdict(list)
    fallback_block = []

    for block_key, block_products in blocks.items():
        if len(block_products) >= min_block_size:
            merged_blocks[block_key].extend(block_products)
        else:
            # Collect features of the small block
            block_features = Counter()
            for product_id in block_products:
                attributes = product_attributes.get(product_id, {})
                block_features.update(attributes["bigrams"])
                block_features.update([attributes["brand"], attributes["resolution"]])

            # Determine the best existing block to merge into
            best_match = None
            best_overlap = 0
            block_feature_keys = set(block_features.keys())  # Extract keys from Counter
            for key, existing_block in merged_blocks.items():
                overlap = sum(
                    1 for prod_id in existing_block
                    if len(block_feature_keys & set(
                        [item for sublist in product_attributes[prod_id].values() for item in sublist]
                    )) > 0
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = key

            # Merge or add to fallback
            if best_match and best_overlap > 0:
                merged_blocks[best_match].extend(block_products)
            else:
                fallback_block.extend(block_products)

    # Add fallback block if it contains products
    if fallback_block:
        merged_blocks["fallback"] = fallback_block

    return merged_blocks



# Create binary matrices for merged blocks
def create_blocked_binary_matrices(products, primary_blocks, secondary_blocks):
    os.makedirs("blocked_binary_matrices/primary", exist_ok=True)
    os.makedirs("blocked_binary_matrices/secondary", exist_ok=True)

    product_id_to_index = {prod["modelID"]: idx for idx, prod in enumerate(products)}
    binary_matrix = np.load("binary_matrix.npy")

    def save_blocks(blocks, output_dir):
        for block_key, block_products in blocks.items():
            cleaned_block_key = re.sub(r"[^\w\s-]", "", block_key).replace(" ", "_")
            product_indices = [product_id_to_index.get(prod, -1) for prod in block_products]
            product_indices = [idx for idx in product_indices if idx != -1]
            if product_indices:
                blocked_matrix = binary_matrix[:, product_indices]
                np.save(f"{output_dir}/{cleaned_block_key}.npy", blocked_matrix)

    save_blocks(primary_blocks, "blocked_binary_matrices/primary")
    save_blocks(secondary_blocks, "blocked_binary_matrices/secondary")

# Main function
if __name__ == "__main__":
    input_file = "cleaned_data.json"
    primary_blocks_file = "primary_blocks.json"
    secondary_blocks_file = "secondary_blocks.json"

    with open(input_file, "r") as f:
        products = json.load(f)

    with open(primary_blocks_file, "r") as f:
        primary_blocks = json.load(f)

    with open(secondary_blocks_file, "r") as f:
        secondary_blocks = json.load(f)

    # Merge small blocks
    primary_blocks = merge_small_blocks(primary_blocks, products, min_block_size=3)
    secondary_blocks = merge_small_blocks(secondary_blocks, products, min_block_size=3)

    # Save merged blocks
    with open("merged_primary_blocks.json", "w") as f:
        json.dump(primary_blocks, f, indent=4)
    with open("merged_secondary_blocks.json", "w") as f:
        json.dump(secondary_blocks, f, indent=4)
    
    

    # Create binary matrices
    create_blocked_binary_matrices(products, primary_blocks, secondary_blocks)
    print("Binary matrices for merged blocks created.")
