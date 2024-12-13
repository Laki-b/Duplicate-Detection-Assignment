import json
from collections import defaultdict
import itertools
from typing import Dict, List

# Resolution keys to look for in the features map
RESOLUTION_KEYS = ["recommended resolution", "resolution", "native resolution", "vertical resolution"]

# Keywords to look for in the title
KEYWORDS = {"led", "lcd", "ledlcd", "plasma"}

def extract_resolution(features: Dict[str, str]) -> str:
    """Extract resolution from the features map."""
    for key in RESOLUTION_KEYS:
        if key in features:
            return features[key]
    # Fallback: Look for any key containing 'resolution'
    for key in features:
        if "resolution" in key.lower():
            return features[key]
    return "unknown"

def create_primary_blocks(data: List[Dict]) -> Dict[str, List[str]]:
    """Create primary blocks based on brand, keywords, and resolution."""
    primary_blocks = defaultdict(list)

    for product in data:
        product_id = product.get("modelID")
        brand = product.get("brand", "unknown").lower()
        features = product.get("featuresMap", {})
        title = product.get("title", "").lower()

        # Extract resolution
        resolution = extract_resolution(features).lower()

        # Check for keywords in the title
        title_tokens = set(product.get("title_tokens", []))
        keyword = next((kw for kw in KEYWORDS if kw in title_tokens), "unknown")

        if brand != "unknown" and keyword != "unknown" and resolution != "unknown":
            block_key = f"{brand}-{keyword}-{resolution}"
            primary_blocks[block_key].append(product_id)

    return primary_blocks

def generate_bi_grams(tokens: List[str]) -> List[str]:
    """Generate bi-grams from a list of tokens."""
    return [' '.join(pair) for pair in zip(tokens, tokens[1:])]

def create_secondary_blocks(data: List[Dict], primary_blocks: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Create secondary blocks using bi-grams for products not in primary blocks."""
    secondary_blocks = defaultdict(list)

    # Get all products already in primary blocks
    all_primary_ids = set(itertools.chain.from_iterable(primary_blocks.values()))

    for product in data:
        product_id = product.get("modelID")
        if product_id in all_primary_ids:
            continue

        title_tokens = product.get("title_tokens", [])
        bi_grams = generate_bi_grams(title_tokens[-5:])  # Use last 5 tokens for bi-grams

        for bi_gram in bi_grams:
            secondary_blocks[bi_gram].append(product_id)

    return secondary_blocks

def main(input_file="cleaned_data.json", primary_output="primary_blocks.json", secondary_output="secondary_blocks.json"):
    """Main function to generate primary and secondary blocks."""
    with open(input_file, "r") as f:
        data = json.load(f)


    # Create primary blocks
    primary_blocks = create_primary_blocks(data)
    with open(primary_output, "w") as f:
        json.dump(primary_blocks, f, indent=4)
    print(f"Primary blocks saved to {primary_output}")

    # Create secondary blocks
    secondary_blocks = create_secondary_blocks(data, primary_blocks)
    with open(secondary_output, "w") as f:
        json.dump(secondary_blocks, f, indent=4)
    print(f"Secondary blocks saved to {secondary_output}")

if __name__ == "__main__":
    main()
