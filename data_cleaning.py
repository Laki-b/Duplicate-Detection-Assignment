import json
import re

# Load the JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Ensure data is a dictionary and extract the list of products
    if isinstance(data, dict):
        # Flatten all values into a list (assuming dictionary values are lists of products)
        products = []
        for value in data.values():
            if isinstance(value, list):  # Add only if it's a list
                products.extend(value)
        return products
    return []

# Cleaning function to remove special characters
def clean_special_characters(text):
    return re.sub(r'[^\w\s]', '', text)  # Replace special characters with an empty string

# Tokenize text into a set of unique words
def tokenize_text(text):
    return set(re.findall(r'\b\w+\b', text))  # Matches alphanumeric words

# Infer brand from title using a predefined list of known brands
def infer_brand_from_title(title, known_brands):
    title_tokens = set(re.findall(r'\b\w+\b', title.lower()))
    for brand in known_brands:
        if brand in title_tokens:
            return brand
    return "unknown"

# Data cleaning function
def clean_product_data(data, known_brands):
    cleaned_data = []
    
    # Define variations for "wifi"
    wifi_variations = [r'wifi', r'wi-fi', r'wifi ready', r'wifi built-in', r'built-in wifi', r'wi-fi built-in']
    # Define variations for "inch", "hertz", and "pounds"
    inch_variations = [r'["”]', r'inch', r'inches', r'-inch', r' inch']
    hertz_variations = [r'hz', r'hertz', r'-hz', r' hz']
    pounds_variations = [r'pounds', r' pounds', r'lb', r' lbs', r'lbs.']
    
    for product in data:
        # Clean and normalize the title
        title = product.get('title', '').lower()
        title = clean_special_characters(title)

        # Normalize "inch", "hertz", and "wifi"
        for var in inch_variations:
            title = re.sub(var, 'inch', title)
        for var in hertz_variations:
            title = re.sub(var, 'hz', title)
        for var in wifi_variations:
            title = re.sub(var, 'wifi', title)

        # Tokenize the cleaned title for blocking and other tasks
        title_tokens = tokenize_text(title)

        # Infer Brand
        brand = product.get('featuresMap', {}).get('Brand', 'unknown').lower()
        if brand == "unknown":
            brand = infer_brand_from_title(title, known_brands)

        # Process the feature map
        features_map = product.get('featuresMap', {})
        cleaned_features_map = {}
        for key, value in features_map.items():
            # Convert key to lowercase
            clean_key = key.lower()
            
            if isinstance(value, str):
                # Convert value to lowercase
                value = value.lower()
                
                # Normalize "inch", "hertz", "pounds", and "wifi"
                for var in inch_variations:
                    value = re.sub(var, 'inch', value)
                for var in hertz_variations:
                    value = re.sub(var, 'hz', value)
                for var in pounds_variations:
                    value = re.sub(var, 'lbs', value)
                for var in wifi_variations:
                    value = re.sub(var, 'wifi', value)
            
            # Add the cleaned key-value pair to the new feature map
            cleaned_features_map[clean_key] = value

        # Sanitize the modelID for consistency and safe use in filenames
        model_id = product.get("modelID", "unknown")
        sanitized_model_id = re.sub(r'[^\w\-_]', '_', model_id)  # Replace invalid characters with '_'

        # Add cleaned data to the output list
        cleaned_data.append({
            "modelID": sanitized_model_id,  # Sanitized modelID
            "title": title,  # Cleaned and normalized title
            "title_tokens": list(title_tokens),  # Tokenized title for blocking and matching
            "brand": brand,  # Inferred or extracted brand
            "featuresMap": cleaned_features_map,  # Cleaned features map
            "shop": product.get('shop', '').lower()  # Convert shop names to lowercase
        })
    
    return cleaned_data


# Save cleaned data to a JSON file
def save_cleaned_data(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

# Main execution
if __name__ == "__main__":
    file_path = "TVs-all-merged.json"  # Input JSON file
    output_path = "cleaned_data.json"  # Output JSON file

    # Predefined list of known brands
    known_brands = {"samsung", "sony", "lg", "panasonic", "sharp", "philips", "toshiba", "vizio", "hisense", "tcl", "vu", "walton" , "akai", "xiaomi", "arise", "itel", "jvc", 
                    "tp vision", "arcam", "micromax", "seiki", "element", "kogan", "duraband", "jensen", "westinghouse", "google", "vizio", "apple", "fujitsu", "tatung",
                    "marantz", "skyworth", "proscan", "onida", "sansui", "haier", "konka", "planar" , "funai", "vestel", "videocon", "hitachi", "memorex", "sanyo", "salora", "zenith",
                    "thomson", "alba" , "bush" , "loewe", "telefunken", "metz", "pensonic" , "rediffusion", "saba", "tpv", "magnavox", "bang", "cge", "changhong", "compal", "curtis", "finlux"}

    # Load, clean, and save the data
    raw_data = load_json(file_path)
    print(f"Loaded data type: {type(raw_data)}, number of products: {len(raw_data)}")  # Debug print
    cleaned_data = clean_product_data(raw_data, known_brands)
    save_cleaned_data(cleaned_data, output_path)

    print(f"Cleaned data saved to {output_path}")

