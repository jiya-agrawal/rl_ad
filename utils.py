def parse_train_data(file_path, batch_size=1024):
    """
    A generator-based parser for the Criteo training dataset.
    
    Args:
        file_path (str): Path to the dataset file (.txt).
        batch_size (int): Number of examples per mini-batch.
    
    Yields:
        list: A list of dictionaries containing parsed examples for a mini-batch.
    """
    def parse_header_line(line):
        """
        Parses the header line of an example.
        
        Args:
            line (str): A single header line from the dataset.
        
        Returns:
            dict: Parsed header information (example ID, label, propensity, display features).
        """
        parts = line.strip().split()
        example_id = parts[0]
        label = None
        propensity = None
        display_features = {}

        i = 1  # Start from index 1 (skip example ID)
        while i < len(parts):
            part = parts[i]
            if part.startswith("|l"):
                label = float(parts[i + 1])  # Extract label
                i += 2
            elif part.startswith("|p"):
                propensity = float(parts[i + 1].split("|")[0])  # Extract propensity
                i += 2
            elif ":" in part:
                # Parse display features
                for feat in parts[i:]:
                    if ":" not in feat:
                        break  # Stop parsing features
                    key, value = feat.split(":")
                    display_features[int(key)] = float(value)
                break  # Exit after parsing features
            else:
                i += 1
        
        return {
            "example_id": example_id,
            "label": label,
            "propensity": propensity,
            "display_features": display_features,
            "product_features": []  # Placeholder for product features
        }
    
    def parse_product_line(line):
        """
        Parses a product line of an example.
        
        Args:
            line (str): A single product line from the dataset.
        
        Returns:
            dict: Parsed product features.
        """
        parts = line.strip().split()
        product_features = {}
        for feat in parts[2:]:  # Skip example ID and "|f"
            key, value = feat.split(":")
            product_features[int(key)] = float(value)
        return product_features
    
    def read_file():
        """Generator to read the file line by line."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line
    
    current_batch = []
    header = None
    expected_products = 0
    
    for line in read_file():
        if line.startswith("|"):
            continue  # Skip malformed lines
        
        parts = line.strip().split()
        if not parts:
            continue  # Skip empty lines
        
        example_id = parts[0]
        
        if "|l" in line and "|p" in line:
            # Process previous example if incomplete
            if header is not None:
                # Incomplete example, finalize it
                header["nb_candidates"] = len(header["product_features"])
                current_batch.append(header)
                if len(current_batch) == batch_size:
                    yield current_batch
                    current_batch = []
            
            # Start a new example
            header = parse_header_line(line)
            expected_products = int(header["display_features"].get(1, 0))  # Get candidate pool size
        
        elif header is not None and example_id == header["example_id"]:
            # Parse product line
            product = parse_product_line(line)
            header["product_features"].append(product)
            expected_products -= 1
            
            # If all products are collected, finalize the example
            if expected_products == 0:
                header["nb_candidates"] = len(header["product_features"])
                current_batch.append(header)
                header = None
                
                # Yield batch if full
                if len(current_batch) == batch_size:
                    yield current_batch
                    current_batch = []
    
    # Finalize any remaining examples
    if header is not None:
        header["nb_candidates"] = len(header["product_features"])
        current_batch.append(header)
    
    if current_batch:
        yield current_batch


def parse_test_data(file_path, batch_size=1024):
    """
    A generator-based parser for the Criteo test dataset.
    
    Args:
        file_path (str): Path to the dataset file (.txt).
        batch_size (int): Number of examples per mini-batch.
    
    Yields:
        list: A list of dictionaries containing parsed examples for a mini-batch.
    """
    def parse_product_line(line):
        """
        Parses a product line of an example.
        
        Args:
            line (str): A single product line from the dataset.
        
        Returns:
            dict: Parsed product features.
        """
        parts = line.strip().split()
        example_id = parts[0]
        product_features = {}
        for feat in parts[2:]:  # Skip example ID and "|f"
            key, value = feat.split(":")
            product_features[int(key)] = float(value)
        return example_id, product_features
    
    def read_file():
        """Generator to read the file line by line."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line
    
    current_batch = []
    current_example = None
    current_example_id = None
    
    for line in read_file():
        if line.startswith("|"):
            continue  # Skip malformed lines
        
        parts = line.strip().split()
        if not parts:
            continue  # Skip empty lines
        
        example_id, product_features = parse_product_line(line)
        
        # Start a new example if the example_id changes
        if example_id != current_example_id:
            if current_example is not None:
                current_batch.append(current_example)
                if len(current_batch) == batch_size:
                    yield current_batch
                    current_batch = []
            
            # Initialize a new example
            current_example = {
                "example_id": example_id,
                "product_features": [product_features]
            }
            current_example_id = example_id
        else:
            # Append product features to the current example
            current_example["product_features"].append(product_features)
    
    # Finalize any remaining examples
    if current_example is not None:
        current_batch.append(current_example)
    
    if current_batch:
        yield current_batch

if __name__ == "__main__":

    # Path to the dataset file
    train_data_path = "criteo_train_small.txt/criteo_train_small.txt"

    # Parse training data
    train_parser = parse_train_data(train_data_path, batch_size=1024)

    # Iterate through mini-batches
    for batch in train_parser:
        print(f"Processed batch with {len(batch)} examples")
        for example in batch:
            print(f"Example ID: {example['example_id']}")
            print(f"Label: {example['label']}")
            print(f"Propensity: {example['propensity']}")
            print(f"Display Features: {example['display_features']}")
            print(f"Number of Candidates: {example['nb_candidates']}")
            print("Product Features:")
            for product in example['product_features']:
                print(f"  {product}")
            print("---")

    # Path to the dataset file
    test_data_path = "criteo_test_release_small.txt/criteo_test_release_small.txt"

    # Parse test data
    test_parser = parse_test_data(test_data_path, batch_size=10)

    # Inspect the first batch
    for batch in test_parser:
        for example in batch:
            print(f"Example ID: {example['example_id']}")
            print(f"Number of Candidates: {len(example['product_features'])}")
            print("Product Features:")
            for product in example['product_features']:
                print(f"  {product}")
            print("---")
        break  # Stop after the first batch