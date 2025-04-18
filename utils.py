def parse_criteo_data(file_path, batch_size=1024):
    """
    A generator-based parser for the Criteo dataset that yields mini-batches of parsed data.
    
    Args:
        file_path (str): Path to the dataset file (.txt).
        batch_size (int): Number of examples per mini-batch.
    
    Yields:
        list: A list of dictionaries containing parsed features, labels, and propensity scores for a mini-batch.
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
        # print(f"Example ID: {example_id}")
        # Initialize variables
        label = None
        propensity = None
        display_features = {}
        # print(f"Parts: {parts}")
        # Iterate through parts to extract label, propensity, and features
        i = 1  # Start from index 1 since index 0 is the example ID
        while i < len(parts):
            part = parts[i]
            if part.startswith("|l"):
                # Label is the next element after "|l"
                label = float(parts[i + 1])
                i += 2  # Skip the next element since it's already processed
            elif part.startswith("|p"):
                # Propensity is the next element after "|p"
                propensity = float(parts[i + 1].split("|")[0])
                i += 2  # Skip the next element since it's already processed
            elif ":" in part:
                # Parse display features (|f)
                # print(f"Pending part {part}")
                key, value = part.split(":")
                display_features[int(key)] = float(value)
                i += 1
            else:
                # Handle unexpected format
                print(f"Unexpected part: {part}")
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
        example_id = parts[0]
        
        # Parse product features (|f)
        product_features = {}
        # print(f"Product line parts: {parts}")
        for feat in parts[2:]:
            # print(f"Pending feature {feat}")
            key, value = feat.split(":")
            product_features[int(key)] = float(value)
        
        return product_features
    
    def read_file():
        """Generator to read the file line by line."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line
    
    # Initialize variables for batching
    current_batch = []
    header = None
    
    for line in read_file():
        if line.startswith("|"):
            continue  # Skip malformed lines
        
        if "|l" in line and "|p" in line:
            # Header line
            if header is not None:
                # If there's an incomplete example, skip it
                header = None
            
            header = parse_header_line(line)
        else:
            # Product line
            if header is not None:
                product_features = parse_product_line(line)
                header["product_features"].append(product_features)
            
            # Yield the batch if we've collected all products for this example
            if header and len(header["product_features"]) == len(header["product_features"]):
                current_batch.append(header)
                header = None  # Reset header for the next example
                
                # Yield the batch if it reaches the desired size
                if len(current_batch) == batch_size:
                    yield current_batch
                    current_batch = []
    
    # Yield any remaining examples in the last batch
    if current_batch:
        yield current_batch

# # Path to the dataset file
# train_data_path = "criteo_train_small.txt/criteo_train_small.txt"
# test_data_path = "criteo_test_release_small.txt/criteo_test_release_small.txt"

# # Create a parser for the training data
# train_parser = parse_criteo_data(train_data_path, batch_size=1024)

# # Iterate through mini-batches
# for batch in train_parser:
#     # Process the batch (e.g., train your epsilon-greedy model)
#     print(f"Processed batch with {len(batch)} examples")


def parse_criteo_test_data(file_path, batch_size=1024):
    """
    A generator-based parser for the Criteo test dataset that yields mini-batches of parsed product features.
    
    Args:
        file_path (str): Path to the test dataset file (.txt).
        batch_size (int): Number of examples per mini-batch.
    
    Yields:
        list: A list of lists containing parsed product features for each example in the mini-batch.
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
        
        # Parse product features (|f)
        product_features = {}
        # print(f"Product line parts: {parts}")
        for feat in parts[2:]:
            # print(f"Pending feature {feat}")
            key, value = feat.split(":")
            product_features[int(key)] = float(value)
        
        return product_features
    
    def read_file():
        """Generator to read the file line by line."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line
    
    # Initialize variables for batching
    current_batch = []
    current_example = []
    current_example_id = None
    
    for line in read_file():
        if line.startswith("|"):
            continue  # Skip malformed lines
        
        # Parse the product line
        parts = line.strip().split()
        example_id = parts[0]
        
        if example_id != current_example_id:
            # If we've started a new example, finalize the previous one
            if current_example:
                current_batch.append(current_example)
                if len(current_batch) == batch_size:
                    yield current_batch
                    current_batch = []
            
            # Start a new example
            current_example_id = example_id
            current_example = []
        
        # Parse the product features and add them to the current example
        product_features = parse_product_line(line)
        current_example.append(product_features)
    
    # Yield any remaining examples in the last batch
    if current_example:
        current_batch.append(current_example)
    if current_batch:
        yield current_batch

# test_parser = parse_criteo_test_data(test_data_path, batch_size=1024)

# for batch in test_parser:
#     # Process the batch (e.g., evaluate your model)
#     print(f"Processed test batch with {len(batch)} examples")

if __name__ == "__main__":
    train_data_path = "criteo_train_small.txt/criteo_train_small.txt"
    test_data_path = "criteo_test_release_small.txt/criteo_test_release_small.txt"

    # Create a parser for the training data
    train_parser = parse_criteo_data(train_data_path, batch_size=1024)

    # Iterate through mini-batches
    for batch in train_parser:
        # Process the batch (e.g., train your epsilon-greedy model)
        print(f"Processed batch with {len(batch)} examples")
    # Create a parser for the test data
    test_parser = parse_criteo_test_data(test_data_path, batch_size=1024)
    # Iterate through mini-batches
    for batch in test_parser:
        # Process the batch (e.g., evaluate your model)
        print(f"Processed test batch with {len(batch)} examples")