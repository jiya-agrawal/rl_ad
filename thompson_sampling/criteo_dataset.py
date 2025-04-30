# START OF FILE criteo_dataset.py
#!/usr/bin/env python
from __future__ import print_function
import utils
import hashlib
import gzip

class CriteoDataset:
    def __init__(self, filepath, isTest=False, isGzip=False, id_map=False, inverse_propensity=True, debug=False):
        # --- Use text mode 'rt' / 'r' ---
        if filepath.endswith(".gz") or isGzip:
            self.fp = gzip.open(filepath, "rt", encoding='utf-8')
        else:
            self.fp = open(filepath, "r", encoding='utf-8')
        self.inverse_propensity = inverse_propensity
        self.debug = debug
        self.isTest = isTest
        self.id_map = id_map
        self.line_buffer = [] # Buffer for the *next* line read ahead

    def __iter__(self):
        return self

    def next(self):
        next_block = self.get_next_impression_block()
        if next_block:
            return next_block
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def get_next_line(self):
        # Read the next line from the file (already a string)
        try:
            line = self.fp.readline()
            return line # Returns '' at EOF
        except Exception as e:
             print(f"Error reading line: {e}")
             return None # Indicate error

    def get_next_impression_block(self):
        # --- Use the line from the buffer if available ---
        if self.line_buffer:
            current_line = self.line_buffer.pop(0) # Get the first (and only) buffered line
            # Assertion should hold now if logic is correct
            assert len(self.line_buffer) == 0
        else:
            # Otherwise, read the next line from the file
            current_line = self.get_next_line()

        if not current_line: # Check for EOF or error from get_next_line
            return False

        # Process the first line of the block
        line_stripped = current_line.strip()
        if not line_stripped: # Skip empty lines encountered as the first line
            return self.get_next_impression_block() # Recursively call to get the next non-empty line

        block_impression_id = utils.extract_impression_id(line_stripped)
        if self.id_map:
            block_impression_id = self.id_map(block_impression_id)

        cost = None
        propensity = None
        if not self.isTest:
            try:
                 cost, propensity = utils.extract_cost_propensity(line_stripped, inverse_propensity=self.inverse_propensity)
            except Exception as e:
                 print(f"Error parsing cost/propensity from line: {line_stripped[:100]}... Error: {e}")
                 # Decide how to handle: skip impression, default values? Let's skip.
                 # To skip, we need to read until the next impression ID changes.
                 # This adds complexity. Simpler: raise the error or return partial data.
                 # For now, let's proceed, cost/propensity will be None, potentially causing issues later.
                 pass # Or return None / raise


        try:
             candidate_features = [utils.extract_features(line_stripped, self.debug)]
        except Exception as e:
             print(f"Error parsing features from line: {line_stripped[:100]}... Error: {e}")
             # Handle error similarly - skip impression or return partial?
             # Let's return what we have so far, but it might be incomplete.
             candidate_features = [] # Empty list if first line fails


        # Read subsequent lines belonging to the same impression block
        while True:
            next_line = self.get_next_line()
            if not next_line: # EOF
                break # End of file, finish current block

            next_line_stripped = next_line.strip()
            if not next_line_stripped: # Skip empty lines within or between blocks
                continue

            try:
                 line_impression_id = utils.extract_impression_id(next_line_stripped)
                 if self.id_map:
                     line_impression_id = self.id_map(line_impression_id)

                 if line_impression_id != block_impression_id:
                     # Found the start of the next block, buffer it and stop current block
                     self.line_buffer.append(next_line) # Buffer the original unstripped line
                     break # Stop processing the current block
                 else:
                     # This line belongs to the current block, add its features
                     candidate_features.append(utils.extract_features(next_line_stripped, debug=self.debug))
            except Exception as e:
                 print(f"Error parsing subsequent line: {next_line_stripped[:100]}... Error: {e}. Skipping line.")
                 # Continue to the next line, skipping the problematic one


        _response = {}
        _response["id"] = block_impression_id
        _response["candidates"] = candidate_features
        if not self.isTest:
            # Assign potentially None values if parsing failed earlier
            _response["cost"] = cost
            _response["propensity"] = propensity
        return _response

    def close(self):
        self.__del__()

    def __del__(self):
        if hasattr(self, 'fp') and self.fp:
            self.fp.close()

# END OF FILE criteo_dataset.py