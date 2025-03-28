# rl_ad


## Exploratory Data Analysis
### File
```eda.py``` does basic EDA on the data, generating 8 graph PNGs and printing out some statistics.
### Running it
Modify the file path in the last line of the code to the filepath to your dataset.
*edit ```max_examples``` parameter when working with larger dataset


## Epsilon-Greedy Algorith
### The following files contain code for implementing epsilon-greedy
- ```epsilon.py```: running on complete dataset in batches
- ```epsilon2.py```: working on sample of entire dataset only
- ```epsilon3.py```: added decaying epsilon, exploration bonus to predicted reward and 10k batch-size
