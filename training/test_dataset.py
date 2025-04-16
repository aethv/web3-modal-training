from datasets import load_from_disk
import os

# Try to load the dataset
try:
    dataset_path = os.path.join("data", "tokenized_dataset")
    print(f"Attempting to load dataset from: {os.path.abspath(dataset_path)}")
    
    # Check if the directory exists
    if not os.path.exists(dataset_path):
        print(f"Error: Directory {dataset_path} does not exist!")
    else:
        # List contents of the directory
        print(f"Directory contents: {os.listdir(dataset_path)}")
    
    # Try to load the dataset
    tokenized_dataset = load_from_disk(dataset_path)
    print("Dataset loaded successfully!")
    print(f"Dataset structure: {tokenized_dataset}")
    print(f"Dataset splits: {tokenized_dataset.keys()}")
    print(f"Train dataset size: {len(tokenized_dataset['train'])}")
except Exception as e:
    print(f"Error loading dataset: {e}")