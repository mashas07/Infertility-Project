"""
Fixed load_data method for DataProcessor class
Replace your current load_data method with this one
"""

import os
import pandas as pd
import kagglehub


def load_data(self, csv_path=None):
    """
    Load data from CSV file.

    Args:
        csv_path (str): Path to CSV file. If None, downloads from Kaggle.

    Returns:
        pd.DataFrame: Loaded data
    """
    if csv_path:
        # If a path is provided, use it directly
        self.csv_path = csv_path
    else:
        # Download from Kaggle
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("fida5073/female-infertility")
        print(f"Dataset downloaded to: {path}")

        # The path is a directory, so we need to find the CSV file inside it
        if os.path.isdir(path):
            # Find all CSV files in the directory
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

            if not csv_files:
                raise ValueError(f"No CSV file found in downloaded dataset at {path}")

            # Use the first CSV file found
            self.csv_path = os.path.join(path, csv_files[0])
            print(f"Found CSV file: {csv_files[0]}")
        else:
            # If it's already a file, use it directly
            self.csv_path = path

    if not self.csv_path:
        raise ValueError("CSV path must be provided")

    # Check if the path exists and is a file
    if not os.path.exists(self.csv_path):
        raise FileNotFoundError(f"File not found: {self.csv_path}")

    if os.path.isdir(self.csv_path):
        raise IsADirectoryError(f"Path is a directory, not a file: {self.csv_path}")

    print(f"Loading data from {self.csv_path}...")
    self.data = pd.read_csv(self.csv_path)
    print(f"Data loaded successfully. Shape: {self.data.shape}")

    # Display column names to help identify the target column
    print(f"\nColumns in dataset: {list(self.data.columns)}")

    return self.data


# Alternative: If you already know the directory path and just need to fix it
def load_data_from_directory(directory_path):
    """
    Helper function to load CSV from a known directory

    Args:
        directory_path (str): Path to directory containing CSV

    Returns:
        pd.DataFrame: Loaded data
    """
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Not a directory: {directory_path}")

    # Find CSV files
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    if not csv_files:
        raise ValueError(f"No CSV files found in {directory_path}")

    print(f"Found {len(csv_files)} CSV file(s):")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f}")

    # Use the first CSV file
    csv_path = os.path.join(directory_path, csv_files[0])
    print(f"\nUsing: {csv_files[0]}")

    df = pd.read_csv(csv_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


# Quick fix for your main() function if you want to use the already downloaded data
def main_with_existing_path():
    """
    Use this if you want to load from the already downloaded directory
    """
    print("=" * 60)
    print("FERTILITY PREDICTION SYSTEM")
    print("=" * 60)

    # The directory where Kaggle downloaded the data
    kaggle_dir = '/Users/mashasobko/.cache/kagglehub/datasets/fida5073/female-infertility/versions/1'

    # Find the CSV file in that directory
    csv_files = [f for f in os.listdir(kaggle_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {kaggle_dir}")
        return

    csv_file_path = os.path.join(kaggle_dir, csv_files[0])
    print(f"Using CSV file: {csv_file_path}")

    # Now proceed with your training
    predictor = FertilityPredictor()

    # Load data using the full path to the CSV file
    predictor.train_from_csv(csv_file_path, target_column='label')  # Update 'label' to your actual target column


if __name__ == "__main__":
    main_with_existing_path()