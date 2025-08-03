# Import required libraries
import pandas as pd
import pyarrow.parquet as pq

# Function to read a parquet file
def read_parquet_file(file_path):
    try:
        # Read the parquet file into a pandas DataFrame
        df = pd.read_parquet(file_path)

        # Print basic information about the dataset
        print("Dataset Info:")
        print("-" * 50)
        print("\nDataset Shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nData Types:")
        print(df.dtypes)
        print("\nFirst few rows:")
        print(df.head())

        return df

    except Exception as e:
        print(f"Error reading parquet file: {str(e)}")
        return None


# Example usage
if __name__ == "__main__":
    file_path = "/home/swadhin/experiments/datasets/train-00000-of-00001-c755640cc4645ad3.parquet"  # Adjust the path as necessary
    df = read_parquet_file(file_path)
    if df is not None:
        print("\nSuccessfully read the parquet file.")
    else:
        print("\nFailed to read the parquet file.")

    print("\nEnd of script.")
