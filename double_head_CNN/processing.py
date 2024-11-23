import h5py
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from utils import bitboard_to_matrix

def preprocess_and_save_hdf5(parquet_file, hdf5_file, chunk_size=10000):
    """
    Preprocesses data from a Parquet file and saves it into an HDF5 file.

    Parameters:
    - parquet_file: Path to the input Parquet file.
    - hdf5_file: Path to the output HDF5 file.
    - chunk_size: Number of samples to process at a time.
    """
    # Open the Parquet file
    parquet_reader = pq.ParquetFile(parquet_file)
    num_rows = parquet_reader.metadata.num_rows

    # Create the HDF5 file
    with h5py.File(hdf5_file, 'w') as h5f:
        # Prepare datasets with appropriate shapes
        board_dataset = h5f.create_dataset('boards', shape=(num_rows, 13, 8, 8), dtype='float32')
        policy_dataset = h5f.create_dataset('policies', shape=(num_rows, 4672), dtype='float32')
        value_dataset = h5f.create_dataset('values', shape=(num_rows,), dtype='float32')

        # Process data in chunks
        for start_idx in tqdm(range(0, num_rows, chunk_size), desc="Processing Data"):
            end_idx = min(start_idx + chunk_size, num_rows)
            num_samples = end_idx - start_idx

            # Read a batch of data from the Parquet file
            table = parquet_reader.read_row_group(start_idx // chunk_size).to_pandas()

            # Slice the table to get the correct rows
            table_chunk = table.iloc[start_idx % chunk_size : (start_idx % chunk_size) + num_samples]

            # Initialize arrays to hold preprocessed data
            boards = np.zeros((num_samples, 13, 8, 8), dtype='float32')
            policies = np.zeros((num_samples, 4672), dtype='float32')
            values = np.zeros((num_samples,), dtype='float32')

            for i in range(num_samples):
                # Extract raw data
                state = np.array(table_chunk['state'].iloc[i], dtype=np.uint8)  # Shape: (12, 64)
                policy = np.array(table_chunk['policy'].iloc[i], dtype=np.float32)  # Shape: (4672,)
                value = np.float32(table_chunk['value'].iloc[i])
                side_to_move = np.float32(table_chunk['side_to_move'].iloc[i])

                # Preprocess the board state
                board_matrix = bitboard_to_matrix(state, side_to_move)  # Shape: (13, 8, 8)

                # Store preprocessed data
                boards[i] = board_matrix
                policies[i] = policy
                values[i] = value

            # Write the chunk to the HDF5 datasets
            board_dataset[start_idx:end_idx] = boards
            policy_dataset[start_idx:end_idx] = policies
            value_dataset[start_idx:end_idx] = values

    print(f"Data preprocessing complete. Preprocessed data saved to {hdf5_file}")