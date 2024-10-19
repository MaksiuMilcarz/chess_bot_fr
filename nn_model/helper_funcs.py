import os
import chess.pgn
import torch
import numpy as np
import random
from tqdm import tqdm

def board_to_matrix(board):
    """
    Convert the board to a 13x8x8 matrix representation.
    - First 12 planes: Piece types.
    - 13th plane: Side to move (1's for white, 0's for black).
    """
    matrix = np.zeros((13, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1  # 0-5 for pawn to king
        piece_color = 0 if piece.color else 6  # 0-5 for white, 6-11 for black
        matrix[piece_type + piece_color, row, col] = 1

    # Side to move plane
    matrix[12, :, :] = 1.0 if board.turn else 0.0

    return matrix

def encode_moves(moves):
    """
    Create a mapping from move UCI notation to unique integer labels.

    Args:
    - moves (list): List of move strings in UCI notation.

    Returns:
    - move_to_int (dict): Dictionary mapping move strings to integers.
    - int_to_move (dict): Dictionary mapping integers back to move strings.
    """
    unique_moves = sorted(set(moves))
    move_to_int = {move: idx for idx, move in enumerate(unique_moves)}
    int_to_move = {idx: move for move, idx in move_to_int.items()}
    return move_to_int, int_to_move

def preprocess_and_save_data(pgn_files, chunk_size=10000, data_dir='data_chunks'):
    """
    Preprocess PGN files, focusing on mid/late game positions using a probabilistic approach,
    and save the data in chunks for efficient training.

    Args:
    - pgn_files (list): List of PGN file paths.
    - chunk_size (int): Number of samples per chunk file.
    - data_dir (str): Directory to save chunk files.

    Returns:
    - move_to_int (dict): Mapping from move UCI notation to integer labels.
    - num_classes (int): Total number of unique moves (classes).
    """
    os.makedirs(data_dir, exist_ok=True)
    move_set = set()
    sample_count = 0
    chunk_idx = 0
    X_chunk = []
    y_chunk = []

    # First pass to collect all unique moves
    print("Collecting unique moves...")
    for file in tqdm(pgn_files, desc="Processing PGN Files for Move Collection"):
        with open(file, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                moves = [move.uci() for move in game.mainline_moves()]
                move_set.update(moves)

    move_to_int, int_to_move = encode_moves(move_set)
    num_classes = len(move_to_int)
    print(f"Total unique moves: {num_classes}")

    # Second pass to process games and save data in chunks
    print("Processing games and saving data in chunks...")
    for file in tqdm(pgn_files, desc="Processing PGN Files for Data Extraction"):
        with open(file, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                board = game.board()
                moves = list(game.mainline_moves())
                total_moves = len(moves)
                for idx, move in enumerate(moves):
                    board.push(move)
                    # Probabilistic sampling focusing on mid/late game
                    probability = idx / total_moves  # Increases as the game progresses
                    if random.random() < probability:
                        X = board_to_matrix(board)
                        y = move_to_int.get(move.uci(), -1)
                        if y == -1:
                            continue  # Skip moves not in the mapping
                        X_chunk.append(X)
                        y_chunk.append(y)
                        sample_count += 1

                        # Save chunk to disk when full
                        if sample_count % chunk_size == 0:
                            # Convert lists to single NumPy arrays
                            X_array = np.array(X_chunk, dtype=np.float32)
                            y_array = np.array(y_chunk, dtype=np.int64)

                            # Convert NumPy arrays to PyTorch tensors
                            X_tensor = torch.from_numpy(X_array)
                            y_tensor = torch.from_numpy(y_array)

                            # Save the tensors
                            chunk_path = os.path.join(data_dir, f'data_chunk_{chunk_idx}.pt')
                            torch.save({'X': X_tensor, 'y': y_tensor}, chunk_path)

                            # Reset chunks for the next batch
                            X_chunk = []
                            y_chunk = []
                            chunk_idx += 1

    # Save any remaining samples
    if X_chunk:
        X_array = np.array(X_chunk, dtype=np.float32)
        y_array = np.array(y_chunk, dtype=np.int64)
        X_tensor = torch.from_numpy(X_array)
        y_tensor = torch.from_numpy(y_array)
        chunk_path = os.path.join(data_dir, f'data_chunk_{chunk_idx}.pt')
        torch.save({'X': X_tensor, 'y': y_tensor}, chunk_path)

    print(f"Total samples processed and saved: {sample_count}")
    return move_to_int, num_classes