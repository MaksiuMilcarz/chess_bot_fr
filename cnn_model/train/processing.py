import gc
import chess
import numpy as np
from chess import Board
import chess.pgn as pgn
import tqdm
import multiprocessing as mp
import h5py
import multiprocessing as mp
from functools import partial
from utils import board_to_matrix

def collect_unique_moves(files, max_games):
    total_games_processed = 0
    all_moves = set()
    for file in tqdm.tqdm(files, desc='Collecting Moves'):
        with open(file, 'r') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                for move in game.mainline_moves():
                    all_moves.add(move.uci())
                total_games_processed += 1
                if total_games_processed >= max_games:
                    break
        if total_games_processed >= max_games:
            break
    move_to_int = {move: idx for idx, move in enumerate(sorted(all_moves))}
    num_classes = len(move_to_int)
    return move_to_int, num_classes

def preprocess_and_save_to_hdf5(files, move_to_int, max_games, positions_per_game=10):
    total_games_processed = 0
    hdf5_file = h5py.File('preprocessed_data.h5', 'w')
    # Estimate the total number of samples
    estimated_samples = max_games * positions_per_game
    X_dataset = hdf5_file.create_dataset('X', shape=(estimated_samples, 16, 8, 8), dtype=np.float32, compression='gzip', chunks=True, maxshape=(None, 16, 8, 8))
    y_dataset = hdf5_file.create_dataset('y', shape=(estimated_samples,), dtype=np.int64, compression='gzip', chunks=True, maxshape=(None,))
    
    sample_index = 0
    for file in tqdm.tqdm(files, desc='Preprocessing Data'):
        with open(file, 'r') as pgn_file:
            while total_games_processed < max_games:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                board = game.board()
                moves = list(game.mainline_moves())
                total_moves = len(moves)

                if total_moves == 0:
                    total_games_processed += 1
                    continue

                # Probabilistic sampling favoring end-game positions
                move_numbers = np.arange(1, total_moves + 1)
                # Adjust alpha to control the weighting (alpha > 1 favors later moves)
                alpha = 1.2
                probabilities = (move_numbers / total_moves) ** alpha
                probabilities /= probabilities.sum()
                positions = np.random.choice(total_moves, size=min(positions_per_game, total_moves), replace=False, p=probabilities)
                positions.sort()

                for idx in positions:
                    board_copy = board.copy(stack=False)
                    for move in moves[:idx]:
                        board_copy.push(move)
                    X = board_to_matrix(board_copy)
                    move_uci = moves[idx].uci()
                    y = move_to_int.get(move_uci)
                    if y is None:
                        continue

                    if sample_index >= X_dataset.shape[0]:
                        # Resize datasets if necessary
                        new_size = X_dataset.shape[0] + max_games * positions_per_game
                        X_dataset.resize((new_size, 16, 8, 8))
                        y_dataset.resize((new_size,))
                    X_dataset[sample_index] = X
                    y_dataset[sample_index] = y
                    sample_index += 1

                total_games_processed += 1
                if total_games_processed >= max_games:
                    break
    # Resize datasets to actual size
    X_dataset.resize((sample_index, 16, 8, 8))
    y_dataset.resize((sample_index,))
    hdf5_file.close()