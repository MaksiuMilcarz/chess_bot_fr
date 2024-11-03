import gc
import numpy as np
from chess import Board
import chess.pgn as pgn
import tqdm

# Function to convert board to matrix
def board_to_matrix(board: Board):
    matrix = np.zeros((14, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1
    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1
    # Side to move plane
    matrix[13, :, :] = 1.0 if board.turn else 0.0
    return matrix

# Function to create input for NN
def create_input_for_nn(games):
    X_list = []
    y_list = []
    for game in games:
        board = game.board()
        total_moves = len(list(game.mainline_moves()))
        move_number = 0
        for move in game.mainline_moves():
            move_number += 1
            # Define probability of including this position
            alpha = 1.5  # Adjust alpha to control the weighting
            p_include = (move_number / total_moves) ** alpha
            # Randomly decide whether to include this position
            if np.random.rand() < p_include:
                X_list.append(board_to_matrix(board))
                y_list.append(move.uci())
            board.push(move)
    X_array = np.array(X_list, dtype=np.float32)
    y_array = np.array(y_list)
    return X_array, y_array

# Collect unique moves
def collect_unique_moves(files, max_games):
    total_games_processed = 0
    all_moves = set()
    for file in tqdm.tqdm(files, desc='Collecting Moves'):
        with open(file, 'r') as pgn_file:
            while True:
                game = pgn.read_game(pgn_file)
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

# Process data and save chunks
def process_data_and_save_chunks(files, move_to_int, chunk_size, max_games):
    total_games_processed = 0
    chunk_index = 0
    data_chunk_files = []
    for file in tqdm.tqdm(files, desc='Processing Data Chunks'):
        with open(file, 'r') as pgn_file:
            while True:
                games = []
                for _ in range(chunk_size):
                    game = pgn.read_game(pgn_file)
                    if game is None:
                        break
                    games.append(game)
                    total_games_processed += 1
                    if total_games_processed >= max_games:
                        break
                if not games:
                    break
                # Process games into X and y
                X_chunk, y_chunk = create_input_for_nn(games)
                # Encode y_chunk using move_to_int mapping
                y_encoded = np.array([move_to_int[move] for move in y_chunk], dtype=np.int64)
                # Save X_chunk and y_encoded to disk
                np.savez_compressed(f'data_chunk_{chunk_index}.npz', X=X_chunk, y=y_encoded)
                data_chunk_files.append(f'data_chunk_{chunk_index}.npz')
                chunk_index += 1
                if total_games_processed >= max_games:
                    break
                gc.collect()
        if total_games_processed >= max_games:
            break
    return data_chunk_files
