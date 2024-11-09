import gc
import chess
import numpy as np
from chess import Board
import chess.pgn as pgn
import tqdm

# Function to convert board to matrix
def board_to_matrix(board: chess.Board):
    # Initialize the matrix with 16 planes
    # Planes 0-5: White pieces (P, N, B, R, Q, K)
    # Planes 6-11: Black pieces (P, N, B, R, Q, K)
    # Plane 12: Side to move
    # Plane 13: Castling rights (KQkq)
    # Plane 14: Squares attacked by White
    # Plane 15: Squares attacked by Black
    matrix = np.zeros((16, 8, 8), dtype=np.float32)
    
    # Map pieces to planes
    piece_type_to_plane = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    
    # Populate piece planes
    for square, piece in board.piece_map().items():
        plane = piece_type_to_plane[(piece.piece_type, piece.color)]
        row, col = divmod(square, 8)
        matrix[plane, row, col] = 1.0
    
    # Side to move plane
    matrix[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Castling rights encoded in specific positions
    matrix[13, 0, 0] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    matrix[13, 0, 7] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    matrix[13, 7, 0] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    matrix[13, 7, 7] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    # Efficient attack map calculations
    white_attacks_bb = 0
    black_attacks_bb = 0

    # Using chess.SquareSet and efficient bitboard calculations
    for color, attack_plane in [(chess.WHITE, 14), (chess.BLACK, 15)]:
        for piece_type in chess.PIECE_TYPES:
            squares = board.pieces(piece_type, color)
            attack_bb = sum(board.attacks_mask(square) for square in squares)
            if color == chess.WHITE:
                white_attacks_bb |= attack_bb
            else:
                black_attacks_bb |= attack_bb
        attack_squares = np.array(chess.SquareSet(white_attacks_bb if color == chess.WHITE else black_attacks_bb).squares)
        if len(attack_squares) > 0:
            rows, cols = np.divmod(attack_squares, 8)
            matrix[attack_plane, rows, cols] = 1.0

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
