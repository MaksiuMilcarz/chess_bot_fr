import chess
import numpy as np

def board_to_matrix(board: chess.Board):
    matrix = np.zeros((16, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()

    # Map pieces to planes using numpy indexing for efficiency
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        matrix[plane, row, col] = 1.0

    # Side to move plane
    matrix[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return matrix

def bitboard_to_matrix(bitboards, side_to_move):
    """
    Converts bitboards into a matrix suitable for CNN input.

    Parameters:
    - bitboards: numpy array of shape (12, 64) representing the bitboards for each piece type.
    - side_to_move: float (1.0 for white, 0.0 for black)

    Returns:
    - matrix: numpy array of shape (13, 8, 8)
    """
    # Reshape bitboards to (12, 8, 8)
    bitboards = bitboards.reshape(12, 8, 8)
    
    # Stack bitboards and side to move into one array
    side_plane = np.full((1, 8, 8), side_to_move, dtype=np.float32)
    matrix = np.concatenate((bitboards.astype(np.float32), side_plane), axis=0)

    return matrix