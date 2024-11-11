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

# Function to convert board to matrix
def board_to_matrix_mark34(board: chess.Board):
    matrix = np.zeros((14, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()

    # Map pieces to planes using numpy indexing for efficiency
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        matrix[plane, row, col] = 1.0

    # Side to move plane
    matrix[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Legal moves
    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix

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

    # Castling rights
    castling_rights = board.castling_rights
    if castling_rights & chess.BB_H1:  # White kingside
        matrix[13, 7, 7] = 1.0
    if castling_rights & chess.BB_A1:  # White queenside
        matrix[13, 7, 0] = 1.0
    if castling_rights & chess.BB_H8:  # Black kingside
        matrix[13, 0, 7] = 1.0
    if castling_rights & chess.BB_A8:  # Black queenside
        matrix[13, 0, 0] = 1.0

    # Squares attacked by white and black
    matrix[14, :, :] = compute_attack_map(board, chess.WHITE)
    matrix[15, :, :] = compute_attack_map(board, chess.BLACK)   

    return matrix

def compute_attack_map(board, color):
    attacks = np.zeros((8, 8), dtype=np.float32)
    attack_squares = set()

    for square in board.pieces(chess.PAWN, color):
        attack_squares |= set(board.attacks(square))
    for square in board.pieces(chess.KNIGHT, color):
        attack_squares |= set(board.attacks(square))
    for square in board.pieces(chess.BISHOP, color):
        attack_squares |= set(board.attacks(square))
    for square in board.pieces(chess.ROOK, color):
        attack_squares |= set(board.attacks(square))
    for square in board.pieces(chess.QUEEN, color):
        attack_squares |= set(board.attacks(square))
    for square in board.pieces(chess.KING, color):
        attack_squares |= set(board.attacks(square))

    if attack_squares:
        rows, cols = np.divmod(list(attack_squares), 8)
        attacks[rows, cols] = 1.0

    return attacks
