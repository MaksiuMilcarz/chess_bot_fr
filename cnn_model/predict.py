import numpy as np
import torch
from helper_funcs import board_to_matrix
import torch.nn.functional as F
from chess import Board
import chess
import chess.svg
import time
from IPython.display import clear_output, display, SVG
import numpy as np
import torch
import torch.nn.functional as F
import copy

def predict_best_move(model, move_to_int, int_to_move, board, temperature=1.0):
    """
    Predicts a move for the given board state by sampling from the probability distribution over legal moves,
    adjusted by a temperature parameter.

    Parameters:
    - model: Trained PyTorch model.
    - move_to_int: Dictionary mapping moves in UCI format to integer indices.
    - int_to_move: Dictionary mapping integer indices to moves in UCI format.
    - board: chess.Board object representing the current game state.
    - temperature: Float value to adjust the randomness of move selection.

    Returns:
    - best_move: Move sampled from the adjusted probability distribution over legal moves, in UCI format (string).
    """
    # Convert the board to the input matrix
    input_matrix = board_to_matrix(board)

    # Convert to PyTorch tensor and add batch dimension
    input_tensor = torch.from_numpy(input_matrix).unsqueeze(0).to(next(model.parameters()).device)

    # Get the model's raw output (logits)
    with torch.no_grad():
        outputs = model(input_tensor)

    # Apply temperature scaling to logits
    logits = outputs.squeeze(0) / temperature  # Shape: (num_classes,)

    # Get list of legal moves and their indices
    legal_moves = [move.uci() for move in board.legal_moves]
    legal_indices = [move_to_int[move] for move in legal_moves if move in move_to_int]

    if not legal_indices:
        return None  # No legal moves found in move_to_int mapping

    # Extract logits for legal moves
    legal_logits = logits[legal_indices]

    # Apply softmax to get probabilities
    legal_probs = torch.softmax(legal_logits, dim=0)

    # Sample a move based on the probabilities
    chosen_idx = torch.multinomial(legal_probs, num_samples=1).item()
    chosen_move_index = legal_indices[chosen_idx]
    best_move = int_to_move[chosen_move_index]

    return best_move

