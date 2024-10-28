import numpy as np
import torch
from helper_funcs import board_to_matrix
import torch.nn.functional as F
from chess import Board


def predict_best_move(model, move_to_int, board):
    """
    Predicts the best move for the given board state using the trained model.

    Parameters:
    - model: Trained PyTorch model
    - move_to_int: Dictionary mapping moves in UCI format to integer indices
    - board: chess.Board object representing the current game state

    Returns:
    - best_move: Best move predicted by the model in UCI format (string)
    """
    # Create reverse mapping from indices to moves
    int_to_move = {v: k for k, v in move_to_int.items()}

    # Convert the board to the input matrix
    input_matrix = board_to_matrix(board)
    # Convert to PyTorch tensor and add batch dimension
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 14, 8, 8)

    # Ensure the model is in evaluation mode
    model.eval()

    # Move tensor to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Get the model's raw output (logits)
    with torch.no_grad():
        outputs = model(input_tensor)

    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs, dim=1)  # Shape: (1, num_classes)
    probabilities = probabilities.cpu().numpy()[0]

    # Get list of legal moves in UCI format
    legal_moves = [move.uci() for move in board.legal_moves]

    # Get indices of moves sorted by probability in descending order
    sorted_indices = np.argsort(-probabilities)

    # Iterate over sorted indices and select the first legal move
    for idx in sorted_indices:
        move_uci = int_to_move.get(idx)
        if move_uci in legal_moves:
            best_move = move_uci
            break
    else:
        best_move = None  # No legal move predicted

    return best_move