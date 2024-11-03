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

def simulate_game(model_1, model_2, move_to_int, int_to_move, initial_board=None, sleep_time=0.5, board_size=500, temperature=1.0):
    # Use the provided board or start from the default position
    board = copy.deepcopy(initial_board) if initial_board else chess.Board()

    # Ensure models are in evaluation mode
    model_1.eval()
    model_2.eval()

    # Start the game loop
    while True:
        # Clear the previous board output
        clear_output(wait=True)

        # Render and display the board as an SVG
        board_svg = chess.svg.board(board=board, size=board_size)
        display(SVG(data=board_svg))

        # Determine which model to use based on the side to move
        if board.turn == chess.WHITE:
            best_move = predict_best_move(model_1, move_to_int, int_to_move, board, temperature)
        else:
            best_move = predict_best_move(model_2, move_to_int, int_to_move, board, temperature)

        if best_move is None:
            # No valid move predicted, check for the result and display it
            result = board.result()
            print(f"Game over! Result: {result}")
            if result == "1-0":
                print("White wins!")
            elif result == "0-1":
                print("Black wins!")
            elif result == "1/2-1/2":
                print("Draw!")
            break  # Exit the game loop

        # Apply the predicted move
        move = board.parse_uci(best_move)
        board.push(move)

        # Wait for the specified time before updating again
        time.sleep(sleep_time)

        # Check if the game is over after applying the move
        if board.is_game_over():
            # Display the final result
            clear_output(wait=True)
            board_svg = chess.svg.board(board=board, size=board_size)
            display(SVG(data=board_svg))

            result = board.result()
            print(f"Game over! Result: {result}")
            if result == "1-0":
                print("Model 1 - White wins!")
            elif result == "0-1":
                print("Model 2 - Black wins!")
            elif result == "1/2-1/2":
                print("Draw!")
            break