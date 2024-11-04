import chess
from predict import predict_best_move
import chess
import chess
import chess.svg
import time
from IPython.display import clear_output, display, SVG
import numpy as np
import copy

def compare_models(model_A, model_B, move_to_int_A, int_to_move_A, move_to_int_B, int_to_move_B, num_games=1000, temperature_A=1.0, temperature_B=1.0):
    """
    Simulates multiple games between two models and collects the results per model.

    Parameters:
    - model_A: First model to evaluate.
    - model_B: Second model to evaluate.
    - move_to_int: Dictionary mapping moves in UCI format to integer indices.
    - int_to_move: Dictionary mapping integer indices to moves in UCI format.
    - num_games: Number of games to simulate.

    Returns:
    - results: Dictionary with counts of 'model_A_wins', 'model_B_wins', and 'draws'.
    """
    # Ensure models are in evaluation mode
    model_A.eval()
    model_B.eval()

    # Initialize result counters
    results = {'model_A_wins': 0, 'model_B_wins': 0, 'draws': 0}

    for game_num in range(num_games):
        board = chess.Board()  # Start from the default position

        # Alternate colors: model_A plays white in even games, black in odd games
        if game_num % 2 == 0:
            white_model = model_A
            black_model = model_B
            temp_white = temperature_A
            temp_black = temperature_B
            move_to_int_white = move_to_int_A
            int_to_move_white = int_to_move_A
            move_to_int_black = move_to_int_B
            int_to_move_black = int_to_move_B
        else:
            white_model = model_B
            black_model = model_A
            temp_white = temperature_B
            temp_black = temperature_A
            move_to_int_white = move_to_int_B
            int_to_move_white = int_to_move_B
            move_to_int_black = move_to_int_A
            int_to_move_black = int_to_move_A

        # Start the game loop
        while not board.is_game_over():
            # Determine which model to use based on the side to move
            if board.turn == chess.WHITE:
                best_move = predict_best_move(white_model, move_to_int_white, int_to_move_white, board, temp_white)
            else:
                best_move = predict_best_move(black_model, move_to_int_black, int_to_move_black, board, temp_black)

            if best_move is None:
                # No valid move predicted; the current player loses
                break

            # Apply the predicted move
            move = board.parse_uci(best_move)
            board.push(move)

        # After the game ends, update the results
        result = board.result()
        if result == '1-0':
            # White won
            if white_model == model_A:
                results['model_A_wins'] += 1
            else:
                results['model_B_wins'] += 1
        elif result == '0-1':
            # Black won
            if black_model == model_A:
                results['model_A_wins'] += 1
            else:
                results['model_B_wins'] += 1
        else:
            # Draw
            results['draws'] += 1

        # Optional: Print progress every 100 games
        if (game_num + 1) % 100 == 0:
            print(f"Completed {game_num + 1}/{num_games} games.")

    # Output the final results
    print("Simulation completed.")
    print(f"Total games: {num_games}")
    print(f"Model A wins: {results['model_A_wins']}")
    print(f"Model B wins: {results['model_B_wins']}")
    print(f"Draws: {results['draws']}")

    return results

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