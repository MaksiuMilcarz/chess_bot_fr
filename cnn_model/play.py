import argparse
import os
import pickle
import torch
import chess
import chess.svg
from IPython.display import clear_output, display, SVG
from train.model import ChessModel_mark34
from train.utils import board_to_matrix_mark34
from predict.predict import predict_best_move


def display_board(board):
    """
    Displays the chess board using IPython display capabilities for better visualization.
    """
    clear_output(wait=True)
    svg_board = chess.svg.board(board=board)
    display(SVG(svg_board))


def main():
    # Load the move_to_int and int_to_move dictionaries
    with open('/Users/maksiuuuuuuu/Desktop/MyWork/game_bots/chess_bot_fr/models/mark4_move_to_int.pkl', 'rb') as f:
        move_to_int = pickle.load(f)
    int_to_move = {v: k for k, v in move_to_int.items()}
    num_classes = len(move_to_int)
    
    # Load the model
    model = ChessModel_mark34(num_classes=num_classes)
    model.load_state_dict(torch.load('/Users/maksiuuuuuuu/Desktop/MyWork/game_bots/chess_bot_fr/models/mark4.3-25e-500k.pth', map_location=torch.device('mps')))
    model.eval()

    # Initialize the board
    board = chess.Board()

    while not board.is_game_over():
        # Display the board
        display_board(board)
        print('Your move? (e.g., e2e4)')

        # Get user's move
        user_move = input().strip()

        try:
            move = chess.Move.from_uci(user_move)
            if move in board.legal_moves:
                board.push(move)
                # Clear and display the board after player's move
                display_board(board)
            else:
                print('Illegal move.')
                input('Press Enter to continue...')
                continue
        except ValueError:
            print('Invalid move format.')
            input('Press Enter to continue...')
            continue

        if board.is_game_over():
            break

        # Bot's turn
        bot_move_uci = predict_best_move(model, move_to_int, int_to_move, board)
        if bot_move_uci is None:
            print('Bot resigns. You win!')
            break

        bot_move = chess.Move.from_uci(bot_move_uci)
        board.push(bot_move)
        print(f'Bot plays: {bot_move_uci}')

        # Clear and display the board after bot's move
        display_board(board)

        if board.is_game_over():
            break

    # Display final board and game result
    display_board(board)
    print('\nGame over.')
    print(f'Result: {board.result()}')

if __name__ == '__main__':
    main()