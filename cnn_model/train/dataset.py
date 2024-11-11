import chess.pgn
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import h5py
from utils import board_to_matrix

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_path):
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
        self.X = self.hdf5_file['X']
        self.y = self.hdf5_file['y']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        
            # Data Augmentation: Randomly flip the board
        if random.random() < 0.5:
            X = np.flip(X, axis=2)  # Flip horizontally
        if random.random() < 0.5:
            X = np.flip(X, axis=1)  # Flip vertically

        # Convert to torch tensors
        X = torch.from_numpy(X)
        y = torch.tensor(y, dtype=torch.long)

        return X, y

    def __del__(self):
        self.hdf5_file.close()

class ChessDataset(Dataset):
    def __init__(self, pgn_files, move_to_int, positions_per_game=10, max_games=None):
        self.pgn_files = pgn_files
        self.move_to_int = move_to_int
        self.positions_per_game = positions_per_game
        self.max_games = max_games
        self.game_offsets = self._index_games()
        self.total_samples = len(self.game_offsets) * self.positions_per_game

    def _index_games(self):
        game_offsets = []
        total_games = 0
        for file in self.pgn_files:
            with open(file, 'r') as pgn_file:
                offset = pgn_file.tell()
                while True:
                    headers = chess.pgn.read_headers(pgn_file)
                    if headers is None:
                        break
                    game_offsets.append((file, offset))
                    total_games += 1
                    if self.max_games and total_games >= self.max_games:
                        return game_offsets
                    offset = pgn_file.tell()
        return game_offsets

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        game_idx = idx // self.positions_per_game
        position_idx = idx % self.positions_per_game

        file, offset = self.game_offsets[game_idx]
        with open(file, 'r') as pgn_file:
            pgn_file.seek(offset)
            game = chess.pgn.read_game(pgn_file)

        board = game.board()
        moves = list(game.mainline_moves())
        total_moves = len(moves)

        if total_moves == 0:
            # Handle games with no moves
            return self.__getitem__((idx + 1) % self.__len__())

        # Adjust the sampling strategy for more variability
        move_indices = self._sample_positions(total_moves)

        move_idx = move_indices[position_idx]

        for move in moves[:move_idx]:
            board.push(move)

        X = board_to_matrix(board)
        move = moves[move_idx].uci()
        y = self.move_to_int.get(move)

        if y is None:
            # Handle unknown moves
            return self.__getitem__((idx + 1) % self.__len__())

        return X.astype(np.float16), y

    def _sample_positions(self, total_moves):
        alpha = 1.5  # Adjust alpha to bias towards later moves
        move_probs = (np.arange(1, total_moves + 1) / total_moves) ** alpha
        move_probs /= move_probs.sum()
        move_indices = np.random.choice(
            range(total_moves), size=self.positions_per_game, p=move_probs, replace=False
        )
        move_indices.sort()
        return move_indices