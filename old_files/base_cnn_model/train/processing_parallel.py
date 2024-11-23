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
from utils import board_to_matrix

def process_pgn_file(file, max_games_per_file):
    local_moves = set()
    total_games_processed = 0
    with open(file, 'r') as pgn_file:
        while total_games_processed < max_games_per_file:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            for move in game.mainline_moves():
                local_moves.add(move.uci())
            total_games_processed += 1
    return local_moves

def collect_unique_moves_parallel(files, max_games):
    num_files = len(files)
    max_games_per_file = max_games // num_files + 1  # Ensure we cover all games

    # Create a partial function with max_games_per_file fixed
    process_file_partial = partial(process_pgn_file, max_games_per_file=max_games_per_file)

    # Use multiprocessing Pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_file_partial, files)

    # Merge the results
    all_moves = set().union(*results)
    move_to_int = {move: idx for idx, move in enumerate(sorted(all_moves))}
    num_classes = len(move_to_int)
    return move_to_int, num_classes

def preprocess_and_save_to_hdf5_parallel(files, move_to_int, max_games, positions_per_game=10):
    num_workers = mp.cpu_count()
    max_games_per_worker = max_games // num_workers + 1
    total_samples_estimate = max_games * positions_per_game

    manager = mp.Manager()
    task_queue = manager.Queue(maxsize=3072)  # Adjust maxsize based on your system

    # Split files among workers
    files_per_worker = [[] for _ in range(num_workers)]
    for idx, file in enumerate(files):
        files_per_worker[idx % num_workers].append(file)

    # Start the writer process
    hdf5_filename = 'preprocessed_data.h5'
    writer_process = mp.Process(target=writer, args=(task_queue, hdf5_filename, total_samples_estimate))
    writer_process.start()

    # Start worker processes
    workers = []
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(files_per_worker[worker_id], move_to_int, positions_per_game, task_queue, max_games_per_worker))
        workers.append(p)
        p.start()

    # Wait for all workers to finish
    for p in workers:
        p.join()

    # Signal the writer that we're done
    task_queue.put('DONE')

    # Wait for the writer to finish
    writer_process.join()
    
    
def worker(file_list, move_to_int, positions_per_game, task_queue, max_games_per_worker):
    total_games_processed = 0
    batch_X = []
    batch_y = []
    batch_size = 512  # Adjust as needed

    for file in file_list:
        with open(file, 'r') as pgn_file:
            while total_games_processed < max_games_per_worker:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                board = game.board()
                moves = list(game.mainline_moves())
                total_moves = len(moves)

                if total_moves == 0:
                    total_games_processed += 1
                    continue

                # Probabilistic sampling favoring end-game positions
                move_numbers = np.arange(1, total_moves + 1)
                alpha = 1.2  # Adjust alpha as needed
                probabilities = (move_numbers / total_moves) ** alpha
                probabilities /= probabilities.sum()
                positions = np.random.choice(
                    total_moves, size=min(positions_per_game, total_moves),
                    replace=False, p=probabilities
                )
                positions.sort()

                for idx in positions:
                    board_copy = board.copy(stack=False)
                    for move in moves[:idx]:
                        board_copy.push(move)
                    X = board_to_matrix(board_copy)
                    move_uci = moves[idx].uci()
                    y = move_to_int.get(move_uci)
                    if y is None:
                        continue

                    batch_X.append(X)
                    batch_y.append(y)

                    if len(batch_X) >= batch_size:
                        # Send batch to queue
                        task_queue.put((np.array(batch_X), np.array(batch_y)))
                        batch_X = []
                        batch_y = []

                total_games_processed += 1
                if total_games_processed >= max_games_per_worker:
                    break

    # Send any remaining data
    if batch_X:
        task_queue.put((np.array(batch_X), np.array(batch_y)))
        
        
def writer(task_queue, hdf5_filename, total_samples_estimate):
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        X_dataset = hdf5_file.create_dataset('X', shape=(total_samples_estimate, 16, 8, 8),
                                             dtype=np.float32, compression='gzip', chunks=True, maxshape=(None, 16, 8, 8))
        y_dataset = hdf5_file.create_dataset('y', shape=(total_samples_estimate,),
                                             dtype=np.int64, compression='gzip', chunks=True, maxshape=(None,))
        sample_index = 0
        while True:
            item = task_queue.get()
            if item == 'DONE':
                break
            X_batch, y_batch = item
            batch_size = X_batch.shape[0]
            if sample_index + batch_size > X_dataset.shape[0]:
                # Resize datasets if necessary
                new_size = X_dataset.shape[0] + total_samples_estimate
                X_dataset.resize((new_size, 16, 8, 8))
                y_dataset.resize((new_size,))
            X_dataset[sample_index:sample_index+batch_size] = X_batch
            y_dataset[sample_index:sample_index+batch_size] = y_batch
            sample_index += batch_size
        # Resize datasets to actual size
        X_dataset.resize((sample_index, 16, 8, 8))
        y_dataset.resize((sample_index,))
    
    
    
    
