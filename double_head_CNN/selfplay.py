import rust_magic

def main():
    num_games = 10
    model_path = '/Users/maksiuuuuuuu/Desktop/MyWork/game_bots/chess_bot_fr/models/chessbot_model.pt'
    output_path = '/Users/maksiuuuuuuu/Desktop/MyWork/game_bots/chess_bot_fr/data/self_play_data.parquet'

    # Call the Rust function
    rust_magic.self_play(num_games, model_path, output_path)

if __name__ == "__main__":
    main()