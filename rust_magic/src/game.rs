use chess::{Board, ChessMove, Color, Piece, ALL_SQUARES};
use tch::{Kind, Tensor, IndexOp};
use std::collections::HashMap;
use once_cell::sync::Lazy;

// Global move index mapping
pub static MOVE_INDEX_MAP: Lazy<HashMap<String, usize>> = Lazy::new(|| {
    let mut map = HashMap::new();
    let mut index = 0;

    let promotion_pieces = [
        Piece::Queen,
        Piece::Rook,
        Piece::Bishop,
        Piece::Knight,
    ];

    for from in ALL_SQUARES {
        for to in ALL_SQUARES {
            // Normal moves and captures
            if from != to {
                // Generate normal move
                let mv = ChessMove::new(from, to, None);
                map.insert(mv.to_string(), index);
                index += 1;

                // Generate promotion moves (only for pawn promotion squares)
                let from_rank = from.get_rank().to_index();
                let to_rank = to.get_rank().to_index();
                // For white pawn promotions
                if from_rank == 6 && to_rank == 7 {
                    for &promo in &promotion_pieces {
                        let mv = ChessMove::new(from, to, Some(promo));
                        map.insert(mv.to_string(), index);
                        index += 1;
                    }
                }
                // For black pawn promotions
                if from_rank == 1 && to_rank == 0 {
                    for &promo in &promotion_pieces {
                        let mv = ChessMove::new(from, to, Some(promo));
                        map.insert(mv.to_string(), index);
                        index += 1;
                    }
                }
            }
        }
    }

    // Ensure the total number of moves matches the policy vector size
    assert_eq!(index, 4672, "Total moves generated ({}) do not match policy vector size (4672)", index);

    map
});

pub struct GameState {
    pub board: Board,
}

impl GameState {
    pub fn new() -> Self {
        Self {
            board: Board::default(),
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.board.status() != chess::BoardStatus::Ongoing
    }

    pub fn legal_moves(&self) -> Vec<ChessMove> {
        chess::MoveGen::new_legal(&self.board).collect()
    }

    pub fn make_move(&mut self, mv: ChessMove) {
        self.board = self.board.make_move_new(mv);
    }

    pub fn get_result(&self) -> f32 {
        match self.board.status() {
            chess::BoardStatus::Stalemate => 0.0,
            chess::BoardStatus::Checkmate => {
                if self.board.side_to_move() == Color::White {
                    -1.0 // Black wins
                } else {
                    1.0 // White wins
                }
            }
            _ => 0.0,
        }
    }

    pub fn to_tensor(&self) -> Tensor {
        let mut matrix = Tensor::zeros(&[1, 13, 8, 8], (Kind::Float, tch::Device::Cpu));

        // Map the pieces to planes
        for square in chess::ALL_SQUARES {
            if let Some(piece) = self.board.piece_on(square) {
                let piece_type = match piece {
                    Piece::Pawn => 0,
                    Piece::Knight => 1,
                    Piece::Bishop => 2,
                    Piece::Rook => 3,
                    Piece::Queen => 4,
                    Piece::King => 5,
                };

                let color_offset = if self.board.color_on(square) == Some(Color::White) {
                    0
                } else {
                    6
                };
                let plane = (piece_type + color_offset) as i64;

                let rank = 7 - (square.get_rank().to_index() as i64);
                let file = square.get_file().to_index() as i64;

                matrix.i((0, plane, rank, file)).fill_(1.0);
            }
        }

        // Side to move plane
        let side_to_move_plane = if self.board.side_to_move() == Color::White { 1.0 } else { 0.0 };
        matrix.i((0, 12, .., ..)).fill_(side_to_move_plane);

        matrix
    }
}