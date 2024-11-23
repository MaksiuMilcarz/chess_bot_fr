// src/mcts.rs
use crate::data::DataWriter;
use crate::game::{GameState, MOVE_INDEX_MAP};
use crate::neural_net::NeuralNetwork;
use chess::{Board, ChessMove, Color, MoveGen, Piece};
use indexmap::IndexMap;
use tch::{Kind, Tensor, IndexOp};
use std::error::Error;
use std::rc::Rc;
use std::cell::RefCell;

const NUM_SIMULATIONS: usize = 800; // Adjust as needed
const CPUCT: f32 = 1.5; // Exploration constant

pub fn self_play(num_games: usize, model_path: &str, output_path: &str) -> Result<(), Box<dyn Error>> {
    let nn = NeuralNetwork::new(model_path)?;

    // Initialize data writer
    let mut data_writer = DataWriter::new(output_path)?;

    for game_index in 0..num_games {
        println!("Starting game {}", game_index + 1);
        let mut game = GameState::new();
        let mut game_history: Vec<GameState> = Vec::new();

        while !game.is_game_over() {
            let (best_move, root_node) = mcts_search(&game, &nn)?;

            // Collect data
            let board_tensor = game.to_tensor();
            let policy = extract_policy(&root_node);

            // Save the move data
            data_writer.write_move(&board_tensor, &policy)?;

            // Make the move
            game.make_move(best_move);
        }

        // Get the game result
        let mut result = game.get_result();

        // Update data with game result
        data_writer.update_game_result(result)?;

        println!("Game {} finished with result {}", game_index + 1, result);
    }

    // Finalize and close the data writer
    data_writer.close()?;

    Ok(())
}

fn mcts_search(game: &GameState, nn: &NeuralNetwork) -> Result<(ChessMove, Rc<Node>), Box<dyn Error>> {
    let root_state = game.board.clone();
    let root_node = Rc::new(Node::new(root_state, None));

    for _ in 0..NUM_SIMULATIONS {
        let node = tree_policy(root_node.clone(), nn)?;
        let value = node.inner.borrow().value;
        backup(node, value);
    }

    let best_move = select_best_move(root_node.clone());
    Ok((best_move, root_node))
}

fn tree_policy(node: Rc<Node>, nn: &NeuralNetwork) -> Result<Rc<Node>, Box<dyn Error>> {
    let mut current_node = node;

    while !current_node.is_terminal() {
        if !current_node.is_fully_expanded() {
            return current_node.expand(nn);
        } else {
            current_node = current_node.select_child();
        }
    }

    Ok(current_node)
}

fn backup(node: Rc<Node>, value: f32) {
    let mut current_node = Some(node);
    let mut v = value;

    while let Some(node) = current_node {
        let mut node_mut = node.inner.borrow_mut();
        node_mut.visit_count += 1;
        node_mut.total_value += v;
        v = -v; // Alternate value for the opponent
        current_node = node.parent.clone();
    }
}

fn select_best_move(root_node: Rc<Node>) -> ChessMove {
    let root = root_node.inner.borrow();
    root.children
        .iter()
        .max_by_key(|(_, child)| child.inner.borrow().visit_count)
        .map(|(mv, _)| *mv)
        .expect("No valid moves found")
}

fn extract_policy(root_node: &Rc<Node>) -> Vec<f32> {
    let root = root_node.inner.borrow();
    let total_visits = root.children.values().map(|child| child.inner.borrow().visit_count).sum::<u32>() as f32;
    let mut policy = vec![0.0; 4672]; // Assuming 4672 possible moves

    for (mv, child) in &root.children {
        let index = move_to_index(mv);
        let visit_count = child.inner.borrow().visit_count as f32;
        policy[index] = visit_count / total_visits;
    }

    policy
}

fn move_to_index(mv: &ChessMove) -> usize {
    let uci_move = mv.to_string();
    *MOVE_INDEX_MAP.get(&uci_move).expect("Move not found in index map")
}

struct Node {
    inner: RefCell<NodeInner>,
    parent: Option<Rc<Node>>,
}

struct NodeInner {
    state: Board,
    visit_count: u32,
    total_value: f32,
    prior: f32,
    value: f32,
    children: IndexMap<ChessMove, Rc<Node>>,
    untried_moves: Vec<ChessMove>,
}

impl Node {
    fn new(state: Board, parent: Option<Rc<Node>>) -> Self {
        let untried_moves = MoveGen::new_legal(&state).collect();
        Self {
            inner: RefCell::new(NodeInner {
                state,
                visit_count: 0,
                total_value: 0.0,
                prior: 0.0,
                value: 0.0,
                children: IndexMap::new(),
                untried_moves,
            }),
            parent,
        }
    }

    fn is_terminal(&self) -> bool {
        self.inner.borrow().untried_moves.is_empty() && self.inner.borrow().children.is_empty()
    }

    fn is_fully_expanded(&self) -> bool {
        self.inner.borrow().untried_moves.is_empty()
    }

    fn expand(self: &Rc<Self>, nn: &NeuralNetwork) -> Result<Rc<Node>, Box<dyn Error>> {
        let mut inner = self.inner.borrow_mut();
        let mv = inner.untried_moves.pop().expect("No moves to expand");
        let child_state = inner.state.make_move_new(mv);
        let child_node = Rc::new(Node::new(child_state, Some(self.clone())));
        inner.children.insert(mv, child_node.clone());

        // Evaluate the child node
        let board_tensor = board_to_tensor(&child_node.inner.borrow().state);
        let (policy_vec, value) = nn.predict(&board_tensor)?;
        {
            let mut child_inner = child_node.inner.borrow_mut();
            child_inner.prior = policy_vec[move_to_index(&mv)]; // Prior probability from policy network
            child_inner.value = value;
        }

        Ok(child_node)
    }

    fn select_child(&self) -> Rc<Node> {
        let inner = self.inner.borrow();
        let parent_visit_count = inner.visit_count as f32;
        inner
            .children
            .values()
            .max_by(|a, b| {
                let a_score = a.ucb_score(parent_visit_count);
                let b_score = b.ucb_score(parent_visit_count);
                a_score.partial_cmp(&b_score).unwrap()
            })
            .cloned()
            .expect("No children to select")
    }

    fn ucb_score(&self, parent_visit_count: f32) -> f32 {
        let inner = self.inner.borrow();
        let q_value = if inner.visit_count > 0 {
            inner.total_value / inner.visit_count as f32
        } else {
            0.0
        };
        let u_value = CPUCT * inner.prior * (parent_visit_count.sqrt() / (1.0 + inner.visit_count as f32));
        q_value + u_value
    }
}

fn board_to_tensor(board: &Board) -> Tensor {
    // Similar to GameState::to_tensor
    let mut matrix = Tensor::zeros(&[1, 13, 8, 8], (Kind::Float, tch::Device::Cpu));

    for square in chess::ALL_SQUARES {
        if let Some(piece) = board.piece_on(square) {
            let piece_type = match piece {
                Piece::Pawn => 0,
                Piece::Knight => 1,
                Piece::Bishop => 2,
                Piece::Rook => 3,
                Piece::Queen => 4,
                Piece::King => 5,
            };

            let color_offset = if board.color_on(square) == Some(Color::White) {
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
    let side_to_move_plane = if board.side_to_move() == Color::White { 1.0 } else { 0.0 };
    matrix.i((0, 12, .., ..)).fill_(side_to_move_plane);

    matrix
}