use pyo3::prelude::*;

mod mcts;
mod neural_net;
mod game;
mod data;

#[pyfunction]
fn self_play(num_games: usize, model_path: &str, output_path: &str) -> PyResult<()> {
    mcts::self_play(num_games, model_path, output_path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Error in self_play: {:?}", e))
    })
}

#[pymodule]
fn my_chess_bot(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(self_play, m)?)?;
    Ok(())
}