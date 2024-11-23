// src/data.rs
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use tch::Tensor;

#[derive(Serialize, Deserialize)]
struct TrainingSample {
    state: Vec<f32>, // Flattened state tensor
    policy: Vec<f32>,
    value: f32,
}

pub struct DataWriter {
    writer: BufWriter<File>,
    current_game_samples: Vec<TrainingSample>,
}

impl DataWriter {
    pub fn new(output_path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::create(output_path)?;
        let writer = BufWriter::new(file);
        Ok(Self {
            writer,
            current_game_samples: Vec::new(),
        })
    }

    pub fn write_move(&mut self, board_tensor: &Tensor, policy: &[f32]) -> Result<(), Box<dyn Error>> {
        let state: Vec<f32> = board_tensor.view(-1).into();
        let sample = TrainingSample {
            state,
            policy: policy.to_vec(),
            value: 0.0, // Placeholder
        };
        self.current_game_samples.push(sample);
        Ok(())
    }

    pub fn update_game_result(&mut self, mut result: f32) -> Result<(), Box<dyn Error>> {
        for sample in &mut self.current_game_samples {
            sample.value = result;
            result = -result; // Switch perspective for the next move
        }

        // Write samples to file
        for sample in &self.current_game_samples {
            let json = serde_json::to_string(sample)?;
            writeln!(self.writer, "{}", json)?;
        }

        // Clear current game samples
        self.current_game_samples.clear();
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), Box<dyn Error>> {
        self.writer.flush()?;
        Ok(())
    }
}