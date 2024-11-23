use std::error::Error;
use tch::{CModule, Device, Kind, Tensor, IValue};

pub struct NeuralNetwork {
    model: CModule,
    device: Device,
}

impl NeuralNetwork {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        let device = Device::cuda_if_available();
        let model = CModule::load_on_device(model_path, device)?;
        Ok(Self { model, device })
    }

    
    pub fn predict(&self, board_state: &Tensor) -> Result<(Vec<f32>, f32), Box<dyn Error>> {
        let input = board_state.to_device(self.device);

            // Convert the input Tensor into an IValue
        let input_ivalue = IValue::Tensor(input);

        // Use forward_is to get an IValue
        let output = self.model.forward_is(&[input_ivalue])?;

        // Match the output to extract the tuple
        if let IValue::Tuple(outputs) = output {
            if outputs.len() != 2 {
                return Err("Expected output tuple of length 2".into());
            }
            // Extract policy_log_probs
            let policy_log_probs = match &outputs[0] {
                IValue::Tensor(tensor) => tensor,
                _ => return Err("Expected policy_log_probs to be a Tensor".into()),
            };
            // Extract value
            let value_tensor = match &outputs[1] {
                IValue::Tensor(tensor) => tensor,
                _ => return Err("Expected value to be a Tensor".into()),
            };
        
            // Exponentiate log-probabilities to get probabilities
            let policy_probs = policy_log_probs.exp().to(Device::Cpu);
            let value_tensor = value_tensor.to(Device::Cpu);

            // Convert policy_probs to Vec<f32>
            let policy_vec: Vec<f32> = policy_probs.into();
            // Extract the scalar value
            let value_scalar = value_tensor.double_value(&[0]) as f32;

            Ok((policy_vec, value_scalar))
        } else {
            Err("Model output is not a tuple.".into())
        }
    }
}