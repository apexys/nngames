use af::*;

use std::fs::File;
use std::error::Error;
use bincode;

use super::nn_trainer::*;
use super::nn_trainer::HaltCondition::{ Epochs, MSE, Timer };
use super::nn_trainer::LearningMode::{ Incremental };
use time::{ Duration, PreciseTime };

pub struct NN{
    layers: Vec<Vec<Array<f32>>>,
    num_inputs: u64,
    num_outputs: u64
}

#[derive(Serialize, Deserialize)]
pub struct NNState{
    layers: Vec<Vec<Vec<f32>>>,
    num_inputs: u64
}

impl NNState{
    pub fn load(file: &str) -> Result<NNState, Box<Error>>{
        let mut f = File::open(file)?;
        Ok(bincode::deserialize_from(f)?)
    }

    pub fn save(&self, file: &str) -> Result<(), Box<Error>>{
        let mut f = File::create(file)?;
        Ok(bincode::serialize_into(f, self)?)
    }
}

impl NN {

    /// Each number in the `layers_sizes` parameter specifies a
    /// layer in the network. The number itself is the number of nodes in that
    /// layer. The first number is the input layer, the last
    /// number is the output layer, and all numbers between the first and
    /// last are hidden layers. There must be at least two layers in the network.
    pub fn new(layers_sizes: &[u64]) -> NN {
        //1. Check input data!
        if layers_sizes.len() < 2 {
            panic!("must have at least two layers");
        }

        for &layer_size in layers_sizes.iter() {
            if layer_size < 1 {
                panic!("can't have any empty layers");
            }
        }

        let mut layers = Vec::new();
        let mut it = layers_sizes.iter();
        // get the first layer size
        let first_layer_size = *it.next().unwrap();

        // setup the rest of the layers
        let mut prev_layer_size = first_layer_size;
        for &layer_size in it {
            let mut layer: Vec<Array<f32>> = Vec::new();
            for _ in 0..layer_size {
                let node_dims = Dim4::new(&[prev_layer_size + 1,1,1,1]);
                let node = randu(node_dims) / 2.0;
                layer.push(node)
            }
            layer.shrink_to_fit();
            layers.push(layer);
            prev_layer_size = layer_size;
        }
        layers.shrink_to_fit();
        NN { layers: layers, num_inputs: first_layer_size, num_outputs: layers_sizes[layers_sizes.len() - 1] }
    }

    pub fn from_state(state: &NNState) -> NN{
        let layers = state.layers.iter()
            .map(|nodes| nodes.iter()
                                .map(|weights| Array::new(&weights, Dim4::new(&[weights.len() as u64, 1, 1, 1])))
                                .collect::<Vec<Array<f32>>>()
            )
            .collect::<Vec<Vec<Array<f32>>>>();

        NN { layers: layers, num_inputs: state.num_inputs, num_outputs: layers[layers.len() - 1].len() as u64 }
    }

    pub fn to_state(&self) -> NNState{
        let layers = self.layers.iter()
        .map(|nodes| nodes.iter()
                                .map(|weights| {
                                    weights.eval(); //Flush any pending ops
                                    let mut weights_buffer = Vec::with_capacity(weights.dims().get()[0] as usize);
                                    weights.host(&mut weights_buffer);
                                    weights_buffer
                                })
                                .collect::<Vec<Vec<f32>>>()
        )
        .collect::<Vec<Vec<Vec<f32>>>>();

        NNState{layers: layers, num_inputs: self.num_inputs}
    }

    /// Runs the network on an input and returns a vector of the results.
    /// The number of `f32`s in the input must be the same
    /// as the number of input nodes in the network. The length of the results
    /// vector will be the number of nodes in the output layer of the network.
    pub fn run(&self, inputs: &[f32]) -> Vec<f32> {
        if inputs.len() != self.num_inputs as usize {
            panic!("input has a different length than the network's input layer");
        }
        self.do_run(inputs).pop().unwrap()
    }

    
    /// Takes in vector of examples and returns a `Trainer` struct that is used
    /// to specify options that dictate how the training should proceed.
    /// No actual training will occur until the `go()` method on the
    /// `Trainer` struct is called.
    pub fn train<'b>(&'b mut self, examples: &'b [(Vec<f32>, Vec<f32>)]) -> Trainer {
        Trainer {
            examples: examples,
            rate: DEFAULT_LEARNING_RATE,
            momentum: DEFAULT_MOMENTUM,
            log_interval: None,
            halt_condition: Epochs(DEFAULT_EPOCHS),
            learning_mode: Incremental,
            nn: self
        }
    }

    pub fn train_details(&mut self, examples: &[(Vec<f32>, Vec<f32>)], rate: f32, momentum: f32, log_interval: Option<u32>,
                    halt_condition: HaltCondition) -> f32 {

        // check that input and output sizes are correct
        let input_layer_size = self.num_inputs;
        let output_layer_size = self.layers[self.layers.len() - 1].len();
        for &(ref inputs, ref outputs) in examples.iter() {
            if inputs.len() as u64 != input_layer_size {
                panic!("input has a different length than the network's input layer");
            }
            if outputs.len() != output_layer_size {
                panic!("output has a different length than the network's output layer");
            }
        }

        self.train_incremental(examples, rate, momentum, log_interval, halt_condition)
    }

    fn train_incremental(&mut self, examples: &[(Vec<f32>, Vec<f32>)], rate: f32, momentum: f32, log_interval: Option<u32>,
                    halt_condition: HaltCondition) -> f32 {

        let mut prev_deltas = self.make_weights_tracker(0.0f32);
        let mut epochs = 0u32;
        let mut training_error_rate = 0f32;
        let start_time = PreciseTime::now();

        loop {

            if epochs > 0 {
                // log error rate if necessary
                match log_interval {
                    Some(interval) if epochs % interval == 0 => {
                        println!("error rate: {}", training_error_rate);
                    },
                    _ => (),
                }

                // check if we've met the halt condition yet
                match halt_condition {
                    Epochs(epochs_halt) => {
                        if epochs == epochs_halt { break }
                    },
                    MSE(target_error) => {
                        if training_error_rate <= target_error { break }
                    },
                    Timer(duration) => {
                        let now = PreciseTime::now();
                        if start_time.to(now) >= duration { break }
                    }
                }
            }

            training_error_rate = 0f32;

            for &(ref inputs, ref targets) in examples.iter() {
                let results = self.do_run(&inputs);
                let weight_updates = self.calculate_weight_updates(&results, &targets);
                training_error_rate += calculate_error(&results, &targets);
                self.update_weights(&weight_updates, &mut prev_deltas, rate, momentum)
            }

            epochs += 1;
        }

        training_error_rate
    }

    //raw run on an input
    fn do_run(&self, inputs: &[f32]) -> Vec<Vec<f32>> {
        let input = Array::new(inputs, Dim4::new(&[inputs.len() as u64, 1,1,1]));
        let mut results = Vec::new();

        results.push(inputs.to_vec());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let mut layer_results = Vec::new();
            for node in layer.iter() {
                layer_results.push( sigmoid(modified_dotprod(&node, &results[layer_index])) )
            }
            results.push(layer_results);
        }
        results
    }

    // updates all weights in the network
    fn update_weights(&mut self, network_weight_updates: &Vec<Vec<Vec<f32>>>, prev_deltas: &mut Vec<Vec<Vec<f32>>>, rate: f32, momentum: f32) {
        for layer_index in 0..self.layers.len() {
            let mut layer = &mut self.layers[layer_index];
            let layer_weight_updates = &network_weight_updates[layer_index];
            for node_index in 0..layer.len() {
                let mut node = &mut layer[node_index];
                let node_weight_updates = &layer_weight_updates[node_index];
                for weight_index in 0..node.len() {
                    let weight_update = node_weight_updates[weight_index];
                    let prev_delta = prev_deltas[layer_index][node_index][weight_index];
                    let delta = (rate * weight_update) + (momentum * prev_delta);
                    node[weight_index] += delta;
                    prev_deltas[layer_index][node_index][weight_index] = delta;
                }
            }
        }

    }

    // calculates all weight updates by backpropagation
    fn calculate_weight_updates(&self, results: &Vec<Vec<f32>>, targets: &[f32]) -> Vec<Vec<Vec<f32>>> {
        let mut network_errors:Vec<Vec<f32>> = Vec::new();
        let mut network_weight_updates = Vec::new();
        let layers = &self.layers;
        let network_results = &results[1..]; // skip the input layer
        let mut next_layer_nodes: Option<&Vec<Vec<f32>>> = None;

        for (layer_index, (layer_nodes, layer_results)) in iter_zip_enum(layers, network_results).rev() {
            let prev_layer_results = &results[layer_index];
            let mut layer_errors = Vec::new();
            let mut layer_weight_updates = Vec::new();


            for (node_index, (node, &result)) in iter_zip_enum(layer_nodes, layer_results) {
                let mut node_weight_updates = Vec::new();
                let mut node_error;

                // calculate error for this node
                if layer_index == layers.len() - 1 {
                    node_error = result * (1f32 - result) * (targets[node_index] - result);
                } else {
                    let mut sum = 0f32;
                    let next_layer_errors = &network_errors[network_errors.len() - 1];
                    for (next_node, &next_node_error_data) in next_layer_nodes.unwrap().iter().zip((next_layer_errors).iter()) {
                        sum += next_node[node_index+1] * next_node_error_data; // +1 because the 0th weight is the threshold
                    }
                    node_error = result * (1f32 - result) * sum;
                }

                // calculate weight updates for this node
                for weight_index in 0..node.len() {
                    let mut prev_layer_result;
                    if weight_index == 0 {
                        prev_layer_result = 1f32; // threshold
                    } else {
                        prev_layer_result = prev_layer_results[weight_index-1];
                    }
                    let weight_update = node_error * prev_layer_result;
                    node_weight_updates.push(weight_update);
                }

                layer_errors.push(node_error);
                layer_weight_updates.push(node_weight_updates);
            }

            network_errors.push(layer_errors);
            network_weight_updates.push(layer_weight_updates);
            next_layer_nodes = Some(&layer_nodes);
        }

        // updates were built by backpropagation so reverse them
        network_weight_updates.reverse();

        network_weight_updates
    }

    fn make_weights_tracker<T: Clone>(&self, place_holder: T) -> Vec<Vec<Vec<T>>> {
        let mut network_level = Vec::new();
        for layer in self.layers.iter() {
            let mut layer_level = Vec::new();
            for node in layer.iter() {
                let mut node_level = Vec::new();
                for _ in node.iter() {
                    node_level.push(place_holder.clone());
                }
                layer_level.push(node_level);
            }
            network_level.push(layer_level);
        }

        network_level
    }
}

fn modified_dotprod(node: &Vec<f32>, values: &Vec<f32>) -> f32 {
    let mut it = node.iter();
    let mut total = *it.next().unwrap(); // start with the threshold weight
    for (weight, value) in it.zip(values.iter()) {
        total += weight * value;
    }
    total
}

fn sigmoid(y: f32) -> f32 {
    1f32 / (1f32 + (-y).exp())
}


// takes two arrays and enumerates the iterator produced by zipping each of
// their iterators together
fn iter_zip_enum<'s, 't, S: 's, T: 't>(s: &'s [S], t: &'t [T]) ->
    Enumerate<Zip<slice::Iter<'s, S>, slice::Iter<'t, T>>>  {
    s.iter().zip(t.iter()).enumerate()
}

// calculates MSE of output layer
fn calculate_error(results: &Vec<Vec<f32>>, targets: &[f32]) -> f32 {
    let ref last_results = results[results.len()-1];
    let mut total:f32 = 0f32;
    for (&result, &target) in last_results.iter().zip(targets.iter()) {
        total += (target - result).powi(2);
    }
    total / (last_results.len() as f32)
}