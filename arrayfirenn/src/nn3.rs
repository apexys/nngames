use af::*;


pub struct ANN{
    weights: Vec<Array<f32>>,
    num_inputs: usize,
    num_outputs: usize
}

impl ANN{
    fn new(layers: Vec<u64>) -> ANN{
        const RANGE: f32 = 0.5f32;
        let arrays_for_weights = layers[0..(layers.len() - 1)].iter()
            .zip(&layers[1..layers.len()])
            .map(|(layer, &next)| {
                let array: Array<f32> = randu(Dim4::new(&[layer + 1 /*+1 is for the bias at element zero*/, next, 1,1]));
                (array * RANGE) - (RANGE / 2f32) //Scale weights linearly throuout the range
            }).collect::<Vec<Array<f32>>>();

            ANN{
                weights: arrays_for_weights,
                num_inputs: layers[0] as usize,
                num_outputs: layers[layers.len() - 1] as usize
            }
    }

    fn forward_propagate(&self, inputs: &Array<f32>) -> Array<f32>{
        let signal = Vec::new();
        signal.push(inputs);
        for w in &self.weights {
            let bias_and_input = join(0, &constant(1f32,Dim4::new(&[1,1,1,1])), signal[0]); //Add a bias node at the START of the input, like this [1, ...inputs]
            
        }

        randu(Dim4::new(&[1,1,1,1]))
    }
}