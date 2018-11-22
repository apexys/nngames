use af::*;


pub struct ANN{
    layers: Vec<Array<f32>>,
    num_inputs: usize,
    num_outputs: usize,
    num_layers: usize
}

impl ANN{
    fn new(layers: Vec<u64>) -> ANN{
        const RANGE: f32 = 0.5f32;
        let layer_arrays = layers[0..(layers.len() - 1)].iter()
            .zip(&layers[1..layers.len()])
            .map(|(layer, &next)| {
                let array: Array<f32> = randu(Dim4::new(&[next + 1, /*+1 is for the bias at element zero*/, layer + 1, 1,1]));
                (array * RANGE) - (RANGE / 2f32) //Scale weights linearly throuout the range
            }).collect::<Vec<Array<f32>>>();

            ANN{
                layers: layer_arrays,
                num_inputs: layers[0] as usize,
                num_outputs: layers[layers.len() - 1] as usize,
                num_layers = layers.len()
            }
    }

    //Helper function for multiplying an input vector with a matrix of weights
    fn vec_mat_mul(vec: &Array<f32>, mat: &Array<f32>) -> Array<f32> {
        let vd = vec.dims();
        let md = mat.dims();
        let vecdims = vd.get();
        let matdims = md.get();
        let temp_vec = join(1, &vec ,&constant(0f32, Dim4::new(&[vecdims[0], matdims[0] - vecdims[1],1,1])));
        col(&matmul(
            &mat,
            &temp_vec,
            MatProp::NONE,
            MatProp::NONE), 0)
    };

    fn add_bias(input: &Array<f32>) -> Array<f32> {
        join(0, &constant(1f32,Dim4::new(&[1,1,1,1])), input[0])
    }

    fn forward_propagate(&self, input: &Array<f32>) -> Vec<Array<f32>>{
        let signal = Vec::with_capacity(self.num_layers);
        let mut layer_in = inputs;
        signal.push(layer_in);
        for w in &self.weights{
            layer_in = vec_mat_mul(&add_bias(layer_in), w);
            signal.push(layer_in);
        }
        signal
    }

    fn predict(&self, input: &Array<f32>) -> Vec<Array<f32>>{
        self.forward_propagate(input)[self.num_layers - 1];
    }
}