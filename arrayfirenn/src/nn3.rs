use af::*;


pub struct ANN{
    layers: Vec<Array<f32>>,
    num_inputs: usize,
    num_outputs: usize,
    num_layers: usize
}

impl ANN{
    pub fn new(layers: Vec<u64>) -> ANN{
        const RANGE: f32 = 0.5f32;
        let layer_arrays = layers[0..(layers.len() - 1)].iter()
            .zip(&layers[1..layers.len()])
            .map(|(layer, &next)| {
                let array: Array<f32> = randu(Dim4::new(&[next, layer + 1 /*+1 is for the bias at element zero*/, 1,1]));
                (array * RANGE) - (RANGE / 2f32) //Scale weights linearly throuout the range
            }).collect::<Vec<Array<f32>>>();

            ANN{
                layers: layer_arrays,
                num_inputs: layers[0] as usize,
                num_outputs: layers[layers.len() - 1] as usize,
                num_layers: layers.len()
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
    }

    fn add_bias(input: &Array<f32>) -> Array<f32> {
        join(0, &constant(1f32,Dim4::new(&[1,1,1,1])), input)
    }

    pub fn forward_propagate(&self, input: &Array<f32>) -> Vec<Array<f32>>{
        let mut signal = Vec::with_capacity(self.num_layers);
        signal.push(input.copy());
        let mut layer_in = input.copy();
        for l in &self.layers{
            layer_in = ANN::vec_mat_mul(&ANN::add_bias(&layer_in), l);
            signal.push(layer_in.copy());
        }
        signal
    }

    pub fn predict(&self, input: &Array<f32>) -> Array<f32>{
        self.forward_propagate(input)[self.num_layers - 1].copy()
    }

    fn derivative(out: &Array<f32>) -> Array<f32>{
        out * (constant(1, out.dims()) - out)
    }

    pub fn back_propagate(&mut self, signal: Vec<Array<f32>>, target: &Array<f32>, alpha: f32){
        //Get error for output layer
        let mut out = &signal[(self.num_layers - 1) as usize];
        let mut error = out - target;
        println!("Error: ");
        print(&error);
        let m = target.dims().get()[0];
        println!("Num layers: {}", self.num_layers);

        for i in (0 .. self.num_layers -1).rev(){
            println!("i: {}", i);
            let input = ANN::add_bias(&signal[i as usize]);
            let delta = ANN::derivative(out) * &error; //How much is each output wrong by?
            println!("Delta: ");
            print(&delta);
            println!("Weights: ");
            let weights = &self.layers[i as usize];
            print(&weights);
            let wd = weights.dims();
            let wdims = wd.get();
            println!("Indims: {:?}", &input.dims());
            println!("Deltadims: {:?}", delta.dims());
            println!("Delta (expanded):");
            let delta_expanded = tile(&delta, Dim4::new(&[1, wdims[1],1,1]));
            print(&delta_expanded);
            println!("Delta_expanded_dims: {:?}", delta_expanded.dims());
            let input_expanded = tile(&transpose(&input,false), Dim4::new(&[wdims[0], 1, 1, 1]));
            print(&input_expanded);
            let grad = -(input_expanded*delta_expanded * alpha) / m;
            self.layers[i as usize] += grad;
            //let grad = -(input * &tile(&delta, Dim4::new(&[1, wdims[1],1,1])) * alpha ) / m;
            //println!("Grad: ");
            //print(&grad);
            break;
            //Adjust weights
            /*let grad = -(matmul(&delta, &input, MatProp::NONE, MatProp::NONE) * alpha) / m;
            self.layers[i as usize] += transpose(&grad, false);

            //Input to current layer is output of previous
            out = &signal[i as usize];
            error = matmul(&transpose(&delta, false), &transpose(&self.layers[i as usize], false), MatProp::NONE, MatProp::NONE);

            //remove the error of bias and propagate backwards
            error = cols(&error, 1, error.dims().get()[0]);*/
        }
    }
}