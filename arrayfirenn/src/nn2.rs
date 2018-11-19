use af::*;

/*fn accuracy(predicted: &Array<f32>, target: &Array<f32>) -> f32{
    let val: Array<f32>; 
    let plabels: Array<f32>;
    let tlabels: Array<f32>;
    max(val, tlabels, target, 1);
    max(val, plabels, predicted, 1);
    return 100.0 * count(plabels == tlabels) / tlabels.elements();
    
}*/

fn derivative(out: &Array<f32>) -> Array<f32>{
    out * (constant(1, out.dims()) - out)
}

fn error(out: &Array<f32>, pred: &Array<f32>) -> f64{
    let dif = out - pred;
    sum_all(&pow(&dif, &2, true)).0.sqrt()
}

pub struct ANN{
    num_layers: u32,
    weights: Vec<Array<f32>>
}

impl ANN{
    fn add_bias(input: &Array<f32>) -> Array<f32>{
        let dims =  Dim4::new(&[input.dims().get()[0], 1, 1, 1]);
        let one_const = constant(1.0f32, dims);
        join(1, &one_const , input)
    }

    fn forward_propagate(&self, input: &Array<f32>) -> Vec<Array<f32>>{
        let mut signal = Vec::with_capacity(self.num_layers as usize);
        signal[0] = input.copy(); //MAYBE THIS COPY DOESN'T WORK
        for i in 0..self.num_layers - 1{
            let input = ANN::add_bias(&signal[i as usize]);
            let output = matmul(&input, &self.weights[i as usize], MatProp::NONE, MatProp::NONE);
            signal[(i + 1) as usize] = sigmoid(&output);
        }
        signal
    }

    fn back_propagate(&self, signal: Vec<Array<f32>>, target: &Array<f32>, alpha: f64){
        let out = &signal[(self.num_layers - 1) as usize];
        let error = out - target;
        let m = target.dims().get()[0];

        for let i = 
    }
}