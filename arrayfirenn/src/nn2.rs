use af::*;

/*fn accuracy(predicted: &Array<f32>, target: &Array<f32>) -> f32{
    let val: Array<f32>; 
    let plabels: Array<f32>;
    let tlabels: Array<f32>;
    max(val, tlabels, target, 1);
    max(val, plabels, predicted, 1);
    return 100.0 * count(plabels == tlabels) / tlabels.elements();
    
}*/



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

    fn derivative(out: &Array<f32>) -> Array<f32>{
        out * (constant(1, out.dims()) - out)
    }

    fn error(out: &Array<f32>, pred: &Array<f32>) -> f64{
        let dif = out - pred;
        sum_all(&pow(&dif, &2, true)).0.sqrt()
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

    fn back_propagate(&mut self, signal: Vec<Array<f32>>, target: &Array<f32>, alpha: f64){
        //Get error for output layer
        let mut out = &signal[(self.num_layers - 1) as usize];
        let mut error = out - target;
        let m = target.dims().get()[0];

        for i in ((self.num_layers -2) as i32)..-1{
            let input = ANN::add_bias(&signal[i as usize]);
            let delta = transpose(&(ANN::derivative(out) * &error), false);
            //Adjust weights
            let grad = -(matmul(&delta, &input, MatProp::NONE, MatProp::NONE) * alpha) / m;
            self.weights[i as usize] += transpose(&grad, false);

            //Input to current layer is output of previous
            out = &signal[i as usize];
            error = matmul(&transpose(&delta, false), &transpose(&self.weights[i as usize], false), MatProp::NONE, MatProp::NONE);

            //remove the error of bias and propagate backwards
            error = error.cols(1,2);
            error = ANN::error(span, seq(1, out.dims().get()[1]));
        }
    }


}