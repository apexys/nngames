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

    fn back_propagate(&mut self, signal: Vec<Array<f32>>, target: &Array<f32>, alpha: f32){
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
            error = cols(&error, 1, error.dims().get()[0]);
        }
    }

    pub fn new(layers: Vec<u64>, range: f32)-> ANN{
        let mut weights: Vec<Array<f32>> = Vec::with_capacity(layers.len() -1);
        for i in 0..(layers.len()-1){
            let r: Array<f32> = randu(Dim4::new(&[layers[i] + 1, layers[i+1], 1,1]));
            weights.push( r * range - (range / 2.0));
        }

        ANN{
            num_layers: layers.len() as u32,
            weights
        }
    }

    pub fn predict(&self, input: &Array<f32>) -> Array<f32>{
        let signal = self.forward_propagate(input);
        let output = &signal[(self.num_layers  -1) as usize];
        return output.copy();
    }

    pub fn train(&mut self, input: &Array<f32>, target: &Array<f32>, alpha: f32, max_epochs: u64, batch_size: u64, max_err: f64, verbose: bool) -> f64{
        println!("{:?}", input.dims().get()[1]);
        let num_samples = input.dims().get()[1];
        let num_batches = num_samples / batch_size;

        let error = 0;

        for i in 0..max_epochs {
            for j in 0 .. (num_batches-1){
                let st = (j * batch_size) as f32;
                let en = (st + (batch_size - 1) as f32) as f32;
                println!("st: {}, en: {}", st, en);
                let x = index(&input, &[Seq::new(st, en,1.0)]);
                let y = index(&input, &[Seq::new(st, en,1.0)]);


                println!("Propagate {}", self.num_layers);
                //Propagate the inputs forward
                print(&x);
                let signals = self.forward_propagate(&x);
                println!("Signal length:{}", signals.len());
                let out = &signals[(self.num_layers - 1) as usize];
                println!("Prop error");
                //Propagate the error backward
                self.back_propagate(signals, &y, alpha);
            }

            let st = (num_batches - 1) as f32 * batch_size as f32;
            let en = (num_samples - 1) as f32;
            let out = self.predict(&index(&input, &[Seq::new(st, en,1.0)]));
            let err = ANN::error(&out, &index(&input, &[Seq::new(st, en,1.0)]));

            if err < max_err {
                println!("Converged on Epoch {}", i + 1);
                return err;
            }

            if verbose {
                if ((i+1) % 10) == 0{
                    println!("Epoch: {}, Error: {}", i + 1, err);
                }
            }

        }



        1.1

    }


}