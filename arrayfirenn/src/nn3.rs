use af::*;
use std::cmp::Ordering;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct ANN{
    pub layers: Vec<Array<f32>>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub num_layers: usize
}

#[derive(Serialize, Deserialize)]
struct HostArray{
    pub dims: [u64;4],
    pub data: Vec<f32>
}

#[derive(Serialize, Deserialize)]
struct HostANN{
    pub layers: Vec<HostArray>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub num_layers: usize
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

    pub fn print(&self){
        print!("ANN: ");
        let mut layercount = 0;
        for l in &self.layers{
            layercount += 1;
            af_print!(format!("Layer {}", layercount), l);
        }
    }


    pub fn save<'a>(&self, file: &'a str){
        let array_to_host_array = |array:&Array<f32>| {
            let dtemp = array.dims();
            let dims = dtemp.get();
            println!("Allocating buffer of size {}", array.elements());
            let mut buffer: Vec<f32> = Vec::new();
            buffer.resize(array.elements(), 0f32);
            println!("Copying to buffer");
            array.host(&mut buffer);
            HostArray{
                dims: *dims,
                data: buffer
            }
        };

        let host_ann = HostANN{
            layers: self.layers.iter().map(array_to_host_array).collect::<Vec<HostArray>>(),
            num_inputs: self.num_inputs,
            num_layers: self.num_layers,
            num_outputs: self.num_outputs
        };

        let f = std::fs::File::create(file).unwrap();
        bincode::serialize_into(f, &host_ann).unwrap();

    }

    pub fn load(file: &'static str) -> ANN{
        let host_array_to_array = |ha: &HostArray| {
            Array::new(&ha.data, Dim4::new(&ha.dims))
        };

        let f = std::fs::File::open(file).unwrap();
        let temp_ann: HostANN = bincode::deserialize_from(f).unwrap();

        ANN{
            layers: temp_ann.layers.iter().map(host_array_to_array).collect::<Vec<Array<f32>>>(),
            num_inputs: temp_ann.num_inputs,
            num_layers: temp_ann.num_layers,
            num_outputs: temp_ann.num_outputs
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
        for l in &self.layers{
            signal.push(sigmoid(&ANN::vec_mat_mul(&ANN::add_bias(&signal[signal.len() - 1]), l)));
        }
        signal
    }

    pub fn predict(&self, input: &Array<f32>) -> Array<f32>{
        self.forward_propagate(&input)[self.num_layers - 1].copy()
    }

    fn derivative(out: &Array<f32>) -> Array<f32>{
        out * (constant(1, out.dims()) - out)
    }

    /*pub fn back_propagate(&mut self, signal: Vec<Array<f32>>, target: &Array<f32>, alpha: f32){
        //Get error for output layer
        let mut out = &signal[(self.num_layers - 1) as usize];
        let mut error = out - target;
        println!("Error: ");
        print(&error);
        let m = target.dims().get()[0];
        println!("Num layers: {}", self.num_layers);

        for i in (0 .. self.num_layers -1).rev(){
            let input = ANN::add_bias(&signal[i as usize]);
            let delta = ANN::derivative(out) * &error; //How much is each output wrong by?
            let weights = &self.layers[i as usize];
            let wd = weights.dims();
            let wdims = wd.get();
            let delta_expanded = tile(&delta, Dim4::new(&[1, wdims[1],1,1]));
            let input_expanded = tile(&transpose(&input,false), Dim4::new(&[wdims[0], 1, 1, 1]));
            let grad = -(input_expanded*delta_expanded * alpha) / m;
            self.layers[i as usize] += grad;
            println!("Updated weights in layer {}", i);
            //let grad = -(input * &tile(&delta, Dim4::new(&[1, wdims[1],1,1])) * alpha ) / m;
            //println!("Grad: ");
            //print(&grad);
            
            //Adjust weights
            /*let grad = -(matmul(&delta, &input, MatProp::NONE, MatProp::NONE) * alpha) / m;
            self.layers[i as usize] += transpose(&grad, false);

            //Input to current layer is output of previous
            out = &signal[i as usize];
            error = matmul(&transpose(&delta, false), &transpose(&self.layers[i as usize], false), MatProp::NONE, MatProp::NONE);

            //remove the error of bias and propagate backwards
            error = cols(&error, 1, error.dims().get()[0]);*/
        }
    }*/

    fn gen_offspring(network: &ANN, offspring: usize, mutation_speed: f32) -> Vec<ANN>{
        (0 .. offspring).map(|_| {
            ANN{
                layers: network.layers.iter().map(|l:&Array<f32>| {
                    let random_change: Array<f32> = randn(Dim4::new(&[l.dims().get()[0],l.dims().get()[1],1,1]));
                    let layer = (l + (random_change * mutation_speed) - (mutation_speed / 2f32));
                    layer.eval();
                    layer
                }).collect::<Vec<Array<f32>>>(),
                ..*network
            }
        }).collect::<Vec<ANN>>()
    }

    /*fn gen_offspring_host(network: &ANN, offspring: usize, mutation_speed: f32) -> Vec<ANN>{

    }*/

    fn test_on(network: &ANN,  testdata: &Vec<(Array<f32>, Array<f32>)>) -> f32{
        testdata.iter().map(|(test, expected)| {
            let result = network.predict(test);
            let error = sum_all(&abs(&(result - expected))).0 as f32;
            return error;
        }).fold(0f32, |prev, next| prev + next)
    }


    pub fn evo_train(&self, testdata: &Vec<(Array<f32>, Array<f32>)>, population: usize, offspring: usize, mutation_speed: f32, generations: u64, generational_callback: Box<Fn(u64, ANN) -> ()>) -> ANN{
        let mut zoo = ANN::gen_offspring(self, population, mutation_speed);

        let fcomp = |f1, f2| {
            if f1 < f2{
                return Ordering::Less;
            }else if f2 > f1{
                return Ordering::Greater;
            }else{
                return Ordering::Equal;
            }
        };

        for gen in 0 .. generations{
            println!("Starting test on generation {}", gen);
            let mut now = Instant::now();
            let offspring = zoo.iter().map(|specimen| ANN::gen_offspring(specimen, offspring / population, mutation_speed)).flatten().collect::<Vec<ANN>>();
            println!("Generation took {}", now.elapsed().as_millis());
            now = Instant::now();
            let mut test_result = zoo.into_iter().chain(offspring).map(|spec| (ANN::test_on(&spec, testdata), spec)).collect::<Vec<(f32, ANN)>>();
            println!("Test took {}", now.elapsed().as_millis());
            now = Instant::now();
            test_result.sort_unstable_by(|sp1, sp2| fcomp(sp1.0, sp2.0));
            println!("Sort took {}", now.elapsed().as_millis());
            println!("Best error rate of generation {}: {}", gen, test_result[0].0);
            //println!("Best network:");
            //test_result[0].1.print();
            generational_callback(gen, test_result[0].1.clone());
            zoo = test_result.into_iter().take(population).map(|(score, spec)| spec).collect::<Vec<ANN>>();
        }

        return zoo[0].clone();
    }
}