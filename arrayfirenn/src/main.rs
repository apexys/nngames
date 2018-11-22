#![feature(extern_crate_item_prelude)]

#[macro_use]
extern crate serde_derive;
extern crate bincode;

#[macro_use(af_print)]
extern crate arrayfire as af;
extern crate num;
use af::*;

/*pub mod nn;
pub mod nn_trainer;*/

//mod nn2; 
mod nn3;
use self::nn3::ANN;


#[allow(unused_must_use)]
fn main() {
    set_backend(Backend::OPENCL);
    println!("{} compute devices found", device_count());
    set_device(0);
    info();
    print!("Info String:\n{}", info_string(true));
    println!("Arrayfire version: {:?}", get_version());
    let (name, platform, toolkit, compute) = device_info();
    print!("Name: {}\nPlatform: {}\nToolkit: {}\nCompute: {}\n", name, platform, toolkit, compute);
    println!("Revision: {}", get_revision());

    println!("Creating test network");
    let mut ann = ANN::new(vec![2,3,2]);

    let ar = |v: Vec<f32>| Array::new(&v, Dim4::new(&[v.len() as u64, 1,1,1]));

    let test_input = ar(vec![0.0,1.0]);

    println!("Prediction: ");
    let pred = ann.predict(&test_input);
    print(&pred);

    ann.back_propagate(ann.forward_propagate(&test_input), &ar(vec![1.0, 0.0]), 0.01);

    /*
    println!("Creating test data");
    let testdata_input = vec![
        (ar(vec![0.0, 0.0]), ar(vec![0.0])),
        (ar(vec![0.0, 1.0]), ar(vec![1.0])),
        (ar(vec![1.0, 0.0]), ar(vec![1.0])),
        (ar(vec![1.0, 1.0]), ar(vec![0.0]))
    ];
       
    println!("Training network");

    ann.train_vec(testdata_input, 2.0, 20000, 0.01, true);

    println!("Training done");
    */
/*
    let input: Array<f32> = randu(Dim4::new(&[2,1,1,1]));
    let weights: Array<f32> = randu(Dim4::new(&[1,2,1,1]));
    let output = matmul(&weights, &input, MatProp::NONE, MatProp::NONE);

    print(&input);
    print(&weights);
    print(&output);
*/
/*

    let input: Array<f32> = randu(Dim4::new(&[2,1,1,1]));
    let weights: Array<f32> = randu(Dim4::new(&[3,2,1,1]));

    println!("Weights generated");

    let vec_mat_mul = |vec: &Array<f32>, mat: &Array<f32>| {
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

    println!("Input:");
    print(&input);
    println!("Weights:");
    print(&weights);
    println!("Output");
    print(&vec_mat_mul(&input, &weights));
    //println!("Sum of all outputs: {}", sum_all(&vec_mat_mul(&input, &weights)).0 );
*/
/*
    let ar: Array<f32> = randu(Dim4::new(&[4,3,1,1]));
    println!("Initial array: ");
    let vec: Array<f32> = randu(Dim4::new(&[4,1,1,1]));

    println!("Constant to add: ");
    let d = vec.dims();
    let dim = d.get();
    let d2 = Dim4::new(&[dim[0], ar.dims().get()[1] - dim[1], dim[2], dim[3]]);
    println!("Dimensions to create: {:?}", d.get());
    let con = constant(0f32, d2);
    print(&con);

    println!("Joined: ");
    let joined = join(1, &vec, &con);
    print(&joined);

    println!("Transposed: ");
    let transp = transpose(&vec, false);
    print(&transp);
*/
    /*

    let resize_to = |ar: &Array<f32>, dims: Dim4| moddims(&join(0, &ar, &constant(0f32, Dim4::new(&[dims.get()[1] - ar.dims().get()[1],1,1,1]))), dims);

    let modvec = resize_to(&vec, Dim4::new(&[4,3,1,1]));
    print(&modvec);
    //let mat: Array<f32> = randu(Dim4::new(&[3,4,1,1]));
    //let res = matmul(&modvec, &mat, MatProp::UPPER, MatProp::NONE);
    //print(&res);*/

}
