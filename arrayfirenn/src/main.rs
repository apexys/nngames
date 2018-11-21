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

mod nn2; 
//mod nn3;
use self::nn2::ANN;


#[allow(unused_must_use)]
fn main() {
    set_backend(Backend::CPU);
    println!("{} compute devices found", device_count());
    set_device(0);
    info();
    print!("Info String:\n{}", info_string(true));
    println!("Arrayfire version: {:?}", get_version());
    let (name, platform, toolkit, compute) = device_info();
    print!("Name: {}\nPlatform: {}\nToolkit: {}\nCompute: {}\n", name, platform, toolkit, compute);
    println!("Revision: {}", get_revision());

    println!("Creating test network");
    let mut ann = ANN::new(vec![2,3,1], 0.5);

    let ar = |v: Vec<f32>| Array::new(&v, Dim4::new(&[v.len() as u64, 1,1,1]));
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

    let ar: Array<f32> = randu(Dim4::new(&[4,3,1,1]));
    let mat: Array<f32> = randu(Dim4::new(&[3,4,1,1]));
    let res = matmul(&ar, &mat, MatProp::NONE, MatProp::NONE);
    print(&res);

}
