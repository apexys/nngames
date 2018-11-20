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

use self::nn2::ANN;


#[allow(unused_must_use)]
fn main() {
    set_device(0);
    info();
    print!("Info String:\n{}", info_string(true));
    println!("Arrayfire version: {:?}", get_version());
    let (name, platform, toolkit, compute) = device_info();
    print!("Name: {}\nPlatform: {}\nToolkit: {}\nCompute: {}\n", name, platform, toolkit, compute);
    println!("Revision: {}", get_revision());

    println!("Creating test network");
    let mut ann = ANN::new(vec![2,3,1], 0.5);

    println!("Creating test data");
    let testdata_input = vec![
        0.0,0.0,
        0.0,1.0,
        1.0,0.0,
        1.0,1.0
    ];

    let testdata_output = vec![
        0.0,
        1.0,
        1.0,
        0.0
    ];

    let testdata_array_input = Array::new(&testdata_input, Dim4::new(&[2,4,1,1]));
    let testdata_array_output = Array::new(&testdata_output, Dim4::new(&[1,4,1,1]));
       
    println!("Training network");

    ann.train(&testdata_array_input, &testdata_array_output, 2.0, 20000, 2, 0.01, true);

    println!("Training done");

}
