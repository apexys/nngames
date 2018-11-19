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


#[allow(unused_must_use)]
fn main() {
    set_device(0);
    info();
    print!("Info String:\n{}", info_string(true));
    println!("Arrayfire version: {:?}", get_version());
    let (name, platform, toolkit, compute) = device_info();
    print!("Name: {}\nPlatform: {}\nToolkit: {}\nCompute: {}\n", name, platform, toolkit, compute);
    println!("Revision: {}", get_revision());

    let dims = Dim4::new(&[128*128, 1, 1, 1]);


    let mut nodes = Vec::new();
    let mut values = Vec::new();
    for _ in 0..128*128{
        let a = randu::<f32>(dims);
        nodes.push(a);
        let b = randu::<f32>(dims);
        values.push(b);
    }

    println!("Test values created");

    let result = nodes.iter().zip(values).map(|(node, value)| sum(&(node * value), 0)).collect::<Vec<_>>();
    let res_refs = result.iter().map(|a| a).collect::<Vec<_>>();
    let result_vec =  sigmoid(&join_many(0, res_refs));

    println!("Result calculated");
    print(&result_vec);

}
