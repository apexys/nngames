#![feature(duration_as_u128)]
extern crate nn;
extern crate image;
extern crate imageproc;
extern crate rand;
extern crate rayon;
use nn::{NN, HaltCondition};
use rayon::prelude::*;
use std::fs;

use std::time::*;

pub mod inputdata;


mod demodata;


fn main() {
    println!("Loading glyphlib from disk");
    let mut now = SystemTime::now();
    let glyphlib = demodata::load_glyphlib();
    println!("Loading glyphlib took {}ms", now.elapsed().unwrap().as_millis());
    let demodatacount = 10000;
    println!("Generating demo data ({} pcs)", demodatacount);
    now = SystemTime::now();
    let testdata = demodata::create_testdata_multiple(&glyphlib, demodatacount);
    testdata[0].image.save("testdata.png").unwrap();
    println!("Generating testdata took {}ms", now.elapsed().unwrap().as_millis());

    println!("Converting testdata into form suitable for NN");
    now = SystemTime::now();
    let examples = testdata.into_par_iter().map(|td| (td.unpacked_pixels, vec![td.glyphid, td.gx, td.gy, td.rotation])).collect::<Vec<(Vec<f64>, Vec<f64>)>>();
    println!("Converting testdata took {}ms", now.elapsed().unwrap().as_millis());

    println!("Creating network");
    now = SystemTime::now();
    let mut net = NN::new(&[128*128, 64 * 64, 32 * 32, 16 * 16, 4]);
    println!("Creating network took {}ms", now.elapsed().unwrap().as_millis());

    println!("Training network");
    now = SystemTime::now();
    net.train(&examples)
    .halt_condition( HaltCondition::Epochs(10000) )
    .log_interval( Some(10) )
    .momentum( 0.1 )
    .rate( 0.3 )
    .go();
    println!("Training network took {}ms", now.elapsed().unwrap().as_millis());


    println!("Writing network");
    fs::write("network.json", &net.to_json()).unwrap();


    println!("All done");
}