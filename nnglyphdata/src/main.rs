#![feature(duration_as_u128)]
extern crate image;
extern crate imageproc;
extern crate rand;
extern crate rayon;
use std::fs;
use std::path::Path;

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


    let testdata = demodata::create_testdata_multiple_iter(&glyphlib, demodatacount);

    println!("Saving data into folder demodata");
    if !Path::new("demodata").exists(){
        fs::create_dir("demodata").unwrap();
    }
    if !Path::new("demodata/pictures").exists(){
        fs::create_dir("demodata/pictures").unwrap();
    }
    let mut datadescription = String::new();
    for (ctr, data) in testdata{
        datadescription.push_str(&format!{"{}.png\t{}\t{}\t{}\t{}\t{}\n",ctr, data.gx, data.gy, data.glyph_id, data.glyph_present, data.rotation});
        data.image.save(&format!{"demodata/pictures/{}.png", ctr}).unwrap();
        println!("Saved image {}", ctr);
    }
    fs::write("demodata/datadescription.txt", datadescription).unwrap();



    println!("All done");
}