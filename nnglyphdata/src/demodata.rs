const glyphpath: &'static str = "./glyphgen/glyphs/";
const glyphext: &'static str = ".png";
const imgsize: u32 = 64u32;


use image::imageops;
use image::ImageBuffer;
use image::GrayImage;
use imageproc::affine;
use image::Luma;
use rand::prelude::*;
use rayon::prelude::*;
use super::inputdata::Inputdata;

type Imgtype = ImageBuffer<image::Luma<u8>, std::vec::Vec<u8>>;

fn load_glyph(id: u8) -> Imgtype{
	println!("{}{}{}", glyphpath, id, glyphext);
    image::open(format!("{}{}{}", glyphpath, id, glyphext)).unwrap().to_luma()
}

pub fn load_glyphlib() -> Vec<Imgtype>{
    (0..32).into_par_iter().map(|id| load_glyph(id as u8)).collect::<Vec<Imgtype>>()
}

pub fn create_testdata(glyphlib: &Vec<Imgtype>)-> Inputdata{
    let mut rng = thread_rng();


    let noise_intensity_max = 0.5;
    let blur_max = 3f32;
    let brighten_min =0.8f32;
    let brighten_max = -0.8f32;
    let contrast_min = -0.5;
    let contrast_max = 0.5;
    let glyphsize_min = 17;
    let glyphsize_max = 23;
    let glyphintensity_min = 0.3;
    let glyphintensity_max = 1f32;

    let mut glyph_present = 0f64;
    if rng.gen::<f32>() > 0.5{
        glyph_present = 1f64;
    }
    let noise = rng.gen::<f32>() * noise_intensity_max;
    let blur_amount = rng.gen::<f32>() * blur_max;
    let brighten_amount =  brighten_min + ((brighten_max - brighten_min) * rng.gen::<f32>());
    let contrast_amount =contrast_min + ((contrast_max - contrast_min) * rng.gen::<f32>());
    let rotate_degs = rng.gen::<f32>() * 3.141 * 2.0;
    
    let glyphsize = (glyphsize_min as f32 + (rng.gen::<f32>() * (glyphsize_max - glyphsize_min) as f32)) as u32;
    let glyphintensity = glyphintensity_min as f32 + (rng.gen::<f32>() * (glyphintensity_max - glyphintensity_min) as f32);

    let gx = (((imgsize - glyphsize) as f32 * rng.gen::<f32>()) * 2.0 * (0.5 - rng.gen::<f32>())) as i32 / 2;
    let gy = (((imgsize - glyphsize) as f32 * rng.gen::<f32>()) * 2.0 * (0.5 - rng.gen::<f32>())) as i32 / 2;
    let glyphid = ((31.0 * rng.gen::<f32>()) as u32) as usize;
    //println!("id: {}, gx: {}, gy: {}", glyphid, gx, gy);

    let mut testdata = ImageBuffer::from_fn(imgsize,imgsize, |x,y| {
        image::Luma([(255.0 * (rng.gen::<f32>() * noise)) as u8])
    });

    let luma = |l: u8| Luma{data: [l]}; 
    let get_luma = |l: Luma<u8>| l.data[0];
    let limit = |u: u32| if u > 255 {255u8}  else {u as u8};

    let mut glyphproj = GrayImage::new(imgsize, imgsize);
    let glyph_real_size = glyphlib[glyphid].width();
    let glyph_scaling_factor = (glyph_real_size as f32) / (glyphsize as f32);
    let glyph_upper_left = (imgsize / 2) - (glyphsize / 2);
    if(glyph_present > 0.5f64){
        for y in 0..glyphsize{
            for x in 0..glyphsize{
                glyphproj[(x + glyph_upper_left,y + glyph_upper_left)] = luma((glyphintensity * (255 - (glyphlib[glyphid][((x as f32 * glyph_scaling_factor) as u32, (y as f32 * glyph_scaling_factor) as u32)].data[0])) as f32) as u8);
            }
        }
    }

    glyphproj = affine::translate(&affine::rotate(&glyphproj, (imgsize as f32 / 2.0, imgsize as f32 / 2.0), rotate_degs, affine::Interpolation::Nearest), (gx as i32, gy as i32));



    for y in 0..imgsize{
        for x in 0..imgsize{
            testdata[(x,y)] = luma(limit(255u32 - limit(get_luma(glyphproj[(x,y)])as u32  +  get_luma(testdata[(x,y)]) as u32) as u32));
        }
    }
    testdata = imageops::contrast(&imageops::brighten(&imageops::blur(&testdata, blur_amount), (brighten_amount * 255.0) as i32) , contrast_amount);
    /*
     for x in 0..imgsize{
        for y in 0..imgsize{
            testdata[(x,y)] = luma(limit(255u32 - limit(get_luma(glyphproj[(x,y)])as u32  +  get_luma(testdata[(x,y)]) as u32) as u32));
        }
    }*/

    let mut unpacked_pixels = Vec::new();
    /*for y in 0..imgsize{
        for x in 0..imgsize{
            unpacked_pixels.push(get_luma(testdata[(x,y)]) as f64);
        }
    }*/

    return Inputdata{
        image: testdata,
        unpacked_pixels,
        glyph_id: glyphid as f64,
        gx: gx as f64,
        gy: gy as f64,
        rotation: rotate_degs as f64,
        glyph_present
    }
}

pub fn create_testdata_multiple(glyphlib: &Vec<Imgtype>, amount: u32) -> Vec<Inputdata>{
    (0..amount).into_par_iter().map(|_id| create_testdata(glyphlib)).collect::<Vec<Inputdata>>()
}

pub fn create_testdata_multiple_iter<'a>(glyphlib: &'a Vec<Imgtype>, amount: u32) -> impl Iterator<Item = (u32, Inputdata)> + 'a{
    (0..amount).into_iter().map(move |id| (id.clone(), create_testdata(glyphlib).clone()))
}