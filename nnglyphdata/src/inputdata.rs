use image::ImageBuffer;
#[derive(Clone)]
pub struct Inputdata{
    pub image: ImageBuffer<image::Luma<u8>, std::vec::Vec<u8>>,
    pub unpacked_pixels: Vec<f64>,
    pub gx: f64,
    pub gy: f64,
    pub glyph_id: f64,
    pub glyph_present: f64,
    pub rotation: f64
}