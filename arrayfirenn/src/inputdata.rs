use image::ImageBuffer;
#[derive(Clone)]
pub struct Inputdata{
    pub image: ImageBuffer<image::Luma<u8>, std::vec::Vec<u8>>,
    pub unpacked_pixels: Vec<f32>,
    pub gx: f32,
    pub gy: f32,
    pub glyph_id: f32,
    pub glyph_present: f32,
    pub rotation: f32
}