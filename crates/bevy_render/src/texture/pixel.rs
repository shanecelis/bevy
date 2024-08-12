use core::ops::{Range, Bound, RangeBounds};
use crate::{
    render_asset::RenderAssetUsages,
    render_resource::{TextureFormat, Extent3d, TextureDimension},
    texture::image::{Image, TextureFormatPixelInfo}
};
use bevy_math::UVec2;
use bevy_color::{
    Color,
    ColorToPacked,
    ColorToComponents,
    LinearRgba,
    Srgba,
};
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum PixelError {
    #[error("pixel operations not supported for compressed images")]
    CompressionNotSupported,
    #[error("pixel operations not supported for depth images")]
    DepthNotSupported,
    #[error("pixel operations not supported for stencil images")]
    StencilNotSupported,
    #[error("pixel operation needed more data")]
    NotEnoughData,
    #[error("no such pixel location")]
    InvalidLocation,
    #[error("could not align pixel data")]
    AlignmentFailed,
    #[error("invalid range")]
    InvalidRange,
}

fn align_to<T, U>(slice: &[T]) -> Result<&[U], PixelError> {
    let (prefix, aligned, suffix) = unsafe { slice.align_to::<U>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        Err(PixelError::AlignmentFailed)
    } else {
        Ok(aligned)
    }
}

fn align_to_mut<T, U>(slice: &mut [T]) -> Result<&mut [U], PixelError> {
    let (prefix, aligned, suffix) = unsafe { slice.align_to_mut::<U>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        Err(PixelError::AlignmentFailed)
    } else {
        Ok(aligned)
    }
}

// #[derive(Debug)]
// pub struct PixelIterMut<'a> {
//     image: &'a mut Image,
//     index: usize,
//     end: usize,
//     color: Option<Color>,
// }

// impl<'a> Iterator for PixelIterMut<'a> {
//     type Item = &mut Color;
//     fn next(&mut self) -> Option<Self::Item> {
//         use TextureFormat::*;
//         if self.index >= self.end {
//             None
//         } else {
//             let format = self.image.texture_descriptor.format;
//             let components = format.components() as usize;
//             let prev = self.index.saturating_sub(1) * components;
//             let start = self.index * components;
//             self.index += 1;
//             match format {
//                 Rgba32Float => {
//                     let floats = align_to_mut::<u8, f32>(&mut self.image.data).ok()?;
//                     let mut a = [0.0f32; 4];
//                     if let Some(color) = self.color.take() {
//                         let c: LinearRgba = color.into();
//                         let a = c.to_f32_array();
//                         floats[prev..prev + components].copy_from_slice(&a);
//                     }
//                     a.copy_from_slice(&floats[start..start + components]);
//                     self.color = Some(LinearRgba::from_f32_array(a).into());
//                     self.color.as_mut()
//                 }
//                 Rgba8Unorm => {
//                     let mut a = [0u8; 4];
//                     a.copy_from_slice(&self.image.data[start..start + components]);
//                     self.color = Some(LinearRgba::from_u8_array(a).into());
//                     self.color.as_mut()
//                 },
//                 Rgba8UnormSrgb => {
//                     let mut a = [0u8; 4];
//                     a.copy_from_slice(&self.image.data[start..start + components]);
//                     self.color = Some(Srgba::from_u8_array(a).into());
//                     self.color.as_mut()
//                 },
//                 _ => {
//                     None
//                 }
//             }
//         }
//     }
// }

#[derive(Debug)]
pub struct PixelIter<'a> {
    image: &'a Image,
    index: usize,
    end: usize,
}

impl<'a> Iterator for PixelIter<'a> {
    type Item = Color;
    fn next(&mut self) -> Option<Self::Item> {
        use TextureFormat::*;
        if self.index >= self.end {
            None
        } else {
            let format = self.image.texture_descriptor.format;
            let components = format.components() as usize;
            let start = self.index * components;
            self.index += 1;
            match format {
                Rgba32Float => {
                    let floats = align_to::<u8, f32>(&self.image.data).ok()?;
                    let mut a = [0.0f32; 4];
                    a.copy_from_slice(&floats[start..start + components]);
                    Some(LinearRgba::from_f32_array(a).into())
                }
                Rgba8Unorm => {
                    let mut a = [0u8; 4];
                    a.copy_from_slice(&self.image.data[start..start + components]);
                    Some(LinearRgba::from_u8_array(a).into())
                },
                Rgba8UnormSrgb => {
                    let mut a = [0u8; 4];
                    a.copy_from_slice(&self.image.data[start..start + components]);
                    Some(Srgba::from_u8_array(a).into())
                },
                _ => {
                    None
                }
            }
        }
    }
}

impl Image {

    pub fn pixels<R: RangeBounds<usize>>(&self, range: R) -> Result<PixelIter, PixelError> {
        use TextureFormat::*;
        let format = self.texture_descriptor.format;
        let components = format.components() as usize;
        match format {
            Rgba32Float => {
                let f = align_to::<u8, f32>(&self.data)?;
                Ok(f.len() / components)
            }
            Rgba8Unorm => {
                Ok(self.data.len() / components)
            },
            Rgba8UnormSrgb => {
                Ok(self.data.len() / components)
            },
            f => {
                if f.is_compressed() {
                    Err(PixelError::CompressionNotSupported)
                } else if f.has_depth_aspect() {
                    Err(PixelError::DepthNotSupported)
                } else if f.has_stencil_aspect() {
                    Err(PixelError::StencilNotSupported)
                } else {
                    todo!("Fix {f:?}");
                }
            }
        }.and_then(|max_length| {
            let index = match range.start_bound() {
                Bound::Unbounded => 0,
                Bound::Included(i) => *i,
                Bound::Excluded(j) => j + 1
            };

            let end = match range.end_bound() {
                Bound::Unbounded => max_length,
                Bound::Included(i) => i + 1,
                Bound::Excluded(j) => *j
            };
            if index >= max_length || end > max_length {
                Err(PixelError::InvalidRange)
            } else {
                Ok(PixelIter { image: self, index, end })
            }
        })
    }

    pub fn set_pixels(&mut self, mut start: usize, source: &[Color]) -> Result<(), PixelError>{
        use TextureFormat::*;
        let format = self.texture_descriptor.format;
        let components = format.components() as usize;
        start *= components;
        match format {
            Rgba32Float => {
                let floats = align_to_mut::<u8, f32>(&mut self.data)?;
                for color in source {
                    let c: LinearRgba = (*color).into();
                    let a = c.to_f32_array();
                    floats[start..start + components].copy_from_slice(&a);
                    start += components;
                }
                Ok(())
            }
            Rgba8Unorm => {
                for color in source {
                    let c: LinearRgba = (*color).into();
                    let a = c.to_u8_array();
                    self.data[start..start + components].copy_from_slice(&a);
                    start += components;
                }
                Ok(())
            },
            Rgba8UnormSrgb => {
                for color in source {
                    let c: Srgba = (*color).into();
                    let a = c.to_u8_array();
                    self.data[start..start + components].copy_from_slice(&a);
                    start += components;
                }
                Ok(())
            },
            f => {
                if f.is_compressed() {
                    Err(PixelError::CompressionNotSupported)
                } else if f.has_depth_aspect() {
                    Err(PixelError::DepthNotSupported)
                } else if f.has_stencil_aspect() {
                    Err(PixelError::StencilNotSupported)
                } else {
                    todo!("Fix {f:?}");
                }
            }
        }
    }

    pub fn get_pixel(&self, location: UVec2) -> Result<Color, PixelError> {
        use TextureFormat::*;
        // TextureFormatPixelInfo
        // texture::DataFormat
        let image_size: Extent3d = self.texture_descriptor.size;
        if location.x >= image_size.width || location.y >= image_size.height {
            return Err(PixelError::InvalidLocation);
        }
        let format = self.texture_descriptor.format;
        let components = format.components() as usize;
        let pixel_size = format.pixel_size() as usize;
        let start = (location.x + location.y * image_size.width) as usize * components;
        match format {
            Rgba32Float => {
                let floats = align_to::<u8, f32>(&self.data)?;
                let mut a = [0.0f32; 4];
                a.copy_from_slice(&floats[start..start + components]);
                Ok(LinearRgba::from_f32_array(a).into())
            }
            Rgba8Unorm => {
                let mut a = [0u8; 4];
                a.copy_from_slice(&self.data[start..start + pixel_size]);
                Ok(LinearRgba::from_u8_array(a).into())
            },
            Rgba8UnormSrgb => {
                let mut a = [0u8; 4];
                a.copy_from_slice(&self.data[start..start + pixel_size]);
                Ok(Srgba::from_u8_array(a).into())
            },
            f => {
                if f.is_compressed() {
                    Err(PixelError::CompressionNotSupported)
                } else if f.has_depth_aspect() {
                    Err(PixelError::DepthNotSupported)
                } else if f.has_stencil_aspect() {
                    Err(PixelError::StencilNotSupported)
                } else {
                    todo!("Fix {f:?}");
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rgba8unorm_size() {
        let format = TextureFormat::Rgba8Unorm;
        assert_eq!(format.pixel_size(), 4);
        assert_eq!(format.components(), 4);
    }

    #[test]
    fn test_get_pixel() {
        let image = Image::default();
        assert_eq!(image.texture_descriptor.size.width, 1);
        assert_eq!(image.texture_descriptor.size.height, 1);
        // assert_eq!(image.get_pixel(0, 0).unwrap(), Srgba::from(Color::WHITE));
        assert_eq!(image.get_pixel(UVec2::new(0, 0)).unwrap(), Srgba::WHITE.into());
        assert_eq!(image.get_pixel(UVec2::new(1, 0)).unwrap_err(), PixelError::InvalidLocation);
        assert_eq!(image.get_pixel(UVec2::new(0, 1)).unwrap_err(), PixelError::InvalidLocation);
    }

    #[test]
    fn test_align_to_from_f32() {
        let pixel = [0.0, 0.0, 0.0, 1.0];
        assert!(&align_to::<f32,u8>(&pixel).is_ok());

        // We can always go to u8 from f32.
        let pixel = [0.0, 0.0, 0.0, 1.0, 0.0];
        assert!(&align_to::<f32,u8>(&pixel).is_ok());

        // We can always go to u8 from f32.
        let pixel = [0.0, 0.0, 0.0];
        assert!(&align_to::<f32,u8>(&pixel).is_ok());
    }

    #[test]
    fn test_align_to_f32_from_u8() {
        let pixel = [0u8; 4];

        // let (prefix, aligned, suffix) = unsafe { pixel.align_to::<f32>() };
        // assert!(prefix.is_empty());
        // assert!(suffix.is_empty());
        // assert_eq!(aligned.len(), 1);

        assert!(align_to::<u8,f32>(&pixel).is_ok());

        let pixel = [0u8; 17];
        assert!(align_to::<u8,f32>(&pixel).is_err());

        let pixel = [0u8; 15];
        assert!(align_to::<u8,f32>(&pixel).is_err());
    }

    fn image_from<T>(width: u32, height: u32, format: TextureFormat, data: &[T]) -> Result<Image, PixelError> {
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        Ok(Image::new_fill(
            size,
            TextureDimension::D2,
            align_to::<T,u8>(data)?,
            format,
            RenderAssetUsages::MAIN_WORLD,
        ))
    }

    #[test]
    fn test_get_pixel_f32() {
        let pixel = [0.0, 0.0, 0.0, 1.0];
        // FIXME: Spooky. If the next line is removed, the following image_from() will fail.
        // Must have to do with alignment.
        assert_eq!(align_to::<f32,u8>(&pixel).unwrap().len(), 16);
        let image = image_from(1, 1, TextureFormat::Rgba32Float, &pixel).unwrap();
        assert_eq!(image.get_pixel(UVec2::new(0, 0)).unwrap(), LinearRgba::BLACK.into());
    }

    #[test]
    fn test_pixels() {
        let pixel = [0.0, 0.0, 0.0, 1.0];
        // FIXME: Spooky. If the next line is removed, the following image_from() will fail.
        // Must have to do with alignment.
        assert_eq!(align_to::<f32,u8>(&pixel).unwrap().len(), 16);
        let image = image_from(1, 1, TextureFormat::Rgba32Float, &pixel).unwrap();
        let mut pixels = image.pixels(..).unwrap();
        assert_eq!(pixels.next().unwrap(), LinearRgba::BLACK.into());
        assert_eq!(pixels.next(), None);

        assert_eq!(image.pixels(1..).unwrap_err(), PixelError::InvalidRange);
        assert_eq!(image.pixels(..=1).unwrap_err(), PixelError::InvalidRange);
        assert!(image.pixels(0..0).unwrap().next().is_none());
        assert!(image.pixels(0..1).unwrap().next().is_some());
    }
}
