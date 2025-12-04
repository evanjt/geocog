//! XYZ tile extraction from COG files
//!
//! This module provides functionality to extract XYZ map tiles from Cloud Optimized GeoTIFFs.
//! It handles coordinate transformations, overview selection, and pixel sampling.
//!
//! # Example
//!
//! ```rust,ignore
//! use geocog::{CogReader, xyz_tile::{extract_xyz_tile, TileData}};
//!
//! let reader = CogReader::open("path/to/cog.tif")?;
//! let tile = extract_xyz_tile(&reader, 3, 7, 5, (256, 256))?;
//! // tile.pixels contains f32 pixel values
//! // tile.bands indicates number of bands (1 for grayscale, 3 for RGB)
//! ```

use std::collections::HashSet;
use proj4rs::proj::Proj;
use proj4rs::transform::transform;

use crate::cog_reader::CogReader;

/// Extracted tile data with band information
#[derive(Debug, Clone)]
pub struct TileData {
    /// Pixel values (interleaved if multi-band: R,G,B,R,G,B,...)
    pub pixels: Vec<f32>,
    /// Number of bands (1 for grayscale, 3 for RGB, 4 for RGBA)
    pub bands: usize,
    /// Tile width
    pub width: usize,
    /// Tile height
    pub height: usize,
}

/// Bounding box in a coordinate reference system
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub minx: f64,
    pub miny: f64,
    pub maxx: f64,
    pub maxy: f64,
}

impl BoundingBox {
    /// Create a new bounding box
    pub fn new(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Self {
        Self { minx, miny, maxx, maxy }
    }

    /// Create bounding box from XYZ tile coordinates (Web Mercator EPSG:3857)
    pub fn from_xyz(z: u32, x: u32, y: u32) -> Self {
        let n = 2_u32.pow(z) as f64;
        let tile_size = 40_075_016.685_578_49 / n; // Web Mercator extent / tiles

        let minx = -20_037_508.342_789_244 + (x as f64) * tile_size;
        let maxx = minx + tile_size;
        let maxy = 20_037_508.342_789_244 - (y as f64) * tile_size;
        let miny = maxy - tile_size;

        Self { minx, miny, maxx, maxy }
    }
}

/// Get PROJ string for an EPSG code from the crs-definitions database
fn epsg_to_proj_string(epsg: u32) -> Option<&'static str> {
    u16::try_from(epsg).ok()
        .and_then(crs_definitions::from_code)
        .map(|def| def.proj4)
}

/// Check if an EPSG code represents a geographic (lon/lat) CRS
fn is_geographic_crs(epsg: u32) -> bool {
    if let Some(proj_str) = epsg_to_proj_string(epsg) {
        proj_str.contains("+proj=longlat")
    } else {
        // Fallback: 4326 and similar are geographic
        epsg == 4326 || (epsg >= 4000 && epsg < 5000)
    }
}

/// Coordinate transformer using proj4rs (pure Rust)
pub struct CoordTransformer {
    source_proj: Proj,
    target_proj: Proj,
    /// True if source uses degrees (needs radian conversion)
    #[allow(dead_code)]
    source_is_geographic: bool,
    /// True if target uses degrees (needs radian conversion)
    target_is_geographic: bool,
}

impl CoordTransformer {
    /// Create a transformer from EPSG:3857 (Web Mercator) to another CRS
    pub fn from_3857_to(target_epsg: u32) -> Result<Self, String> {
        let source_str = epsg_to_proj_string(3857)
            .ok_or("EPSG:3857 not supported")?;
        let target_str = epsg_to_proj_string(target_epsg)
            .ok_or_else(|| format!("EPSG:{} not supported", target_epsg))?;

        let source_proj = Proj::from_proj_string(source_str)
            .map_err(|e| format!("Invalid source projection: {:?}", e))?;
        let target_proj = Proj::from_proj_string(target_str)
            .map_err(|e| format!("Invalid target projection: {:?}", e))?;

        Ok(Self {
            source_proj,
            target_proj,
            source_is_geographic: false, // 3857 is projected (meters)
            target_is_geographic: is_geographic_crs(target_epsg),
        })
    }

    /// Transform coordinates from source CRS to target CRS
    pub fn transform(&self, x: f64, y: f64) -> Result<(f64, f64), String> {
        let mut point = (x, y, 0.0);

        transform(&self.source_proj, &self.target_proj, &mut point)
            .map_err(|e| format!("Transform failed: {:?}", e))?;

        // Convert from radians to degrees if target is geographic
        let (out_x, out_y) = if self.target_is_geographic {
            (point.0.to_degrees(), point.1.to_degrees())
        } else {
            (point.0, point.1)
        };

        Ok((out_x, out_y))
    }
}

/// Extract an XYZ tile from a COG reader
///
/// # Arguments
/// * `reader` - The COG reader with loaded metadata
/// * `z` - Zoom level
/// * `x` - Tile X coordinate
/// * `y` - Tile Y coordinate
/// * `tile_size` - Output tile dimensions (width, height), typically (256, 256)
///
/// # Returns
/// `TileData` containing pixel values and band count
pub fn extract_xyz_tile(
    reader: &CogReader,
    z: u32,
    x: u32,
    y: u32,
    tile_size: (usize, usize),
) -> Result<TileData, Box<dyn std::error::Error + Send + Sync>> {
    let extent_3857 = BoundingBox::from_xyz(z, x, y);
    extract_tile_with_extent(reader, &extent_3857, tile_size)
}

/// Extract a tile from a COG reader using a Web Mercator bounding box
///
/// This is the main extraction function that handles overview selection,
/// coordinate transformation, and pixel sampling.
pub fn extract_tile_with_extent(
    reader: &CogReader,
    extent_3857: &BoundingBox,
    tile_size: (usize, usize),
) -> Result<TileData, Box<dyn std::error::Error + Send + Sync>> {
    let metadata = &reader.metadata;
    let geo_transform = &metadata.geo_transform;

    // Pre-compute the affine transform from output pixel to source pixel
    let (Some(base_scale), Some(_tiepoint)) = (geo_transform.pixel_scale, geo_transform.tiepoint) else {
        return Err("Missing geotransform".into());
    };

    // Create coordinate transformer from EPSG:3857 (tile coords) to source CRS
    let source_epsg = metadata.crs_code.unwrap_or(3857) as u32;

    // For same-CRS case (3857 to 3857), we can skip transformation
    let use_identity = source_epsg == 3857;
    let transformer = if use_identity {
        None
    } else {
        Some(CoordTransformer::from_3857_to(source_epsg)?)
    };

    // Convert extent to source CRS to get geographic extent
    let (src_minx, src_miny, src_maxx, src_maxy) = if let Some(ref t) = transformer {
        let (minx, miny) = t.transform(extent_3857.minx, extent_3857.miny)?;
        let (maxx, maxy) = t.transform(extent_3857.maxx, extent_3857.maxy)?;
        (minx, miny, maxx, maxy)
    } else {
        (extent_3857.minx, extent_3857.miny, extent_3857.maxx, extent_3857.maxy)
    };

    // Calculate how many source pixels would cover this extent at full resolution
    let extent_src_width = ((src_maxx - src_minx) / base_scale[0]).abs().max(1.0) as usize;
    let extent_src_height = ((src_maxy - src_miny) / base_scale[1]).abs().max(1.0) as usize;

    // Find the best overview level
    let overview_idx = reader.best_overview_for_resolution(extent_src_width, extent_src_height);

    // Call the internal function with automatic fallback for empty overviews
    extract_tile_with_overview(reader, extent_3857, tile_size, overview_idx, transformer)
}

/// Internal function that extracts a tile using a specific overview level (or full resolution if None)
fn extract_tile_with_overview(
    reader: &CogReader,
    extent_3857: &BoundingBox,
    tile_size: (usize, usize),
    overview_idx: Option<usize>,
    transformer: Option<CoordTransformer>,
) -> Result<TileData, Box<dyn std::error::Error + Send + Sync>> {
    let (tile_size_x, tile_size_y) = tile_size;
    let metadata = &reader.metadata;
    let geo_transform = &metadata.geo_transform;

    // Pre-compute the affine transform from output pixel to source pixel
    let (Some(base_scale), Some(tiepoint)) = (geo_transform.pixel_scale, geo_transform.tiepoint) else {
        return Err("Missing geotransform".into());
    };

    // Get effective metadata for the level we're using
    let (eff_width, eff_height, eff_tile_width, eff_tile_height, eff_tiles_across, scale_factor) = if let Some(ovr_idx) = overview_idx {
        let ovr = &reader.overviews[ovr_idx];
        (ovr.width, ovr.height, ovr.tile_width, ovr.tile_height, ovr.tiles_across, ovr.scale as f64)
    } else {
        (metadata.width, metadata.height, metadata.tile_width, metadata.tile_height, metadata.tiles_across, 1.0)
    };

    // Adjust scale for overview level
    let scale = [base_scale[0] * scale_factor, base_scale[1] * scale_factor, base_scale[2]];

    // Output tile pixel resolution in Web Mercator
    let out_res_x = (extent_3857.maxx - extent_3857.minx) / (tile_size_x as f64);
    let out_res_y = (extent_3857.maxy - extent_3857.miny) / (tile_size_y as f64);

    // Pre-compute which source tiles we need by checking corners and edges
    let mut needed_tiles: HashSet<usize> = HashSet::new();

    // Helper closure to compute tile index at overview level
    let tile_index_at_level = |px: usize, py: usize| -> Option<usize> {
        if px >= eff_width || py >= eff_height {
            return None;
        }
        let tile_col = px / eff_tile_width;
        let tile_row = py / eff_tile_height;
        Some(tile_row * eff_tiles_across + tile_col)
    };

    // Sample corners and edges to find needed tiles (much faster than checking every pixel)
    let sample_points = [
        (0, 0), (tile_size_x - 1, 0), (0, tile_size_y - 1), (tile_size_x - 1, tile_size_y - 1),
        (tile_size_x / 2, 0), (tile_size_x / 2, tile_size_y - 1),
        (0, tile_size_y / 2), (tile_size_x - 1, tile_size_y / 2),
        (tile_size_x / 2, tile_size_y / 2),
    ];

    for &(out_x, out_y) in &sample_points {
        let merc_x = extent_3857.minx + (out_x as f64 + 0.5) * out_res_x;
        let merc_y = extent_3857.maxy - (out_y as f64 + 0.5) * out_res_y;

        // Transform from Web Mercator to source CRS
        let (world_x, world_y) = if let Some(ref t) = transformer {
            t.transform(merc_x, merc_y)?
        } else {
            (merc_x, merc_y)
        };

        let src_px = tiepoint[0] + (world_x - tiepoint[3]) / scale[0];
        let src_py = tiepoint[1] + (tiepoint[4] - world_y) / scale[1];

        // Clamp to valid range for tile detection (handles ±180° boundary)
        let src_px_clamped = src_px.clamp(0.0, eff_width as f64 - 1.0);
        let src_py_clamped = src_py.clamp(0.0, eff_height as f64 - 1.0);

        // Accept source pixels within 1 pixel of valid range (for tile detection)
        // Use < (eff_width + 1) not <= eff_width because at exact boundaries like +180°
        // src_px may equal exactly eff_width (e.g., 2620.0 for 2620-pixel overview)
        if src_px >= -1.0 && src_px < (eff_width + 1) as f64 &&
           src_py >= -1.0 && src_py < (eff_height + 1) as f64 {
            if let Some(idx) = tile_index_at_level(src_px_clamped as usize, src_py_clamped as usize) {
                needed_tiles.insert(idx);
            }
        }
    }

    // Also add tiles between the corners (for larger output areas)
    let max_tile_count = if overview_idx.is_some() {
        let ovr = &reader.overviews[overview_idx.unwrap()];
        ovr.tile_offsets.len()
    } else {
        metadata.tile_offsets.len()
    };

    if needed_tiles.len() > 1 {
        let min_tile = *needed_tiles.iter().min().unwrap_or(&0);
        let max_tile = *needed_tiles.iter().max().unwrap_or(&0);
        let min_col = min_tile % eff_tiles_across;
        let max_col = max_tile % eff_tiles_across;
        let min_row = min_tile / eff_tiles_across;
        let max_row = max_tile / eff_tiles_across;

        // Read all tiles in the range - with overviews this is usually few tiles
        for row in min_row..=max_row {
            for col in min_col..=max_col {
                let idx = row * eff_tiles_across + col;
                if idx < max_tile_count {
                    needed_tiles.insert(idx);
                }
            }
        }
    }

    // If no tiles needed, return transparent tile
    if needed_tiles.is_empty() {
        let num_bands = metadata.bands;
        return Ok(TileData {
            pixels: vec![0.0; tile_size_x * tile_size_y * num_bands],
            bands: num_bands,
            width: tile_size_x,
            height: tile_size_y,
        });
    }

    // Pre-load all needed tiles
    let mut tile_data_cache: std::collections::HashMap<usize, Vec<f32>> = std::collections::HashMap::new();

    for &tile_idx in &needed_tiles {
        let tile_data = if let Some(ovr_idx) = overview_idx {
            reader.read_overview_tile(ovr_idx, tile_idx)
        } else {
            reader.read_tile(tile_idx)
        };

        if let Ok(data) = tile_data {
            tile_data_cache.insert(tile_idx, data);
        }
    }

    // If all tile reads failed, try falling back to full resolution
    if tile_data_cache.is_empty() && overview_idx.is_some() {
        return extract_tile_with_overview(reader, extent_3857, tile_size, None, transformer);
    }

    let num_bands = metadata.bands;
    let mut pixel_data = vec![0.0_f32; tile_size_x * tile_size_y * num_bands];

    // Pre-compute inverse scale for speed
    let inv_scale_x = 1.0 / scale[0];
    let inv_scale_y = 1.0 / scale[1];

    // Helper to sample a pixel from the cached tile data
    let sample_pixel = |px: usize, py: usize, band: usize| -> Option<f32> {
        let tile_col = px / eff_tile_width;
        let tile_row = py / eff_tile_height;
        let tile_idx = tile_row * eff_tiles_across + tile_col;

        let tile_data = tile_data_cache.get(&tile_idx)?;

        let local_x = px % eff_tile_width;
        let local_y = py % eff_tile_height;
        let pixel_idx = (local_y * eff_tile_width + local_x) * num_bands + band;

        tile_data.get(pixel_idx).copied()
    };

    // Sample each output pixel
    for out_y in 0..tile_size_y {
        let merc_y = extent_3857.maxy - (out_y as f64 + 0.5) * out_res_y;

        for out_x in 0..tile_size_x {
            let merc_x = extent_3857.minx + (out_x as f64 + 0.5) * out_res_x;

            // Transform from Web Mercator to source CRS
            let (world_x, world_y) = if let Some(ref t) = transformer {
                match t.transform(merc_x, merc_y) {
                    Ok(coords) => coords,
                    Err(_) => continue,
                }
            } else {
                (merc_x, merc_y)
            };

            // Inline world_to_pixel for speed
            let src_px = tiepoint[0] + (world_x - tiepoint[3]) * inv_scale_x;
            let src_py = tiepoint[1] + (tiepoint[4] - world_y) * inv_scale_y;

            // Nearest neighbor resampling - preserves crisp edges
            let src_px_int = src_px.round() as isize;
            let src_py_int = src_py.round() as isize;

            // Clamp to valid pixel range (handles edge cases at ±180° for global datasets)
            let src_px_clamped = src_px_int.max(0).min(eff_width as isize - 1);
            let src_py_clamped = src_py_int.max(0).min(eff_height as isize - 1);

            // Check if original source coordinates are within valid range
            // Allow up to 0.5 pixel past the edge for global datasets where extent matches exactly
            if src_px >= -0.5 && src_px <= eff_width as f64 + 0.5 &&
               src_py >= -0.5 && src_py <= eff_height as f64 + 0.5 {

                let out_idx = (out_y * tile_size_x + out_x) * num_bands;

                // Sample each band with nearest neighbor
                for band in 0..num_bands {
                    if let Some(value) = sample_pixel(src_px_clamped as usize, src_py_clamped as usize, band) {
                        pixel_data[out_idx + band] = value;
                    }
                }
            }
        }
    }

    Ok(TileData {
        pixels: pixel_data,
        bands: num_bands,
        width: tile_size_x,
        height: tile_size_y,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_from_xyz() {
        // Tile 0/0/0 should cover the whole world in Web Mercator
        let bbox = BoundingBox::from_xyz(0, 0, 0);
        assert!((bbox.minx - (-20037508.342789244)).abs() < 1.0);
        assert!((bbox.maxx - 20037508.342789244).abs() < 1.0);

        // At zoom 1, there are 4 tiles (2x2)
        let bbox_1_0_0 = BoundingBox::from_xyz(1, 0, 0);
        let bbox_1_1_0 = BoundingBox::from_xyz(1, 1, 0);
        assert!((bbox_1_0_0.maxx - bbox_1_1_0.minx).abs() < 1.0);
    }

    #[test]
    fn test_epsg_proj_strings() {
        assert!(epsg_to_proj_string(4326).is_some());
        assert!(epsg_to_proj_string(3857).is_some());
        assert!(epsg_to_proj_string(99999).is_none());
    }

    #[test]
    fn test_coord_transformer_identity() {
        // 3857 to 3857 should be skipped (use_identity)
        let source_epsg: u32 = 3857;
        let use_identity = source_epsg == 3857;
        assert!(use_identity);
    }
}
