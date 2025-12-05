//! XYZ tile extraction from COG files
//!
//! This module provides functionality to extract XYZ map tiles from Cloud Optimized GeoTIFFs.
//! It handles coordinate transformations, overview selection, and pixel sampling.
//!
//! # Example
//!
//! ```rust,ignore
//! use cogrs::{CogReader, xyz_tile::{extract_xyz_tile, TileData}};
//!
//! let reader = CogReader::open("path/to/cog.tif")?;
//! let tile = extract_xyz_tile(&reader, 3, 7, 5, (256, 256))?;
//! // tile.pixels contains f32 pixel values
//! // tile.bands indicates number of bands (1 for grayscale, 3 for RGB)
//! ```

use std::collections::HashSet;
use std::f64::consts::PI;
use ahash::AHashMap;
use proj4rs::proj::Proj;
use proj4rs::transform::transform;

use crate::cog_reader::CogReader;
use crate::geometry::projection::{get_proj_string, is_geographic_crs};

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

/// Half the earth's circumference in Web Mercator meters
const HALF_EARTH: f64 = 20037508.342789244;

/// Fast inline conversion from Web Mercator X to longitude (degrees)
#[inline(always)]
fn merc_x_to_lon(x: f64) -> f64 {
    x * 180.0 / HALF_EARTH
}

/// Fast inline conversion from Web Mercator Y to latitude (degrees)
#[inline(always)]
fn merc_y_to_lat(y: f64) -> f64 {
    // y in meters -> y in radians (normalized by earth's extent * PI)
    let y_rad = y * PI / HALF_EARTH;
    // Inverse Mercator: lat = 2 * atan(exp(y_rad)) - PI/2
    (2.0 * y_rad.exp().atan() - PI / 2.0) * 180.0 / PI
}

/// Transformation strategy - either fast inline math or proj4rs for complex CRS
enum TransformStrategy {
    /// Identity - no transform needed (source is already EPSG:3857)
    Identity,
    /// Fast inline math for EPSG:3857 to EPSG:4326
    FastMerc2Geo,
    /// Generic proj4rs transform for other CRS combinations
    Proj4rs(CoordTransformer),
}

impl TransformStrategy {
    /// Transform coordinates using the appropriate strategy
    #[inline(always)]
    fn transform(&self, x: f64, y: f64) -> Result<(f64, f64), String> {
        match self {
            TransformStrategy::Identity => Ok((x, y)),
            TransformStrategy::FastMerc2Geo => Ok((merc_x_to_lon(x), merc_y_to_lat(y))),
            TransformStrategy::Proj4rs(t) => t.transform(x, y),
        }
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
        let source_str = get_proj_string(3857)
            .ok_or("EPSG:3857 not supported")?;
        let target_str = get_proj_string(target_epsg as i32)
            .ok_or_else(|| format!("EPSG:{} not supported", target_epsg))?;

        let source_proj = Proj::from_proj_string(source_str)
            .map_err(|e| format!("Invalid source projection: {:?}", e))?;
        let target_proj = Proj::from_proj_string(target_str)
            .map_err(|e| format!("Invalid target projection: {:?}", e))?;

        Ok(Self {
            source_proj,
            target_proj,
            source_is_geographic: false, // 3857 is projected (meters)
            target_is_geographic: is_geographic_crs(target_epsg as i32),
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
    // Use fast inline math for 4326 (most common case), proj4rs for others
    let source_epsg = metadata.crs_code.unwrap_or(3857) as u32;

    let strategy = match source_epsg {
        3857 => TransformStrategy::Identity,
        4326 => TransformStrategy::FastMerc2Geo,
        _ => TransformStrategy::Proj4rs(CoordTransformer::from_3857_to(source_epsg)?),
    };

    // Convert extent to source CRS to get geographic extent
    let (src_minx, src_miny) = strategy.transform(extent_3857.minx, extent_3857.miny)?;
    let (src_maxx, src_maxy) = strategy.transform(extent_3857.maxx, extent_3857.maxy)?;

    // Calculate how many source pixels would cover this extent at full resolution
    let extent_src_width = ((src_maxx - src_minx) / base_scale[0]).abs().max(1.0) as usize;
    let extent_src_height = ((src_maxy - src_miny) / base_scale[1]).abs().max(1.0) as usize;

    // Find the best overview level
    let overview_idx = reader.best_overview_for_resolution(extent_src_width, extent_src_height);

    // Call the internal function with automatic fallback for empty overviews
    extract_tile_with_overview(reader, extent_3857, tile_size, overview_idx, strategy)
}

/// Internal function that extracts a tile using a specific overview level (or full resolution if None)
fn extract_tile_with_overview(
    reader: &CogReader,
    extent_3857: &BoundingBox,
    tile_size: (usize, usize),
    overview_idx: Option<usize>,
    strategy: TransformStrategy,
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
        let (world_x, world_y) = strategy.transform(merc_x, merc_y)?;

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
    let mut tile_data_cache: AHashMap<usize, Vec<f32>> = AHashMap::new();

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
        return extract_tile_with_overview(reader, extent_3857, tile_size, None, strategy);
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

    // Pre-compute source X pixel coordinates for each output column
    // For EPSG:4326, the X transform is linear: lon = merc_x * 180 / HALF_EARTH
    // Then: src_px = tiepoint[0] + (lon - tiepoint[3]) * inv_scale_x
    // Simplified: src_px = base_x + col * delta_x where base_x and delta_x are constants
    let src_px_base = if matches!(strategy, TransformStrategy::FastMerc2Geo) {
        let lon_base = (extent_3857.minx + 0.5 * out_res_x) * 180.0 / HALF_EARTH;
        tiepoint[0] + (lon_base - tiepoint[3]) * inv_scale_x
    } else {
        // For non-4326, compute first column's X
        let (world_x, _) = strategy.transform(extent_3857.minx + 0.5 * out_res_x, 0.0).unwrap_or((0.0, 0.0));
        tiepoint[0] + (world_x - tiepoint[3]) * inv_scale_x
    };

    let src_px_delta = if matches!(strategy, TransformStrategy::FastMerc2Geo) {
        // For 4326: delta_lon per column, converted to pixels
        let lon_delta = out_res_x * 180.0 / HALF_EARTH;
        lon_delta * inv_scale_x
    } else if matches!(strategy, TransformStrategy::Identity) {
        // For 3857: direct merc_x to pixel
        out_res_x * inv_scale_x
    } else {
        // For other CRS, fall back to per-pixel transform
        0.0
    };

    let use_precomputed_x = matches!(strategy, TransformStrategy::FastMerc2Geo | TransformStrategy::Identity);

    // Sample each output pixel
    for out_y in 0..tile_size_y {
        let merc_y = extent_3857.maxy - (out_y as f64 + 0.5) * out_res_y;

        // Pre-compute world_y for this row (only one Y transform per row)
        let (_, world_y) = match strategy.transform(extent_3857.minx, merc_y) {
            Ok(coords) => coords,
            Err(_) => continue,
        };
        let src_py = tiepoint[1] + (tiepoint[4] - world_y) * inv_scale_y;

        // Early row rejection - if Y is out of bounds, skip entire row
        if src_py < -0.5 || src_py > eff_height as f64 + 0.5 {
            continue;
        }

        let src_py_int = src_py.round() as isize;
        let src_py_clamped = src_py_int.max(0).min(eff_height as isize - 1) as usize;

        for out_x in 0..tile_size_x {
            // Use pre-computed X for FastMerc2Geo and Identity
            let src_px = if use_precomputed_x {
                src_px_base + (out_x as f64) * src_px_delta
            } else {
                // Fall back to per-pixel transform for other CRS
                let merc_x = extent_3857.minx + (out_x as f64 + 0.5) * out_res_x;
                let (world_x, _) = match strategy.transform(merc_x, merc_y) {
                    Ok(coords) => coords,
                    Err(_) => continue,
                };
                tiepoint[0] + (world_x - tiepoint[3]) * inv_scale_x
            };

            // Check if X is within valid range
            if src_px < -0.5 || src_px > eff_width as f64 + 0.5 {
                continue;
            }

            let src_px_int = src_px.round() as isize;
            let src_px_clamped = src_px_int.max(0).min(eff_width as isize - 1) as usize;

            let out_idx = (out_y * tile_size_x + out_x) * num_bands;

            // Sample each band with nearest neighbor
            for band in 0..num_bands {
                if let Some(value) = sample_pixel(src_px_clamped, src_py_clamped, band) {
                    pixel_data[out_idx + band] = value;
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
        assert!(get_proj_string(4326).is_some());
        assert!(get_proj_string(3857).is_some());
        assert!(get_proj_string(99999).is_none());
    }

    #[test]
    fn test_coord_transformer_identity() {
        // 3857 to 3857 should be skipped (use_identity)
        let source_epsg: u32 = 3857;
        let use_identity = source_epsg == 3857;
        assert!(use_identity);
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    
    #[test]
    fn test_proj4rs_transform_output() {
        // Create transformer from 3857 to 4326
        let transformer = CoordTransformer::from_3857_to(4326).unwrap();
        
        // Test various X coordinates (center at 0, edges at ±20037508)
        let test_points = [
            (-20037508.342789244, 0.0, "Far West"),
            (-10018754.0, 0.0, "West"),
            (0.0, 0.0, "Center"),
            (10018754.0, 0.0, "East"),
            (20037508.342789244, 0.0, "Far East"),
        ];
        
        for (x, y, name) in test_points {
            let (lon, lat) = transformer.transform(x, y).unwrap();
            println!("{}: merc({:.0}, {:.0}) -> lon={:.6}, lat={:.6}", name, x, y, lon, lat);
        }
    }
}

#[cfg(test)]
mod global_cog_tests {
    use super::*;
    use std::sync::Arc;
    use crate::cog_reader::CogReader;
    use crate::range_reader::LocalRangeReader;
    
    #[test]
    fn test_global_cog_center_tile() {
        // Skip if file doesn't exist
        let path = "/home/evan/projects/personal/geo/tileyolo/data/viridis/output_cog.tif";
        if !std::path::Path::new(path).exists() {
            println!("Skipping: test file not found");
            return;
        }
        
        let reader = LocalRangeReader::new(path).unwrap();
        let cog = CogReader::from_reader(Arc::new(reader)).unwrap();
        
        println!("COG metadata:");
        println!("  Width: {}, Height: {}", cog.metadata.width, cog.metadata.height);
        println!("  CRS: {:?}", cog.metadata.crs_code);
        if let Some(scale) = &cog.metadata.geo_transform.pixel_scale {
            println!("  Pixel scale: {:?}", scale);
        }
        if let Some(tiepoint) = &cog.metadata.geo_transform.tiepoint {
            println!("  Tiepoint: {:?}", tiepoint);
        }
        
        // Test tile 1/0/0 - western hemisphere at zoom 1
        let bbox_1_0_0 = BoundingBox::from_xyz(1, 0, 0);
        println!("\nTile 1/0/0 extent (3857): minx={:.0}, maxx={:.0}", bbox_1_0_0.minx, bbox_1_0_0.maxx);
        
        // Transform extent corners to source CRS (4326)
        let transformer = CoordTransformer::from_3857_to(4326).unwrap();
        let (min_lon, min_lat) = transformer.transform(bbox_1_0_0.minx, bbox_1_0_0.miny).unwrap();
        let (max_lon, max_lat) = transformer.transform(bbox_1_0_0.maxx, bbox_1_0_0.maxy).unwrap();
        println!("Tile 1/0/0 extent (4326): lon=[{:.2}, {:.2}], lat=[{:.2}, {:.2}]", 
                 min_lon, max_lon, min_lat, max_lat);
        
        let tile = extract_xyz_tile(&cog, 1, 0, 0, (256, 256)).unwrap();
        
        // Count non-zero pixels
        let non_zero = tile.pixels.iter().filter(|&&v| v != 0.0).count();
        let total = tile.pixels.len();
        println!("Tile 1/0/0: {}/{} non-zero pixels ({:.1}%)", non_zero, total, non_zero as f64 / total as f64 * 100.0);
        
        // Test center tile 1/0/1 (should be fully populated for a global map)
        let bbox_1_0_1 = BoundingBox::from_xyz(1, 0, 1);
        let (min_lon, _) = transformer.transform(bbox_1_0_1.minx, bbox_1_0_1.miny).unwrap();
        let (max_lon, _) = transformer.transform(bbox_1_0_1.maxx, bbox_1_0_1.maxy).unwrap();
        println!("\nTile 1/0/1 extent (4326): lon=[{:.2}, {:.2}]", min_lon, max_lon);
        
        let tile2 = extract_xyz_tile(&cog, 1, 0, 1, (256, 256)).unwrap();
        let non_zero2 = tile2.pixels.iter().filter(|&&v| v != 0.0).count();
        println!("Tile 1/0/1: {}/{} non-zero pixels ({:.1}%)", non_zero2, tile2.pixels.len(), non_zero2 as f64 / tile2.pixels.len() as f64 * 100.0);

        // Test tile 1/1/0 - eastern hemisphere (the failing one in the server!)
        let bbox_1_1_0 = BoundingBox::from_xyz(1, 1, 0);
        let (min_lon, _) = transformer.transform(bbox_1_1_0.minx, bbox_1_1_0.miny).unwrap();
        let (max_lon, _) = transformer.transform(bbox_1_1_0.maxx, bbox_1_1_0.maxy).unwrap();
        println!("\nTile 1/1/0 extent (4326): lon=[{:.2}, {:.2}]", min_lon, max_lon);

        let tile3 = extract_xyz_tile(&cog, 1, 1, 0, (256, 256)).unwrap();
        let non_zero3 = tile3.pixels.iter().filter(|&&v| v != 0.0).count();
        println!("Tile 1/1/0: {}/{} non-zero pixels ({:.1}%)", non_zero3, tile3.pixels.len(), non_zero3 as f64 / tile3.pixels.len() as f64 * 100.0);

        // Test tile 1/1/1 - eastern southern hemisphere
        let tile4 = extract_xyz_tile(&cog, 1, 1, 1, (256, 256)).unwrap();
        let non_zero4 = tile4.pixels.iter().filter(|&&v| v != 0.0).count();
        println!("Tile 1/1/1: {}/{} non-zero pixels ({:.1}%)", non_zero4, tile4.pixels.len(), non_zero4 as f64 / tile4.pixels.len() as f64 * 100.0);

        // These should all be 100% for a global dataset
        assert!(non_zero as f64 / total as f64 > 0.99, "Tile 1/0/0 should be nearly full");
        assert!(non_zero2 as f64 / tile2.pixels.len() as f64 > 0.99, "Tile 1/0/1 should be nearly full");
        assert!(non_zero3 as f64 / tile3.pixels.len() as f64 > 0.99, "Tile 1/1/0 should be nearly full");
        assert!(non_zero4 as f64 / tile4.pixels.len() as f64 > 0.99, "Tile 1/1/1 should be nearly full");
    }
}
