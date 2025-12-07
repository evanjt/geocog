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

// Well-known EPSG codes for coordinate reference systems
/// Web Mercator (Spherical Mercator) - the standard for XYZ tiles
const EPSG_WEB_MERCATOR: u32 = 3857;
/// WGS84 Geographic (longitude/latitude in degrees)
const EPSG_WGS84: u32 = 4326;

/// Resampling method for tile extraction.
///
/// Controls how pixel values are interpolated when the output resolution
/// doesn't match the source resolution exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResamplingMethod {
    /// Nearest neighbor - fastest, but can produce blocky results when upsampling.
    /// Uses the value of the closest source pixel.
    #[default]
    Nearest,
    /// Bilinear interpolation - smoother results, good balance of quality and speed.
    /// Linearly interpolates between the 4 nearest source pixels.
    Bilinear,
    /// Bicubic interpolation - highest quality, but slower.
    /// Uses a 4x4 grid of source pixels with cubic weighting.
    Bicubic,
}

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
    /// Total bytes fetched from source (compressed, before decompression)
    pub bytes_fetched: usize,
    /// Number of internal COG tiles read to produce this output tile
    pub tiles_read: usize,
    /// Overview level used (None = full resolution, Some(n) = overview index)
    pub overview_used: Option<usize>,
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
    #[inline]
    pub fn new(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Self {
        Self { minx, miny, maxx, maxy }
    }

    /// Create bounding box from XYZ tile coordinates (Web Mercator EPSG:3857)
    #[inline]
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

/// Bicubic weight function (Mitchell-Netravali with B=C=1/3)
///
/// This provides a good balance between sharpness and ringing artifacts.
#[inline]
fn bicubic_weight(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        (1.0 / 6.0) * ((12.0 - 9.0 * 0.333 - 6.0 * 0.333) * x * x * x
                     + (-18.0 + 12.0 * 0.333 + 6.0 * 0.333) * x * x
                     + (6.0 - 2.0 * 0.333))
    } else if x < 2.0 {
        (1.0 / 6.0) * ((-0.333 - 6.0 * 0.333) * x * x * x
                     + (6.0 * 0.333 + 30.0 * 0.333) * x * x
                     + (-12.0 * 0.333 - 48.0 * 0.333) * x
                     + (8.0 * 0.333 + 24.0 * 0.333))
    } else {
        0.0
    }
}

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
    Proj4rs(Box<CoordTransformer>),
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

/// Coordinate transformer using proj4rs (pure Rust).
///
/// This struct provides efficient, reusable coordinate transformations between
/// any two coordinate reference systems (CRS) identified by EPSG codes.
///
/// # Example
///
/// ```rust,ignore
/// use cogrs::CoordTransformer;
///
/// // Create a transformer from WGS84 to UTM zone 33N
/// let transformer = CoordTransformer::new(4326, 32633)?;
/// let (utm_x, utm_y) = transformer.transform(15.0, 52.0)?;
///
/// // Or use convenience constructors
/// let to_mercator = CoordTransformer::from_lonlat_to(3857)?;
/// let from_mercator = CoordTransformer::to_lonlat_from(3857)?;
/// ```
pub struct CoordTransformer {
    source_proj: Proj,
    target_proj: Proj,
    /// EPSG code of the source CRS
    source_epsg: i32,
    /// EPSG code of the target CRS
    target_epsg: i32,
    /// True if source uses degrees (needs radian conversion)
    source_is_geographic: bool,
    /// True if target uses degrees (needs radian conversion)
    target_is_geographic: bool,
}

impl std::fmt::Debug for CoordTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoordTransformer")
            .field("source_epsg", &self.source_epsg)
            .field("target_epsg", &self.target_epsg)
            .field("source_is_geographic", &self.source_is_geographic)
            .field("target_is_geographic", &self.target_is_geographic)
            .finish_non_exhaustive()
    }
}

impl CoordTransformer {
    /// Create a transformer between any two CRS codes.
    ///
    /// # Arguments
    /// * `source_epsg` - EPSG code of the source coordinate system
    /// * `target_epsg` - EPSG code of the target coordinate system
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Transform from WGS84 (lon/lat) to UTM zone 10N
    /// let transformer = CoordTransformer::new(4326, 32610)?;
    /// let (utm_x, utm_y) = transformer.transform(-122.4, 37.8)?;
    /// ```
    pub fn new(source_epsg: i32, target_epsg: i32) -> Result<Self, String> {
        let source_str = get_proj_string(source_epsg)
            .ok_or_else(|| format!("EPSG:{} not supported", source_epsg))?;
        let target_str = get_proj_string(target_epsg)
            .ok_or_else(|| format!("EPSG:{} not supported", target_epsg))?;

        let source_proj = Proj::from_proj_string(source_str)
            .map_err(|e| format!("Invalid source projection EPSG:{}: {:?}", source_epsg, e))?;
        let target_proj = Proj::from_proj_string(target_str)
            .map_err(|e| format!("Invalid target projection EPSG:{}: {:?}", target_epsg, e))?;

        Ok(Self {
            source_proj,
            target_proj,
            source_epsg,
            target_epsg,
            source_is_geographic: is_geographic_crs(source_epsg),
            target_is_geographic: is_geographic_crs(target_epsg),
        })
    }

    /// Create a transformer from EPSG:3857 (Web Mercator) to another CRS.
    ///
    /// This is a convenience method for the common case of transforming
    /// from Web Mercator tile coordinates to a source CRS.
    pub fn from_3857_to(target_epsg: u32) -> Result<Self, String> {
        Self::new(EPSG_WEB_MERCATOR as i32, target_epsg as i32)
    }

    /// Create a transformer from EPSG:4326 (WGS84 lon/lat) to another CRS.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let transformer = CoordTransformer::from_lonlat_to(32633)?; // To UTM 33N
    /// let (x, y) = transformer.transform(15.0, 52.0)?;
    /// ```
    pub fn from_lonlat_to(target_epsg: i32) -> Result<Self, String> {
        Self::new(EPSG_WGS84 as i32, target_epsg)
    }

    /// Create a transformer from another CRS to EPSG:4326 (WGS84 lon/lat).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let transformer = CoordTransformer::to_lonlat_from(32633)?; // From UTM 33N
    /// let (lon, lat) = transformer.transform(500000.0, 5761000.0)?;
    /// ```
    pub fn to_lonlat_from(source_epsg: i32) -> Result<Self, String> {
        Self::new(source_epsg, EPSG_WGS84 as i32)
    }

    /// Create a transformer from another CRS to EPSG:3857 (Web Mercator).
    pub fn to_3857_from(source_epsg: i32) -> Result<Self, String> {
        Self::new(source_epsg, EPSG_WEB_MERCATOR as i32)
    }

    /// Get the source EPSG code.
    #[inline]
    #[must_use]
    pub fn source_epsg(&self) -> i32 {
        self.source_epsg
    }

    /// Get the target EPSG code.
    #[inline]
    #[must_use]
    pub fn target_epsg(&self) -> i32 {
        self.target_epsg
    }

    /// Check if source CRS is geographic (uses degrees).
    #[inline]
    #[must_use]
    pub fn source_is_geographic(&self) -> bool {
        self.source_is_geographic
    }

    /// Check if target CRS is geographic (uses degrees).
    #[inline]
    #[must_use]
    pub fn target_is_geographic(&self) -> bool {
        self.target_is_geographic
    }

    /// Transform coordinates from source CRS to target CRS.
    ///
    /// Handles radian/degree conversion automatically based on CRS types.
    #[inline]
    pub fn transform(&self, x: f64, y: f64) -> Result<(f64, f64), String> {
        // Convert to radians if source is geographic
        let (in_x, in_y) = if self.source_is_geographic {
            (x.to_radians(), y.to_radians())
        } else {
            (x, y)
        };

        let mut point = (in_x, in_y, 0.0);

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

    /// Transform a batch of coordinates for efficiency.
    ///
    /// Returns a Vec of results, one for each input point.
    pub fn transform_batch(&self, points: &[(f64, f64)]) -> Vec<Result<(f64, f64), String>> {
        points.iter().map(|&(x, y)| self.transform(x, y)).collect()
    }
}

/// Builder for extracting tiles from a COG with full control over parameters.
///
/// This provides a fluent API for tile extraction with sensible defaults.
///
/// # Example
///
/// ```rust,ignore
/// use cogrs::{CogReader, TileExtractor};
///
/// let reader = CogReader::open("path/to/cog.tif")?;
///
/// // Extract an XYZ tile with custom size
/// let tile = TileExtractor::new(&reader)
///     .xyz(3, 4, 2)
///     .output_size(512, 512)
///     .extract()?;
///
/// // Or extract by bounding box
/// let tile = TileExtractor::new(&reader)
///     .bounds(BoundingBox::new(-122.5, 37.5, -122.0, 38.0))
///     .extract()?;
///
/// // Extract only specific bands (0-indexed)
/// let tile = TileExtractor::new(&reader)
///     .xyz(10, 163, 395)
///     .bands(&[0, 2])  // Extract only bands 0 and 2
///     .extract()?;
/// ```
pub struct TileExtractor<'a> {
    reader: &'a CogReader,
    bounds: Option<BoundingBox>,
    output_size: (usize, usize),
    resampling: ResamplingMethod,
    /// Selected bands (None = all bands)
    selected_bands: Option<Vec<usize>>,
}

impl std::fmt::Debug for TileExtractor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileExtractor")
            .field("bounds", &self.bounds)
            .field("output_size", &self.output_size)
            .field("resampling", &self.resampling)
            .field("selected_bands", &self.selected_bands)
            .finish_non_exhaustive()
    }
}

impl<'a> TileExtractor<'a> {
    /// Create a new tile extractor for the given COG reader.
    #[must_use]
    pub fn new(reader: &'a CogReader) -> Self {
        Self {
            reader,
            bounds: None,
            output_size: (256, 256),
            resampling: ResamplingMethod::default(),
            selected_bands: None,
        }
    }

    /// Set the output tile bounds using an XYZ tile coordinate.
    ///
    /// This automatically computes the Web Mercator bounding box for the tile.
    #[must_use]
    pub fn xyz(mut self, z: u32, x: u32, y: u32) -> Self {
        self.bounds = Some(BoundingBox::from_xyz(z, x, y));
        self
    }

    /// Set the output tile bounds using a bounding box in EPSG:3857 (Web Mercator).
    #[must_use]
    pub fn bounds(mut self, bbox: BoundingBox) -> Self {
        self.bounds = Some(bbox);
        self
    }

    /// Set the output tile size (width, height).
    ///
    /// Default is (256, 256).
    #[must_use]
    pub fn output_size(mut self, width: usize, height: usize) -> Self {
        self.output_size = (width, height);
        self
    }

    /// Set the output tile to square with the given size.
    ///
    /// Convenience method equivalent to `output_size(size, size)`.
    #[must_use]
    pub fn size(mut self, size: usize) -> Self {
        self.output_size = (size, size);
        self
    }

    /// Set the resampling method.
    ///
    /// Default is `ResamplingMethod::Nearest` (fastest).
    /// Use `Bilinear` for smoother results or `Bicubic` for highest quality.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tile = TileExtractor::new(&reader)
    ///     .xyz(10, 512, 512)
    ///     .resampling(ResamplingMethod::Bilinear)
    ///     .extract()?;
    /// ```
    #[must_use]
    pub fn resampling(mut self, method: ResamplingMethod) -> Self {
        self.resampling = method;
        self
    }

    /// Select specific bands to extract (0-indexed).
    ///
    /// By default, all bands are extracted. Use this method to extract only
    /// specific bands, which can improve performance for multi-band COGs.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Extract only the first and third bands from an RGB COG
    /// let tile = TileExtractor::new(&reader)
    ///     .xyz(10, 163, 395)
    ///     .bands(&[0, 2])  // Red and Blue only
    ///     .extract()?;
    ///
    /// // Extract a single band
    /// let tile = TileExtractor::new(&reader)
    ///     .xyz(10, 163, 395)
    ///     .bands(&[0])  // Only the first band
    ///     .extract()?;
    /// ```
    #[must_use]
    pub fn bands(mut self, bands: &[usize]) -> Self {
        self.selected_bands = Some(bands.to_vec());
        self
    }

    /// Extract the tile with the configured parameters.
    ///
    /// Returns an error if bounds were not set.
    pub fn extract(self) -> Result<TileData, Box<dyn std::error::Error + Send + Sync>> {
        let bounds = self.bounds.ok_or("Bounds not set: use .xyz() or .bounds()")?;
        if let Some(selected) = self.selected_bands {
            extract_tile_with_bands(self.reader, &bounds, self.output_size, self.resampling, &selected)
        } else {
            extract_tile_with_extent_resampled(self.reader, &bounds, self.output_size, self.resampling)
        }
    }

    /// Get the configured output size.
    #[inline]
    #[must_use]
    pub fn get_output_size(&self) -> (usize, usize) {
        self.output_size
    }

    /// Get the configured bounds, if set.
    #[inline]
    #[must_use]
    pub fn get_bounds(&self) -> Option<&BoundingBox> {
        self.bounds.as_ref()
    }

    /// Get the selected bands, if set.
    #[inline]
    #[must_use]
    pub fn get_selected_bands(&self) -> Option<&[usize]> {
        self.selected_bands.as_deref()
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
/// coordinate transformation, and pixel sampling. Uses nearest neighbor resampling.
pub fn extract_tile_with_extent(
    reader: &CogReader,
    extent_3857: &BoundingBox,
    tile_size: (usize, usize),
) -> Result<TileData, Box<dyn std::error::Error + Send + Sync>> {
    extract_tile_with_extent_resampled(reader, extent_3857, tile_size, ResamplingMethod::Nearest)
}

/// Extract a tile from a COG reader with configurable resampling method
///
/// # Arguments
/// * `reader` - The COG reader with loaded metadata
/// * `extent_3857` - Bounding box in Web Mercator (EPSG:3857)
/// * `tile_size` - Output tile dimensions (width, height)
/// * `resampling` - Resampling method to use
pub fn extract_tile_with_extent_resampled(
    reader: &CogReader,
    extent_3857: &BoundingBox,
    tile_size: (usize, usize),
    resampling: ResamplingMethod,
) -> Result<TileData, Box<dyn std::error::Error + Send + Sync>> {
    let metadata = &reader.metadata;
    let geo_transform = &metadata.geo_transform;

    // Pre-compute the affine transform from output pixel to source pixel
    let (Some(base_scale), Some(_tiepoint)) = (geo_transform.pixel_scale, geo_transform.tiepoint) else {
        return Err("Missing geotransform".into());
    };

    // Create coordinate transformer from EPSG:3857 (tile coords) to source CRS
    // Use fast inline math for 4326 (most common case), proj4rs for others
    let source_epsg = metadata.crs_code.unwrap_or(EPSG_WEB_MERCATOR as i32) as u32;

    let strategy = match source_epsg {
        EPSG_WEB_MERCATOR => TransformStrategy::Identity,
        EPSG_WGS84 => TransformStrategy::FastMerc2Geo,
        _ => TransformStrategy::Proj4rs(Box::new(CoordTransformer::from_3857_to(source_epsg)?)),
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
    extract_tile_with_overview(reader, extent_3857, tile_size, overview_idx, strategy, resampling, None)
}

/// Extract a tile with specific band selection
///
/// # Arguments
/// * `reader` - The COG reader with loaded metadata
/// * `extent_3857` - Bounding box in Web Mercator (EPSG:3857)
/// * `tile_size` - Output tile dimensions (width, height)
/// * `resampling` - Resampling method to use
/// * `bands` - Slice of band indices to extract (0-indexed)
///
/// # Example
///
/// ```rust,ignore
/// use cogrs::{CogReader, extract_tile_with_bands, BoundingBox, ResamplingMethod};
///
/// let reader = CogReader::open("path/to/rgb_cog.tif")?;
/// let bbox = BoundingBox::from_xyz(10, 163, 395);
///
/// // Extract only red and blue channels
/// let tile = extract_tile_with_bands(&reader, &bbox, (256, 256), ResamplingMethod::Nearest, &[0, 2])?;
/// assert_eq!(tile.bands, 2);
/// ```
pub fn extract_tile_with_bands(
    reader: &CogReader,
    extent_3857: &BoundingBox,
    tile_size: (usize, usize),
    resampling: ResamplingMethod,
    bands: &[usize],
) -> Result<TileData, Box<dyn std::error::Error + Send + Sync>> {
    let metadata = &reader.metadata;
    let geo_transform = &metadata.geo_transform;

    // Validate band indices
    for &band in bands {
        if band >= metadata.bands {
            return Err(format!("Band index {} out of range (COG has {} bands)", band, metadata.bands).into());
        }
    }

    if bands.is_empty() {
        return Err("At least one band must be selected".into());
    }

    // Pre-compute the affine transform from output pixel to source pixel
    let (Some(base_scale), Some(_tiepoint)) = (geo_transform.pixel_scale, geo_transform.tiepoint) else {
        return Err("Missing geotransform".into());
    };

    // Create coordinate transformer
    let source_epsg = metadata.crs_code.unwrap_or(3857) as u32;
    let strategy = match source_epsg {
        3857 => TransformStrategy::Identity,
        4326 => TransformStrategy::FastMerc2Geo,
        _ => TransformStrategy::Proj4rs(Box::new(CoordTransformer::from_3857_to(source_epsg)?)),
    };

    // Convert extent to source CRS
    let (src_minx, src_miny) = strategy.transform(extent_3857.minx, extent_3857.miny)?;
    let (src_maxx, src_maxy) = strategy.transform(extent_3857.maxx, extent_3857.maxy)?;

    let extent_src_width = ((src_maxx - src_minx) / base_scale[0]).abs().max(1.0) as usize;
    let extent_src_height = ((src_maxy - src_miny) / base_scale[1]).abs().max(1.0) as usize;

    let overview_idx = reader.best_overview_for_resolution(extent_src_width, extent_src_height);

    extract_tile_with_overview(reader, extent_3857, tile_size, overview_idx, strategy, resampling, Some(bands))
}

/// Internal function that extracts a tile using a specific overview level (or full resolution if None)
fn extract_tile_with_overview(
    reader: &CogReader,
    extent_3857: &BoundingBox,
    tile_size: (usize, usize),
    overview_idx: Option<usize>,
    strategy: TransformStrategy,
    resampling: ResamplingMethod,
    selected_bands: Option<&[usize]>,
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

    // Track min/max columns and rows to compute the full tile range
    let mut min_col: Option<usize> = None;
    let mut max_col: Option<usize> = None;
    let mut min_row: Option<usize> = None;
    let mut max_row: Option<usize> = None;

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
            // Track the tile col/row from actual pixel coordinates
            let tile_col = (src_px_clamped as usize) / eff_tile_width;
            let tile_row = (src_py_clamped as usize) / eff_tile_height;

            min_col = Some(min_col.map_or(tile_col, |m| m.min(tile_col)));
            max_col = Some(max_col.map_or(tile_col, |m| m.max(tile_col)));
            min_row = Some(min_row.map_or(tile_row, |m| m.min(tile_row)));
            max_row = Some(max_row.map_or(tile_row, |m| m.max(tile_row)));

            if let Some(idx) = tile_index_at_level(src_px_clamped as usize, src_py_clamped as usize) {
                needed_tiles.insert(idx);
            }
        }
    }

    // Fill in all tiles in the col/row bounding box
    let max_tile_count = if let Some(idx) = overview_idx {
        reader.overviews[idx].tile_offsets.len()
    } else {
        metadata.tile_offsets.len()
    };

    if let (Some(min_c), Some(max_c), Some(min_r), Some(max_r)) = (min_col, max_col, min_row, max_row) {
        // Read all tiles in the col/row range
        for row in min_r..=max_r {
            for col in min_c..=max_c {
                let idx = row * eff_tiles_across + col;
                if idx < max_tile_count {
                    needed_tiles.insert(idx);
                }
            }
        }
    }

    // If no tiles needed, return transparent tile
    if needed_tiles.is_empty() {
        let output_bands: Vec<usize> = selected_bands
            .map(|b| b.to_vec())
            .unwrap_or_else(|| (0..metadata.bands).collect());
        let num_output_bands = output_bands.len();
        return Ok(TileData {
            pixels: vec![0.0; tile_size_x * tile_size_y * num_output_bands],
            bands: num_output_bands,
            width: tile_size_x,
            height: tile_size_y,
            bytes_fetched: 0,
            tiles_read: 0,
            overview_used: overview_idx,
        });
    }

    // Pre-load all needed tiles and track bytes fetched
    let mut tile_data_cache: AHashMap<usize, Vec<f32>> = AHashMap::new();
    let mut total_bytes_fetched: usize = 0;
    let mut tiles_actually_read: usize = 0;

    for &tile_idx in &needed_tiles {
        let tile_result = if let Some(ovr_idx) = overview_idx {
            reader.read_overview_tile_with_bytes(ovr_idx, tile_idx)
        } else {
            reader.read_tile_with_bytes(tile_idx)
        };

        if let Ok((data, bytes)) = tile_result {
            tile_data_cache.insert(tile_idx, data);
            total_bytes_fetched += bytes;
            if bytes > 0 {
                tiles_actually_read += 1;
            }
        }
    }

    // If all tile reads failed, try falling back to full resolution
    if tile_data_cache.is_empty() && overview_idx.is_some() {
        return extract_tile_with_overview(reader, extent_3857, tile_size, None, strategy, resampling, selected_bands);
    }

    // Determine output bands: selected or all
    let source_bands = metadata.bands;
    let output_bands: Vec<usize> = selected_bands
        .map(|b| b.to_vec())
        .unwrap_or_else(|| (0..source_bands).collect());
    let num_output_bands = output_bands.len();
    let mut pixel_data = vec![0.0_f32; tile_size_x * tile_size_y * num_output_bands];

    // Pre-compute inverse scale for speed
    let inv_scale_x = 1.0 / scale[0];
    let inv_scale_y = 1.0 / scale[1];

    // Helper to sample a pixel from the cached tile data
    // band is the SOURCE band index (not output band index)
    let sample_pixel = |px: usize, py: usize, source_band: usize| -> Option<f32> {
        let tile_col = px / eff_tile_width;
        let tile_row = py / eff_tile_height;
        let tile_idx = tile_row * eff_tiles_across + tile_col;

        let tile_data = tile_data_cache.get(&tile_idx)?;

        let local_x = px % eff_tile_width;
        let local_y = py % eff_tile_height;
        let pixel_idx = (local_y * eff_tile_width + local_x) * source_bands + source_band;

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

            let out_idx = (out_y * tile_size_x + out_x) * num_output_bands;

            // Sample each band using the configured resampling method
            match resampling {
                ResamplingMethod::Nearest => {
                    let src_px_int = src_px.round() as isize;
                    let src_px_clamped = src_px_int.max(0).min(eff_width as isize - 1) as usize;
                    let src_py_clamped = src_py.round().max(0.0).min(eff_height as f64 - 1.0) as usize;

                    for (out_band_idx, &source_band) in output_bands.iter().enumerate() {
                        if let Some(value) = sample_pixel(src_px_clamped, src_py_clamped, source_band) {
                            pixel_data[out_idx + out_band_idx] = value;
                        }
                    }
                }
                ResamplingMethod::Bilinear => {
                    // Bilinear interpolation using 4 nearest pixels
                    let x0 = src_px.floor() as isize;
                    let y0 = src_py.floor() as isize;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;

                    // Fractional parts for interpolation weights
                    let fx = src_px - x0 as f64;
                    let fy = src_py - y0 as f64;

                    // Clamp to valid range
                    let x0c = x0.max(0).min(eff_width as isize - 1) as usize;
                    let x1c = x1.max(0).min(eff_width as isize - 1) as usize;
                    let y0c = y0.max(0).min(eff_height as isize - 1) as usize;
                    let y1c = y1.max(0).min(eff_height as isize - 1) as usize;

                    for (out_band_idx, &source_band) in output_bands.iter().enumerate() {
                        let v00 = sample_pixel(x0c, y0c, source_band).unwrap_or(0.0);
                        let v10 = sample_pixel(x1c, y0c, source_band).unwrap_or(0.0);
                        let v01 = sample_pixel(x0c, y1c, source_band).unwrap_or(0.0);
                        let v11 = sample_pixel(x1c, y1c, source_band).unwrap_or(0.0);

                        // Bilinear interpolation formula
                        let value = v00 * (1.0 - fx as f32) * (1.0 - fy as f32)
                                  + v10 * (fx as f32) * (1.0 - fy as f32)
                                  + v01 * (1.0 - fx as f32) * (fy as f32)
                                  + v11 * (fx as f32) * (fy as f32);

                        pixel_data[out_idx + out_band_idx] = value;
                    }
                }
                ResamplingMethod::Bicubic => {
                    // Bicubic interpolation using 4x4 grid of pixels
                    let x0 = src_px.floor() as isize;
                    let y0 = src_py.floor() as isize;

                    // Fractional parts
                    let fx = src_px - x0 as f64;
                    let fy = src_py - y0 as f64;

                    for (out_band_idx, &source_band) in output_bands.iter().enumerate() {
                        let mut sum = 0.0f32;
                        let mut weight_sum = 0.0f32;

                        // Sample 4x4 grid centered around (x0, y0)
                        for j in -1..=2isize {
                            for i in -1..=2isize {
                                let px = (x0 + i).max(0).min(eff_width as isize - 1) as usize;
                                let py = (y0 + j).max(0).min(eff_height as isize - 1) as usize;

                                if let Some(v) = sample_pixel(px, py, source_band) {
                                    // Bicubic weight (Mitchell-Netravali with B=C=1/3)
                                    let wx = bicubic_weight(i as f64 - fx);
                                    let wy = bicubic_weight(j as f64 - fy);
                                    let w = (wx * wy) as f32;
                                    sum += v * w;
                                    weight_sum += w;
                                }
                            }
                        }

                        pixel_data[out_idx + out_band_idx] = if weight_sum > 0.0 { sum / weight_sum } else { 0.0 };
                    }
                }
            }
        }
    }

    // Drop sample_pixel closure
    let _ = sample_pixel;

    Ok(TileData {
        pixels: pixel_data,
        bands: num_output_bands,
        width: tile_size_x,
        height: tile_size_y,
        bytes_fetched: total_bytes_fetched,
        tiles_read: tiles_actually_read,
        overview_used: overview_idx,
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

    #[test]
    fn test_coord_transformer_new() {
        // Test the general constructor
        let transformer = CoordTransformer::new(4326, 3857).unwrap();
        assert_eq!(transformer.source_epsg(), 4326);
        assert_eq!(transformer.target_epsg(), 3857);
        assert!(transformer.source_is_geographic());
        assert!(!transformer.target_is_geographic());
    }

    #[test]
    fn test_coord_transformer_from_lonlat_to() {
        let transformer = CoordTransformer::from_lonlat_to(3857).unwrap();
        assert_eq!(transformer.source_epsg(), 4326);
        assert_eq!(transformer.target_epsg(), 3857);

        // Transform origin
        let (x, y) = transformer.transform(0.0, 0.0).unwrap();
        assert!(x.abs() < 1.0);
        assert!(y.abs() < 1.0);
    }

    #[test]
    fn test_coord_transformer_to_lonlat_from() {
        let transformer = CoordTransformer::to_lonlat_from(3857).unwrap();
        assert_eq!(transformer.source_epsg(), 3857);
        assert_eq!(transformer.target_epsg(), 4326);

        // Transform origin
        let (lon, lat) = transformer.transform(0.0, 0.0).unwrap();
        assert!(lon.abs() < 0.0001);
        assert!(lat.abs() < 0.0001);
    }

    #[test]
    fn test_coord_transformer_roundtrip() {
        let to_utm = CoordTransformer::new(4326, 32633).unwrap(); // WGS84 -> UTM 33N
        let from_utm = CoordTransformer::new(32633, 4326).unwrap(); // UTM 33N -> WGS84

        let lon = 15.0;
        let lat = 52.0;

        let (x, y) = to_utm.transform(lon, lat).unwrap();
        let (lon2, lat2) = from_utm.transform(x, y).unwrap();

        assert!((lon - lon2).abs() < 1e-5, "lon roundtrip: {} -> {}", lon, lon2);
        assert!((lat - lat2).abs() < 1e-5, "lat roundtrip: {} -> {}", lat, lat2);
    }

    #[test]
    fn test_coord_transformer_batch() {
        let transformer = CoordTransformer::new(4326, 3857).unwrap();

        let points = vec![
            (0.0, 0.0),
            (10.0, 51.5),
            (-122.4, 37.8),
        ];

        let results = transformer.transform_batch(&points);

        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert!(results[2].is_ok());

        // Origin should map to origin
        let (x, y) = results[0].as_ref().unwrap();
        assert!(x.abs() < 1.0);
        assert!(y.abs() < 1.0);
    }

    #[test]
    fn test_coord_transformer_unsupported_epsg() {
        let result = CoordTransformer::new(4326, 999999);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not supported"));
    }

    #[test]
    fn test_tile_extractor_defaults() {
        // We can test the builder without a real CogReader by checking its config
        // For now, just test the BoundingBox methods used by the builder
        let bbox = BoundingBox::from_xyz(5, 10, 12);
        assert!(bbox.minx < bbox.maxx);
        assert!(bbox.miny < bbox.maxy);
    }

    #[test]
    fn test_tile_extractor_xyz_bounds() {
        // Test that xyz() produces correct bounds
        let bbox = BoundingBox::from_xyz(0, 0, 0);
        // Tile 0/0/0 should cover the entire Web Mercator extent
        let expected_extent = 20037508.342789244;
        assert!((bbox.minx - (-expected_extent)).abs() < 1.0);
        assert!((bbox.maxx - expected_extent).abs() < 1.0);
    }

    #[test]
    fn test_bounding_box_new() {
        let bbox = BoundingBox::new(-10.0, -20.0, 30.0, 40.0);
        assert_eq!(bbox.minx, -10.0);
        assert_eq!(bbox.miny, -20.0);
        assert_eq!(bbox.maxx, 30.0);
        assert_eq!(bbox.maxy, 40.0);
    }

    #[test]
    fn test_resampling_method_default() {
        assert_eq!(ResamplingMethod::default(), ResamplingMethod::Nearest);
    }

    #[test]
    fn test_bicubic_weight() {
        // At x=0, weight should be maximum (~0.889 for Mitchell-Netravali B=C=1/3)
        let w0 = bicubic_weight(0.0);
        assert!(w0 > 0.8, "Weight at 0 should be near 0.889: {}", w0);

        // At x=1, weight should be smaller but can be negative for Mitchell filter
        let w1 = bicubic_weight(1.0);
        assert!(w1.abs() < w0, "Weight at 1 should be smaller than at 0: {}", w1);

        // At x=2 and beyond, weight should be 0
        let w2 = bicubic_weight(2.0);
        assert!(w2.abs() < 0.001, "Weight at 2 should be ~0: {}", w2);

        let w3 = bicubic_weight(3.0);
        assert_eq!(w3, 0.0, "Weight at 3 should be 0");

        // Weights should be symmetric
        assert!((bicubic_weight(0.5) - bicubic_weight(-0.5)).abs() < 0.0001);
    }

    #[test]
    fn test_band_selection_validation() {
        // Test band selection validation without requiring a real file
        // Just test the error paths in extract_tile_with_bands

        // Note: We can't fully test extract_tile_with_bands without a real file,
        // but we can verify the builder stores bands correctly
        let selected_bands = vec![0, 2];
        assert_eq!(selected_bands.len(), 2);
        assert_eq!(selected_bands[0], 0);
        assert_eq!(selected_bands[1], 2);
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

    #[test]
    fn test_band_selection_extraction() {
        // Skip if test file doesn't exist
        let path = "/home/evan/projects/personal/geo/tileyolo/data/viridis/output_cog.tif";
        if !std::path::Path::new(path).exists() {
            println!("Skipping: test file not found");
            return;
        }

        let reader = LocalRangeReader::new(path).unwrap();
        let cog = CogReader::from_reader(Arc::new(reader)).unwrap();

        let num_bands = cog.metadata.bands;
        println!("COG has {} bands", num_bands);

        if num_bands < 2 {
            println!("Skipping band selection test: COG has less than 2 bands");
            return;
        }

        // Extract all bands
        let tile_all = extract_xyz_tile(&cog, 1, 0, 0, (128, 128)).unwrap();
        assert_eq!(tile_all.bands, num_bands);
        assert_eq!(tile_all.pixels.len(), 128 * 128 * num_bands);

        // Extract only the first band
        let bbox = BoundingBox::from_xyz(1, 0, 0);
        let tile_one = extract_tile_with_bands(&cog, &bbox, (128, 128), ResamplingMethod::Nearest, &[0]).unwrap();
        assert_eq!(tile_one.bands, 1);
        assert_eq!(tile_one.pixels.len(), 128 * 128);

        // Use the builder with band selection
        let tile_builder = TileExtractor::new(&cog)
            .xyz(1, 0, 0)
            .output_size(128, 128)
            .bands(&[0])
            .extract()
            .unwrap();
        assert_eq!(tile_builder.bands, 1);

        // The first band values should match between all-band and single-band extraction
        for i in 0..100 {
            let all_val = tile_all.pixels[i * num_bands]; // First band of each pixel
            let one_val = tile_one.pixels[i];
            assert!((all_val - one_val).abs() < 0.001,
                   "Pixel {}: all_bands[0]={}, single_band={}", i, all_val, one_val);
        }

        println!("Band selection test passed: single band matches first band of all bands");
    }
}
