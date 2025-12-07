//! Point query functionality for sampling COG values at geographic coordinates.
//!
//! This module provides the [`PointQuery`] trait for sampling pixel values at geographic
//! coordinates, with support for any coordinate reference system (CRS).
//!
//! # Example
//!
//! ```rust,ignore
//! use cogrs::{CogReader, PointQuery};
//!
//! let reader = CogReader::open("path/to/cog.tif")?;
//!
//! // Sample all bands at a lon/lat coordinate
//! let result = reader.sample_lonlat(-122.4, 37.8)?;
//! for (band, value) in &result.values {
//!     println!("Band {}: {}", band, value);
//! }
//!
//! // Sample at coordinates in a specific CRS (e.g., UTM zone 10N)
//! let result = reader.sample_crs(32610, 551000.0, 4185000.0)?;
//! ```

use std::collections::HashMap;

use crate::cog_reader::CogReader;
use crate::geometry::projection::project_point;
use crate::tiff_utils::AnyResult;

/// Result of a point query containing band values as a HashMap.
///
/// The `values` field maps band index (0-based) to the sampled value.
/// This makes it easy to access specific bands and iterate over all values.
#[derive(Debug, Clone)]
pub struct PointQueryResult {
    /// Values for each band, keyed by band index (0-based).
    /// Values are f32::NAN if the pixel is nodata.
    pub values: HashMap<usize, f32>,

    /// Number of bands in the source raster
    pub bands: usize,

    /// Whether this point was within the raster bounds
    pub is_valid: bool,

    /// The pixel coordinates that were sampled (if valid)
    pub pixel_coords: Option<(usize, usize)>,

    /// The CRS of the input coordinates
    pub input_crs: i32,

    /// The native CRS of the raster
    pub raster_crs: i32,
}

impl PointQueryResult {
    /// Get the value for a specific band, or None if band doesn't exist
    pub fn get(&self, band: usize) -> Option<f32> {
        self.values.get(&band).copied()
    }

    /// Get all values as a Vec, ordered by band index
    pub fn to_vec(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.bands);
        for i in 0..self.bands {
            result.push(self.values.get(&i).copied().unwrap_or(f32::NAN));
        }
        result
    }

    /// Check if any band has a valid (non-NaN) value
    pub fn has_valid_data(&self) -> bool {
        self.values.values().any(|v| !v.is_nan())
    }

    /// Get the number of bands with valid (non-NaN) values
    pub fn valid_band_count(&self) -> usize {
        self.values.values().filter(|v| !v.is_nan()).count()
    }

    /// Iterate over (band_index, value) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &f32)> {
        self.values.iter()
    }
}

/// Trait for point queries at geographic coordinates.
///
/// This trait extends [`CogReader`] with methods to sample pixel values
/// at geographic coordinates in any supported CRS.
pub trait PointQuery {
    /// Sample all bands at a lon/lat coordinate (EPSG:4326).
    ///
    /// # Arguments
    /// * `lon` - Longitude in degrees (-180 to 180)
    /// * `lat` - Latitude in degrees (-90 to 90)
    ///
    /// # Returns
    /// A [`PointQueryResult`] containing values for all bands.
    fn sample_lonlat(&self, lon: f64, lat: f64) -> AnyResult<PointQueryResult>;

    /// Sample all bands at coordinates in a specific CRS.
    ///
    /// # Arguments
    /// * `crs` - EPSG code of the input coordinates
    /// * `x` - X coordinate in the specified CRS
    /// * `y` - Y coordinate in the specified CRS
    ///
    /// # Returns
    /// A [`PointQueryResult`] containing values for all bands.
    fn sample_crs(&self, crs: i32, x: f64, y: f64) -> AnyResult<PointQueryResult>;

    /// Sample a single band at lon/lat coordinates (EPSG:4326).
    ///
    /// # Arguments
    /// * `lon` - Longitude in degrees
    /// * `lat` - Latitude in degrees
    /// * `band` - Band index (0-based)
    ///
    /// # Returns
    /// The sampled value, or None if out of bounds or band doesn't exist.
    fn sample_band_lonlat(&self, lon: f64, lat: f64, band: usize) -> AnyResult<Option<f32>>;

    /// Sample a single band at coordinates in a specific CRS.
    ///
    /// # Arguments
    /// * `crs` - EPSG code of the input coordinates
    /// * `x` - X coordinate in the specified CRS
    /// * `y` - Y coordinate in the specified CRS
    /// * `band` - Band index (0-based)
    ///
    /// # Returns
    /// The sampled value, or None if out of bounds or band doesn't exist.
    fn sample_band_crs(&self, crs: i32, x: f64, y: f64, band: usize) -> AnyResult<Option<f32>>;

    /// Sample multiple points at once (batch query) for efficiency.
    ///
    /// # Arguments
    /// * `points` - Slice of (lon, lat) coordinates in EPSG:4326
    ///
    /// # Returns
    /// A Vec of [`PointQueryResult`], one for each input point.
    fn sample_points_lonlat(&self, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>>;

    /// Sample multiple points in a specific CRS.
    ///
    /// # Arguments
    /// * `crs` - EPSG code of the input coordinates
    /// * `points` - Slice of (x, y) coordinates in the specified CRS
    ///
    /// # Returns
    /// A Vec of [`PointQueryResult`], one for each input point.
    fn sample_points_crs(&self, crs: i32, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>>;
}

impl PointQuery for CogReader {
    fn sample_lonlat(&self, lon: f64, lat: f64) -> AnyResult<PointQueryResult> {
        self.sample_crs(4326, lon, lat)
    }

    fn sample_crs(&self, crs: i32, x: f64, y: f64) -> AnyResult<PointQueryResult> {
        let source_crs = self.metadata.crs_code.unwrap_or(4326);

        // Transform to source CRS if needed
        let (src_x, src_y) = if crs == source_crs {
            (x, y)
        } else {
            project_point(crs, source_crs, x, y)?
        };

        // Convert world coordinates to pixel coordinates
        let Some((px, py)) = self.metadata.geo_transform.world_to_pixel(src_x, src_y) else {
            return Ok(PointQueryResult {
                values: HashMap::new(),
                bands: self.metadata.bands,
                is_valid: false,
                pixel_coords: None,
                input_crs: crs,
                raster_crs: source_crs,
            });
        };

        // Bounds check
        if px < 0.0 || py < 0.0 ||
           px >= self.metadata.width as f64 ||
           py >= self.metadata.height as f64 {
            return Ok(PointQueryResult {
                values: HashMap::new(),
                bands: self.metadata.bands,
                is_valid: false,
                pixel_coords: None,
                input_crs: crs,
                raster_crs: source_crs,
            });
        }

        let px_int = px as usize;
        let py_int = py as usize;

        // Sample all bands
        let mut values = HashMap::with_capacity(self.metadata.bands);

        for band in 0..self.metadata.bands {
            let val = self.sample(band, px_int, py_int)?.unwrap_or(f32::NAN);
            values.insert(band, val);
        }

        Ok(PointQueryResult {
            values,
            bands: self.metadata.bands,
            is_valid: true,
            pixel_coords: Some((px_int, py_int)),
            input_crs: crs,
            raster_crs: source_crs,
        })
    }

    fn sample_band_lonlat(&self, lon: f64, lat: f64, band: usize) -> AnyResult<Option<f32>> {
        self.sample_band_crs(4326, lon, lat, band)
    }

    fn sample_band_crs(&self, crs: i32, x: f64, y: f64, band: usize) -> AnyResult<Option<f32>> {
        if band >= self.metadata.bands {
            return Ok(None);
        }

        let source_crs = self.metadata.crs_code.unwrap_or(4326);

        // Transform to source CRS if needed
        let (src_x, src_y) = if crs == source_crs {
            (x, y)
        } else {
            project_point(crs, source_crs, x, y)?
        };

        // Convert world coordinates to pixel coordinates
        let Some((px, py)) = self.metadata.geo_transform.world_to_pixel(src_x, src_y) else {
            return Ok(None);
        };

        // Bounds check
        if px < 0.0 || py < 0.0 ||
           px >= self.metadata.width as f64 ||
           py >= self.metadata.height as f64 {
            return Ok(None);
        }

        self.sample(band, px as usize, py as usize)
    }

    fn sample_points_lonlat(&self, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>> {
        self.sample_points_crs(4326, points)
    }

    fn sample_points_crs(&self, crs: i32, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>> {
        let mut results = Vec::with_capacity(points.len());

        for &(x, y) in points {
            results.push(self.sample_crs(crs, x, y)?);
        }

        Ok(results)
    }
}

/// Convenience function to sample all bands at a lon/lat coordinate.
///
/// This is a shorthand for `reader.sample_lonlat(lon, lat)?.values`.
///
/// # Example
///
/// ```rust,ignore
/// use cogrs::{CogReader, sample_point};
///
/// let reader = CogReader::open("path/to/cog.tif")?;
/// let values = sample_point(&reader, -122.4, 37.8)?;
/// ```
pub fn sample_point(reader: &CogReader, lon: f64, lat: f64) -> AnyResult<HashMap<usize, f32>> {
    Ok(reader.sample_lonlat(lon, lat)?.values)
}

/// Convenience function to sample all bands at a coordinate in a specific CRS.
pub fn sample_point_crs(
    reader: &CogReader,
    crs: i32,
    x: f64,
    y: f64,
) -> AnyResult<HashMap<usize, f32>> {
    Ok(reader.sample_crs(crs, x, y)?.values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_query_result_to_vec() {
        let mut values = HashMap::new();
        values.insert(0, 1.0);
        values.insert(1, 2.0);
        values.insert(2, 3.0);

        let result = PointQueryResult {
            values,
            bands: 3,
            is_valid: true,
            pixel_coords: Some((10, 20)),
            input_crs: 4326,
            raster_crs: 4326,
        };

        let vec = result.to_vec();
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_point_query_result_get() {
        let mut values = HashMap::new();
        values.insert(0, 42.0);
        values.insert(1, 100.0);

        let result = PointQueryResult {
            values,
            bands: 2,
            is_valid: true,
            pixel_coords: Some((5, 5)),
            input_crs: 4326,
            raster_crs: 3857,
        };

        assert_eq!(result.get(0), Some(42.0));
        assert_eq!(result.get(1), Some(100.0));
        assert_eq!(result.get(2), None);
    }

    #[test]
    fn test_point_query_result_has_valid_data() {
        let mut values = HashMap::new();
        values.insert(0, f32::NAN);
        values.insert(1, 5.0);

        let result = PointQueryResult {
            values,
            bands: 2,
            is_valid: true,
            pixel_coords: Some((0, 0)),
            input_crs: 4326,
            raster_crs: 4326,
        };

        assert!(result.has_valid_data());
        assert_eq!(result.valid_band_count(), 1);
    }

    #[test]
    fn test_point_query_result_all_nan() {
        let mut values = HashMap::new();
        values.insert(0, f32::NAN);
        values.insert(1, f32::NAN);

        let result = PointQueryResult {
            values,
            bands: 2,
            is_valid: true,
            pixel_coords: Some((0, 0)),
            input_crs: 4326,
            raster_crs: 4326,
        };

        assert!(!result.has_valid_data());
        assert_eq!(result.valid_band_count(), 0);
    }
}
