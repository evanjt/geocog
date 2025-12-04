//! COG source discovery and abstraction.
//!
//! This module provides traits and types for discovering COG files from various sources
//! (local directories, S3 buckets, etc.) and extracting their metadata.
//!
//! # Example
//!
//! ```rust,ignore
//! use geocog::source::{CogSource, LocalCogSource, LocalScanOptions, CogEntry};
//!
//! // Discover COGs in a directory with default options (recursive)
//! let source = LocalCogSource::scan("/path/to/data", LocalScanOptions::default())?;
//!
//! // Or scan layer-style directories (depth=2)
//! let source = LocalCogSource::scan("/path/to/layers", LocalScanOptions::layers())?;
//!
//! for entry in source.entries() {
//!     println!("Found COG: {} at {:?}", entry.name, entry.location);
//! }
//! ```

pub mod local;
pub mod s3;

pub use local::{LocalCogSource, LocalScanOptions, LocalSourceStats};
pub use s3::{S3CogSource, S3ScanOptions, S3SourceStats};

use std::path::PathBuf;

use crate::cog_reader::{CogMetadata, CogReader};
use crate::geometry::projection::project_point;
use crate::xyz_tile::BoundingBox;

/// Location of a COG file (local path or remote URL).
#[derive(Debug, Clone)]
pub enum CogLocation {
    /// Local filesystem path
    Local(PathBuf),
    /// S3 URL (s3://bucket/key)
    S3 { bucket: String, key: String },
    /// HTTP(S) URL
    Http(String),
}

impl CogLocation {
    /// Get a display string for the location
    pub fn display(&self) -> String {
        match self {
            CogLocation::Local(path) => path.display().to_string(),
            CogLocation::S3 { bucket, key } => format!("s3://{}/{}", bucket, key),
            CogLocation::Http(url) => url.clone(),
        }
    }
}

/// Metadata about a discovered COG file.
///
/// This struct contains essential information extracted from a COG file during discovery,
/// without loading the full raster data. It's designed to be lightweight and cacheable.
#[derive(Debug, Clone)]
pub struct CogEntry {
    /// Unique name/identifier for this COG (usually derived from filename)
    pub name: String,
    /// Location of the COG file
    pub location: CogLocation,
    /// File size in bytes (if known)
    pub size_bytes: Option<u64>,
    /// Last modification time (if known)
    pub last_modified: Option<std::time::SystemTime>,
    /// CRS EPSG code (e.g., 4326, 3857, 32633)
    pub crs_code: i32,
    /// Bounding box in the native CRS
    pub bounds: BoundingBox,
    /// Bounding box in EPSG:4326 (WGS84) for easy comparison
    pub bounds_wgs84: BoundingBox,
    /// Number of bands (1=grayscale, 3=RGB, 4=RGBA)
    pub bands: usize,
    /// Image width in pixels
    pub width: usize,
    /// Image height in pixels
    pub height: usize,
    /// Whether the file is tiled (COG-optimized) vs stripped
    pub is_tiled: bool,
    /// Whether the file has overviews (pyramid levels)
    pub has_overviews: bool,
    /// Number of overview levels
    pub overview_count: usize,
    /// NoData value (if defined)
    pub nodata: Option<f64>,
}

impl CogEntry {
    /// Create a CogEntry from a CogReader's metadata.
    ///
    /// Extracts all relevant metadata from the reader without loading raster data.
    pub fn from_reader(
        reader: &CogReader,
        name: String,
        location: CogLocation,
        size_bytes: Option<u64>,
        last_modified: Option<std::time::SystemTime>,
    ) -> Result<Self, String> {
        let metadata = &reader.metadata;

        // Extract bounds from GeoTransform
        let bounds = Self::extract_bounds(metadata);
        let crs_code = metadata.crs_code.unwrap_or(0);

        // Project bounds to WGS84 for easy comparison
        let bounds_wgs84 = if crs_code == 4326 {
            bounds
        } else if crs_code > 0 {
            Self::project_bounds_to_wgs84(&bounds, crs_code)?
        } else {
            bounds // Unknown CRS, keep as-is
        };

        Ok(Self {
            name,
            location,
            size_bytes,
            last_modified,
            crs_code,
            bounds,
            bounds_wgs84,
            bands: metadata.bands,
            width: metadata.width,
            height: metadata.height,
            is_tiled: metadata.is_tiled,
            has_overviews: !reader.overviews.is_empty(),
            overview_count: reader.overviews.len(),
            nodata: metadata.nodata,
        })
    }

    /// Extract bounding box from COG metadata.
    fn extract_bounds(metadata: &CogMetadata) -> BoundingBox {
        let gt = &metadata.geo_transform;
        if let (Some(scale), Some(tie)) = (&gt.pixel_scale, &gt.tiepoint) {
            // Tiepoint (i, j, k, x, y, z) maps pixel (i,j,k) to world (x,y,z)
            let origin_x = tie[3];
            let origin_y = tie[4];
            let pixel_width = scale[0];
            let pixel_height = scale[1];

            let minx = origin_x;
            let maxy = origin_y;
            let maxx = origin_x + pixel_width * metadata.width as f64;
            let miny = origin_y - pixel_height * metadata.height as f64;

            BoundingBox::new(minx, miny, maxx, maxy)
        } else {
            // No geotransform - use pixel coordinates
            BoundingBox::new(0.0, 0.0, metadata.width as f64, metadata.height as f64)
        }
    }

    /// Project bounds to WGS84 (EPSG:4326).
    fn project_bounds_to_wgs84(bounds: &BoundingBox, source_crs: i32) -> Result<BoundingBox, String> {
        let (minx, miny) = project_point(source_crs, 4326, bounds.minx, bounds.miny)?;
        let (maxx, maxy) = project_point(source_crs, 4326, bounds.maxx, bounds.maxy)?;
        Ok(BoundingBox::new(minx, miny, maxx, maxy))
    }

    /// Check if this COG intersects with a given bounding box (in WGS84).
    pub fn intersects_wgs84(&self, bbox: &BoundingBox) -> bool {
        !(self.bounds_wgs84.maxx < bbox.minx
            || self.bounds_wgs84.minx > bbox.maxx
            || self.bounds_wgs84.maxy < bbox.miny
            || self.bounds_wgs84.miny > bbox.maxy)
    }
}

/// Trait for discovering COG files from a source.
///
/// Implementors scan a data source (directory, S3 bucket, etc.) and return
/// metadata about discovered COG files.
pub trait CogSource: Send + Sync {
    /// Get all discovered COG entries.
    fn entries(&self) -> &[CogEntry];

    /// Get a COG entry by name.
    fn get(&self, name: &str) -> Option<&CogEntry>;

    /// Get the number of discovered COGs.
    fn len(&self) -> usize {
        self.entries().len()
    }

    /// Check if no COGs were discovered.
    fn is_empty(&self) -> bool {
        self.entries().is_empty()
    }

    /// Find all COGs that intersect a bounding box (in WGS84).
    fn find_intersecting(&self, bbox: &BoundingBox) -> Vec<&CogEntry> {
        self.entries()
            .iter()
            .filter(|e| e.intersects_wgs84(bbox))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cog_location_display() {
        let local = CogLocation::Local(PathBuf::from("/path/to/file.tif"));
        assert_eq!(local.display(), "/path/to/file.tif");

        let s3 = CogLocation::S3 {
            bucket: "my-bucket".to_string(),
            key: "data/file.tif".to_string(),
        };
        assert_eq!(s3.display(), "s3://my-bucket/data/file.tif");

        let http = CogLocation::Http("https://example.com/file.tif".to_string());
        assert_eq!(http.display(), "https://example.com/file.tif");
    }

    #[test]
    fn test_bounding_box_intersection() {
        let entry = CogEntry {
            name: "test".to_string(),
            location: CogLocation::Local(PathBuf::from("test.tif")),
            size_bytes: None,
            last_modified: None,
            crs_code: 4326,
            bounds: BoundingBox::new(0.0, 0.0, 10.0, 10.0),
            bounds_wgs84: BoundingBox::new(0.0, 0.0, 10.0, 10.0),
            bands: 1,
            width: 100,
            height: 100,
            is_tiled: true,
            has_overviews: true,
            overview_count: 3,
            nodata: None,
        };

        // Intersecting bbox
        let intersecting = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        assert!(entry.intersects_wgs84(&intersecting));

        // Non-intersecting bbox
        let non_intersecting = BoundingBox::new(20.0, 20.0, 30.0, 30.0);
        assert!(!entry.intersects_wgs84(&non_intersecting));
    }
}
