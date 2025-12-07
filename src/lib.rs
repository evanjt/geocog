//! # cogrs - Pure Rust COG (Cloud Optimized GeoTIFF) Reader
//!
//! A high-performance library for reading Cloud Optimized GeoTIFFs without GDAL.
//!
//! ## Features
//!
//! - **Metadata-only initialization**: Reads only IFD headers on open (~5-20ms)
//! - **Range requests**: Efficient partial reads from local files, HTTP, or S3
//! - **Streaming**: Never loads entire file into memory
//! - **Compression**: Supports DEFLATE, LZW, ZSTD, and uncompressed
//! - **Overviews**: Automatic pyramid level selection with quality validation
//! - **Coordinate transforms**: Pure Rust proj4rs for CRS transformations
//! - **Point queries**: Sample values at geographic coordinates
//! - **XYZ tiles**: Extract map tiles with automatic reprojection
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use cogrs::{CogReader, PointQuery, extract_xyz_tile};
//!
//! // Open a COG from local file, HTTP, or S3
//! let reader = CogReader::open("path/to/file.tif")?;
//!
//! // Sample pixel values at a geographic coordinate
//! let result = reader.sample_lonlat(-122.4, 37.8)?;
//! for (band, value) in &result.values {
//!     println!("Band {}: {}", band, value);
//! }
//!
//! // Extract an XYZ map tile
//! let tile = extract_xyz_tile(&reader, 10, 163, 395, (256, 256))?;
//! println!("Tile has {} bands, {} pixels", tile.bands, tile.pixels.len());
//! ```
//!
//! ## Architecture
//!
//! The library is organized into several modules:
//!
//! - [`cog_reader`]: Core COG metadata parsing and tile reading
//! - [`point_query`]: Geographic coordinate sampling via [`PointQuery`] trait
//! - [`xyz_tile`]: XYZ tile extraction with [`TileExtractor`] builder
//! - [`geometry`]: Coordinate types ([`Point`], [`BoundingBox`]) and projections
//! - [`range_reader`]: I/O abstraction for local/HTTP/S3 sources
//! - [`source`]: COG discovery from directories and S3 buckets
//! - [`tile_cache`]: Global LRU cache for decompressed tiles
//! - [`s3`]: S3-compatible storage backend
//! - [`raster`]: Raster data abstraction trait

// ============================================================================
// Public modules
// ============================================================================

pub mod cog_reader;
pub mod geometry;
pub mod lzw_fallback;
pub mod point_query;
pub mod range_reader;
pub mod raster;
pub mod s3;
pub mod source;
pub mod tiff_chunked;
pub mod tiff_utils;
pub mod tile_cache;
pub mod xyz_tile;

// ============================================================================
// Core COG Types
// ============================================================================

pub use cog_reader::{
    CogReader,
    CogMetadata,
    CogDataType,
    Compression,
    GeoTransform,
    OverviewMetadata,
    OverviewQualityHint,
};

// ============================================================================
// Point Queries
// ============================================================================

pub use point_query::{
    PointQuery,
    PointQueryResult,
    sample_point,
    sample_point_crs,
};

// ============================================================================
// XYZ Tile Extraction
// ============================================================================

pub use xyz_tile::{
    TileData,
    TileExtractor,
    BoundingBox,
    CoordTransformer,
    extract_xyz_tile,
    extract_tile_with_extent,
};

// ============================================================================
// Geometry & Projections
// ============================================================================

pub use geometry::Point;
pub use geometry::projection::{
    project_point,
    lon_lat_to_mercator,
    mercator_to_lon_lat,
    get_proj_string,
    is_geographic_crs,
};

// ============================================================================
// Range Readers (I/O Abstraction)
// ============================================================================

pub use range_reader::{
    RangeReader,
    LocalRangeReader,
    HttpRangeReader,
    MemoryRangeReader,
    create_range_reader,
};

// ============================================================================
// S3 Support
// ============================================================================

pub use s3::{
    S3Config,
    S3RangeReaderAsync,
    S3RangeReaderSync,
};

// ============================================================================
// Source Discovery
// ============================================================================

pub use source::{
    CogSource,
    CogEntry,
    CogLocation,
    LocalCogSource,
    LocalScanOptions,
    LocalSourceStats,
    S3CogSource,
    S3ScanOptions,
    S3SourceStats,
};

// ============================================================================
// Caching
// ============================================================================

pub use tile_cache::TileCache;

// ============================================================================
// Raster Abstraction
// ============================================================================

pub use raster::RasterSource;
