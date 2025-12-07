//! # cogrs - Pure Rust COG (Cloud Optimized GeoTIFF) Reader
//!
//! A library for reading Cloud Optimized GeoTIFFs without GDAL.
//!
//! ## Features
//!
//! - Pure Rust, no GDAL dependency
//! - Range requests for local files, HTTP, and S3
//! - Compression: DEFLATE, LZW, ZSTD, JPEG
//! - Coordinate transforms via proj4rs
//! - Point queries at geographic coordinates
//! - XYZ tile extraction with resampling options
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use cogrs::{CogReader, PointQuery, extract_xyz_tile};
//!
//! let reader = CogReader::open("path/to/file.tif")?;
//!
//! // Point query
//! let result = reader.sample_lonlat(-122.4, 37.8)?;
//!
//! // XYZ tile
//! let tile = extract_xyz_tile(&reader, 10, 163, 395, (256, 256))?;
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
    ResamplingMethod,
    extract_xyz_tile,
    extract_tile_with_extent,
    extract_tile_with_extent_resampled,
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
