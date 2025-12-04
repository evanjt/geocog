//! # geocog - Pure Rust COG (Cloud Optimized GeoTIFF) Reader
//!
//! A high-performance library for reading Cloud Optimized GeoTIFFs without GDAL.
//!
//! ## Features
//!
//! - **Metadata-only initialization**: Reads only IFD headers (~16KB) on open
//! - **Range requests**: Efficient partial reads from local files, HTTP, or S3
//! - **Streaming**: Never loads entire file into memory
//! - **Compression**: Supports DEFLATE, LZW, ZSTD, and uncompressed
//! - **Overviews**: Automatic pyramid level selection with data density validation
//! - **Coordinate transforms**: Built-in proj support for CRS transformations
//!
//! ## Example
//!
//! ```rust,ignore
//! use geocog::CogReader;
//!
//! let reader = CogReader::open("path/to/file.tif")?;
//! let tile_data = reader.read_tile(10, 512, 384)?;
//! ```
//!
//! ## Architecture
//!
//! The library is organized into several modules:
//!
//! - [`cog_reader`]: Core COG metadata parsing and tile reading
//! - [`cog`]: High-level processing pipeline with caching
//! - [`range_reader`]: Abstraction for reading byte ranges from various sources
//! - [`s3`]: S3-compatible storage backend
//! - [`tile_cache`]: LRU cache for decompressed tiles
//! - [`raster`]: Raster data abstraction trait

pub mod cog_reader;
pub mod geometry;
pub mod lzw_fallback;
pub mod range_reader;
pub mod raster;
pub mod s3;
pub mod source;
pub mod tiff_chunked;
pub mod tiff_utils;
pub mod tile_cache;
pub mod xyz_tile;

// Re-export main types
pub use cog_reader::{CogReader, CogMetadata, CogDataType, Compression, GeoTransform, OverviewMetadata};
pub use geometry::projection::{project_point, lon_lat_to_mercator, mercator_to_lon_lat, get_proj_string, is_geographic_crs};
pub use range_reader::RangeReader;
pub use raster::RasterSource;
pub use s3::{S3Config, S3RangeReaderAsync, S3RangeReaderSync};
pub use tile_cache::TileCache;
pub use source::{CogEntry, CogLocation, CogSource, LocalCogSource, LocalScanOptions, LocalSourceStats, S3CogSource, S3ScanOptions, S3SourceStats};
pub use xyz_tile::{TileData, BoundingBox, extract_xyz_tile, extract_tile_with_extent};
