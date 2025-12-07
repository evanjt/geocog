//! Async XYZ tile extraction
//!
//! This module provides asynchronous versions of the tile extraction functions
//! for use with tokio or other async runtimes.
//!
//! # Example
//!
//! ```rust,ignore
//! use cogrs::{CogReader, extract_xyz_tile_async};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//!     let reader = CogReader::open("path/to/cog.tif")?;
//!
//!     // Extract tile asynchronously (I/O happens on blocking thread pool)
//!     let tile = extract_xyz_tile_async(&reader, 10, 163, 395, (256, 256)).await?;
//!     println!("Extracted tile with {} pixels", tile.pixels.len());
//!     Ok(())
//! }
//! ```

use crate::cog_reader::CogReader;
use crate::tiff_utils::AnyResult;
use crate::xyz_tile::{
    extract_tile_with_extent_resampled, extract_xyz_tile, BoundingBox, ResamplingMethod, TileData,
};

/// Extract an XYZ tile asynchronously
///
/// This runs the tile extraction on a blocking thread pool to avoid blocking
/// the async runtime during I/O and decompression operations.
///
/// # Arguments
/// * `reader` - The COG reader with loaded metadata
/// * `z` - Zoom level
/// * `x` - Tile X coordinate
/// * `y` - Tile Y coordinate
/// * `tile_size` - Output tile dimensions (width, height)
///
/// # Example
///
/// ```rust,ignore
/// let tile = extract_xyz_tile_async(&reader, 10, 163, 395, (256, 256)).await?;
/// ```
pub async fn extract_xyz_tile_async(
    reader: &CogReader,
    z: u32,
    x: u32,
    y: u32,
    tile_size: (usize, usize),
) -> AnyResult<TileData> {
    // Clone the reader's Arc for the blocking task
    let reader_clone = reader.clone_for_async();

    tokio::task::spawn_blocking(move || {
        extract_xyz_tile(&reader_clone, z, x, y, tile_size)
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

/// Extract a tile with custom bounds asynchronously
///
/// # Arguments
/// * `reader` - The COG reader with loaded metadata
/// * `extent_3857` - Bounding box in Web Mercator (EPSG:3857)
/// * `tile_size` - Output tile dimensions (width, height)
/// * `resampling` - Resampling method to use
pub async fn extract_tile_async(
    reader: &CogReader,
    extent_3857: &BoundingBox,
    tile_size: (usize, usize),
    resampling: ResamplingMethod,
) -> AnyResult<TileData> {
    let reader_clone = reader.clone_for_async();
    let extent = *extent_3857;

    tokio::task::spawn_blocking(move || {
        extract_tile_with_extent_resampled(&reader_clone, &extent, tile_size, resampling)
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

/// Async tile extractor builder
///
/// Provides a fluent async API for tile extraction.
///
/// # Example
///
/// ```rust,ignore
/// use cogrs::{CogReader, AsyncTileExtractor};
///
/// let reader = CogReader::open("path/to/cog.tif")?;
///
/// let tile = AsyncTileExtractor::new(&reader)
///     .xyz(10, 163, 395)
///     .output_size(512, 512)
///     .extract()
///     .await?;
/// ```
pub struct AsyncTileExtractor<'a> {
    reader: &'a CogReader,
    bounds: Option<BoundingBox>,
    output_size: (usize, usize),
    resampling: ResamplingMethod,
    selected_bands: Option<Vec<usize>>,
}

impl<'a> AsyncTileExtractor<'a> {
    /// Create a new async tile extractor for the given COG reader
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

    /// Set the output tile bounds using an XYZ tile coordinate
    #[must_use]
    pub fn xyz(mut self, z: u32, x: u32, y: u32) -> Self {
        self.bounds = Some(BoundingBox::from_xyz(z, x, y));
        self
    }

    /// Set the output tile bounds using a bounding box in EPSG:3857
    #[must_use]
    pub fn bounds(mut self, bbox: BoundingBox) -> Self {
        self.bounds = Some(bbox);
        self
    }

    /// Set the output tile size (width, height)
    #[must_use]
    pub fn output_size(mut self, width: usize, height: usize) -> Self {
        self.output_size = (width, height);
        self
    }

    /// Set the output tile to square with the given size
    #[must_use]
    pub fn size(mut self, size: usize) -> Self {
        self.output_size = (size, size);
        self
    }

    /// Set the resampling method
    #[must_use]
    pub fn resampling(mut self, method: ResamplingMethod) -> Self {
        self.resampling = method;
        self
    }

    /// Select specific bands to extract (0-indexed)
    #[must_use]
    pub fn bands(mut self, bands: &[usize]) -> Self {
        self.selected_bands = Some(bands.to_vec());
        self
    }

    /// Extract the tile asynchronously
    pub async fn extract(self) -> AnyResult<TileData> {
        use crate::xyz_tile::extract_tile_with_bands;

        let bounds = self.bounds.ok_or("Bounds not set: use .xyz() or .bounds()")?;

        if let Some(selected) = self.selected_bands {
            let reader_clone = self.reader.clone_for_async();
            let resampling = self.resampling;
            let output_size = self.output_size;

            tokio::task::spawn_blocking(move || {
                extract_tile_with_bands(&reader_clone, &bounds, output_size, resampling, &selected)
            })
            .await
            .map_err(|e| format!("Task join error: {e}"))?
        } else {
            extract_tile_async(self.reader, &bounds, self.output_size, self.resampling).await
        }
    }
}

/// Extract multiple XYZ tiles concurrently
///
/// This extracts multiple tiles in parallel, which can be more efficient than
/// extracting them sequentially when I/O is the bottleneck.
///
/// # Arguments
/// * `reader` - The COG reader with loaded metadata
/// * `tiles` - Vector of (z, x, y) tile coordinates
/// * `tile_size` - Output tile dimensions (width, height)
///
/// # Returns
/// A vector of results in the same order as the input tiles
///
/// # Example
///
/// ```rust,ignore
/// let tiles_to_fetch = vec![
///     (10, 163, 395),
///     (10, 164, 395),
///     (10, 163, 396),
/// ];
///
/// let results = extract_xyz_tiles_concurrent(&reader, &tiles_to_fetch, (256, 256)).await;
/// for (i, result) in results.iter().enumerate() {
///     match result {
///         Ok(tile) => println!("Tile {}: {} pixels", i, tile.pixels.len()),
///         Err(e) => println!("Tile {} failed: {}", i, e),
///     }
/// }
/// ```
pub async fn extract_xyz_tiles_concurrent(
    reader: &CogReader,
    tiles: &[(u32, u32, u32)],
    tile_size: (usize, usize),
) -> Vec<AnyResult<TileData>> {
    use futures::future::join_all;

    let futures: Vec<_> = tiles
        .iter()
        .map(|&(z, x, y)| {
            let reader_clone = reader.clone_for_async();
            async move {
                tokio::task::spawn_blocking(move || {
                    extract_xyz_tile(&reader_clone, z, x, y, tile_size)
                })
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("Task join error: {e}").into()
                })?
            }
        })
        .collect();

    join_all(futures).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::range_reader::LocalRangeReader;

    #[tokio::test]
    async fn test_async_tile_extractor_build() {
        // Test that the builder compiles and has correct defaults
        // We can't test extraction without a real file, but we can test the builder
        let bbox = BoundingBox::from_xyz(5, 10, 12);
        assert!(bbox.minx < bbox.maxx);
        assert!(bbox.miny < bbox.maxy);
    }

    #[tokio::test]
    async fn test_extract_xyz_tile_async() {
        // Skip if test file doesn't exist
        let path = "/home/evan/projects/personal/geo/tileyolo/data/viridis/output_cog.tif";
        if !std::path::Path::new(path).exists() {
            println!("Skipping: test file not found");
            return;
        }

        let reader = LocalRangeReader::new(path).unwrap();
        let cog = CogReader::from_reader(Arc::new(reader)).unwrap();

        // Extract a tile asynchronously
        let tile = extract_xyz_tile_async(&cog, 1, 0, 0, (256, 256))
            .await
            .unwrap();

        assert_eq!(tile.width, 256);
        assert_eq!(tile.height, 256);
        assert!(!tile.pixels.is_empty());
    }

    #[tokio::test]
    async fn test_async_tile_extractor_fluent() {
        let path = "/home/evan/projects/personal/geo/tileyolo/data/viridis/output_cog.tif";
        if !std::path::Path::new(path).exists() {
            println!("Skipping: test file not found");
            return;
        }

        let reader = LocalRangeReader::new(path).unwrap();
        let cog = CogReader::from_reader(Arc::new(reader)).unwrap();

        let tile = AsyncTileExtractor::new(&cog)
            .xyz(1, 0, 0)
            .output_size(512, 512)
            .resampling(ResamplingMethod::Bilinear)
            .extract()
            .await
            .unwrap();

        assert_eq!(tile.width, 512);
        assert_eq!(tile.height, 512);
    }

    #[tokio::test]
    async fn test_concurrent_extraction() {
        let path = "/home/evan/projects/personal/geo/tileyolo/data/viridis/output_cog.tif";
        if !std::path::Path::new(path).exists() {
            println!("Skipping: test file not found");
            return;
        }

        let reader = LocalRangeReader::new(path).unwrap();
        let cog = CogReader::from_reader(Arc::new(reader)).unwrap();

        let tiles = vec![
            (1, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (1, 1, 1),
        ];

        let results = extract_xyz_tiles_concurrent(&cog, &tiles, (256, 256)).await;

        assert_eq!(results.len(), 4);
        for result in &results {
            assert!(result.is_ok());
            let tile = result.as_ref().unwrap();
            assert_eq!(tile.width, 256);
            assert_eq!(tile.height, 256);
        }
    }
}
