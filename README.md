# cogrs

Pure Rust COG (Cloud Optimized GeoTIFF) reader library for building tile servers, GIS applications, and geospatial data pipelines.

[![Crates.io](https://img.shields.io/crates/v/cogrs.svg)](https://crates.io/crates/cogrs)
[![Documentation](https://docs.rs/cogrs/badge.svg)](https://docs.rs/cogrs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Zero GDAL dependency** - Pure Rust implementation, no system libraries needed
- **Range requests** - Efficient partial reads from local files, HTTP URLs, or S3
- **Streaming** - Never loads entire file into memory
- **Compression** - DEFLATE, LZW, ZSTD, JPEG, and uncompressed
- **Overviews** - Automatic pyramid level selection for optimal performance
- **Coordinate transforms** - Pure Rust via proj4rs (thousands of EPSG codes)
- **Point queries** - Sample values at geographic coordinates
- **XYZ tiles** - Extract map tiles with automatic reprojection
- **Resampling** - Nearest neighbor, bilinear, and bicubic interpolation
- **Caching** - Built-in LRU tile cache (configurable, default 512MB)

## Quick Start

```rust
use cogrs::{CogReader, PointQuery, extract_xyz_tile};

// Open a COG from local file, HTTP, or S3
let reader = CogReader::open("path/to/file.tif")?;

// Sample pixel values at a geographic coordinate
let result = reader.sample_lonlat(-122.4, 37.8)?;
for (band, value) in &result.values {
    println!("Band {}: {}", band, value);
}

// Extract an XYZ map tile
let tile = extract_xyz_tile(&reader, 10, 163, 395, (256, 256))?;
println!("Tile has {} bands, {} pixels", tile.bands, tile.pixels.len());
```

## XYZ Tile Extraction

Use the `TileExtractor` builder for full control:

```rust
use cogrs::{CogReader, TileExtractor, ResamplingMethod};

let reader = CogReader::open("satellite.tif")?;

// Extract with bilinear resampling for smoother results
let tile = TileExtractor::new(&reader)
    .xyz(10, 163, 395)
    .output_size(512, 512)
    .resampling(ResamplingMethod::Bilinear)
    .extract()?;
```

## Point Queries

Sample pixel values at any coordinate:

```rust
use cogrs::{CogReader, PointQuery};

let reader = CogReader::open("dem.tif")?;

// Query at lon/lat (EPSG:4326)
let result = reader.sample_lonlat(-122.4, 37.8)?;
println!("Elevation: {:?}", result.get(0));

// Query at coordinates in any CRS (e.g., UTM zone 10N)
let result = reader.sample_crs(32610, 551000.0, 4185000.0)?;

// Batch queries for efficiency
let points = vec![(-122.4, 37.8), (-122.5, 37.9), (-122.3, 37.7)];
let results = reader.sample_points_lonlat(&points)?;
```

## Coordinate Transforms

Transform between any supported coordinate systems:

```rust
use cogrs::{CoordTransformer, project_point};

// One-off transform
let (x, y) = project_point(4326, 3857, -122.4, 37.8)?;

// Reusable transformer for efficiency
let transformer = CoordTransformer::new(4326, 32633)?; // WGS84 -> UTM 33N
let (utm_x, utm_y) = transformer.transform(15.0, 52.0)?;

// Batch transform
let points = vec![(0.0, 0.0), (10.0, 51.5), (-122.4, 37.8)];
let results = transformer.transform_batch(&points);
```

## HTTP and S3 Support

Read COGs directly from URLs or S3:

```rust
use cogrs::{CogReader, HttpRangeReader, S3RangeReaderSync};
use std::sync::Arc;

// HTTP URL
let reader = HttpRangeReader::new("https://example.com/cog.tif")?;
let cog = CogReader::from_reader(Arc::new(reader))?;

// S3 (uses AWS_* environment variables for credentials)
let reader = S3RangeReaderSync::new("s3://bucket/path/to/cog.tif")?;
let cog = CogReader::from_reader(Arc::new(reader))?;
```

## Supported Formats

| Feature | Support |
|---------|---------|
| TIFF/BigTIFF | Yes |
| COG structure | Yes (tiled, with overviews) |
| Strip TIFF | Yes (converted internally) |
| Compression | None, DEFLATE, LZW, ZSTD, JPEG |
| Sample types | UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, Float64 |
| Predictor | Horizontal differencing (predictor=2) |
| Photometric | MinIsBlack, MinIsWhite, RGB, Palette |

## Performance

- **Metadata parsing**: ~5-20ms (reads only IFD headers)
- **Tile extraction (256x256)**: ~1.5ms with overview
- **Point query**: ~200µs (single band)
- **Memory**: Only decompressed tiles are cached, not raw file data

## License

MIT
