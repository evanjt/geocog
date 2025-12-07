# cogrs

Pure Rust COG (Cloud Optimized GeoTIFF) reader library.

## Features

- Pure Rust, no GDAL dependency
- Range requests for local files, HTTP, and S3
- Compression: DEFLATE, LZW, ZSTD, JPEG, WebP
- Coordinate transforms via proj4rs
- Point queries at geographic coordinates
- XYZ tile extraction with resampling options

## Usage

```rust
use cogrs::{CogReader, PointQuery, extract_xyz_tile};

let reader = CogReader::open("path/to/file.tif")?;

// Point query
let result = reader.sample_lonlat(-122.4, 37.8)?;

// XYZ tile
let tile = extract_xyz_tile(&reader, 10, 163, 395, (256, 256))?;
```

## License

MIT
