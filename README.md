# geocog

Pure Rust COG (Cloud Optimized GeoTIFF) reader library.

## Usage

```rust
use geocog::CogReader;

let reader = CogReader::open("path/to/file.tif")?;
```

## License

MIT
