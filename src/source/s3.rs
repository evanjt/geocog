//! S3 COG source.
//!
//! Scans S3 buckets for GeoTIFF files and extracts metadata.

use std::collections::HashMap;
use std::sync::Arc;

use futures::TryStreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use tracing::{debug, warn};

use crate::cog_reader::CogReader;
use crate::s3::S3RangeReaderSync;

use super::{CogEntry, CogLocation, CogSource};

/// Options for scanning S3 buckets.
#[derive(Debug, Clone)]
pub struct S3ScanOptions {
    /// Prefix to filter objects (e.g., "data/layers/")
    pub prefix: Option<String>,
    /// File extensions to consider as GeoTIFFs (case-insensitive)
    pub extensions: Vec<String>,
    /// Maximum number of objects to scan
    pub max_objects: Option<usize>,
    /// Custom endpoint URL (for MinIO, LocalStack, etc.)
    pub endpoint_url: Option<String>,
    /// AWS region
    pub region: Option<String>,
    /// Allow HTTP connections
    pub allow_http: bool,
}

impl Default for S3ScanOptions {
    fn default() -> Self {
        Self {
            prefix: None,
            extensions: vec![
                "tif".to_string(),
                "tiff".to_string(),
                "geotiff".to_string(),
                "geotif".to_string(),
            ],
            max_objects: None,
            endpoint_url: std::env::var("AWS_ENDPOINT_URL").ok(),
            region: std::env::var("AWS_REGION").ok().or(Some("us-east-1".to_string())),
            allow_http: std::env::var("AWS_ALLOW_HTTP")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(false),
        }
    }
}

impl S3ScanOptions {
    /// Create options with a specific prefix
    pub fn with_prefix(prefix: &str) -> Self {
        Self {
            prefix: Some(prefix.to_string()),
            ..Default::default()
        }
    }

    /// Set the endpoint URL (for MinIO, etc.)
    pub fn with_endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint_url = Some(endpoint.to_string());
        self
    }

    /// Set the region
    pub fn with_region(mut self, region: &str) -> Self {
        self.region = Some(region.to_string());
        self
    }

    /// Allow HTTP connections
    pub fn with_allow_http(mut self, allow: bool) -> Self {
        self.allow_http = allow;
        self
    }

    /// Set maximum number of objects to scan
    pub fn with_max_objects(mut self, max: usize) -> Self {
        self.max_objects = Some(max);
        self
    }
}

/// COG source that scans S3 buckets.
///
/// Discovers GeoTIFF files in an S3 bucket and extracts metadata from each.
///
/// # Example
///
/// ```rust,ignore
/// use geocog::source::{S3CogSource, S3ScanOptions};
///
/// // Scan with default options
/// let source = S3CogSource::scan("my-bucket", S3ScanOptions::default()).await?;
///
/// // Or scan with a prefix
/// let source = S3CogSource::scan("my-bucket", S3ScanOptions::with_prefix("data/cogs/")).await?;
///
/// for entry in source.entries() {
///     println!("Found: {} ({} bands)", entry.name, entry.bands);
/// }
/// ```
pub struct S3CogSource {
    bucket: String,
    entries: Vec<CogEntry>,
    entries_by_name: HashMap<String, usize>,
}

impl S3CogSource {
    /// Scan an S3 bucket for COG files.
    ///
    /// # Arguments
    /// * `bucket` - S3 bucket name
    /// * `options` - Scanning options (prefix, extensions, etc.)
    ///
    /// # Returns
    /// A `S3CogSource` containing metadata for all discovered COGs.
    pub async fn scan(
        bucket: &str,
        options: S3ScanOptions,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Build the S3 client
        let mut builder = AmazonS3Builder::new()
            .with_bucket_name(bucket);

        if let Some(region) = &options.region {
            builder = builder.with_region(region);
        }

        if let Some(endpoint) = &options.endpoint_url {
            builder = builder.with_endpoint(endpoint);
        }

        if let Some(access_key) = std::env::var("AWS_ACCESS_KEY_ID").ok() {
            builder = builder.with_access_key_id(&access_key);
        }

        if let Some(secret_key) = std::env::var("AWS_SECRET_ACCESS_KEY").ok() {
            builder = builder.with_secret_access_key(&secret_key);
        }

        if options.allow_http {
            builder = builder.with_allow_http(true);
        }

        let store = builder.build()?;

        // List objects in the bucket
        let prefix = options.prefix.as_deref().map(ObjectPath::from);
        let list_stream = store.list(prefix.as_ref());

        let mut entries = Vec::new();
        let mut entries_by_name = HashMap::new();
        let mut count = 0;

        let mut items: Vec<_> = list_stream.try_collect().await?;

        // Sort by key for deterministic ordering
        items.sort_by(|a, b| a.location.as_ref().cmp(b.location.as_ref()));

        for meta in items {
            // Check max objects limit
            if let Some(max) = options.max_objects {
                if count >= max {
                    break;
                }
            }

            let key = meta.location.as_ref();

            // Check extension
            let is_geotiff = key
                .rsplit('.')
                .next()
                .map(|ext| options.extensions.iter().any(|x| x.eq_ignore_ascii_case(ext)))
                .unwrap_or(false);

            if !is_geotiff {
                continue;
            }

            // Try to read COG metadata
            let s3_url = format!("s3://{}/{}", bucket, key);
            match Self::read_cog_entry(&s3_url, &meta).await {
                Ok(cog_entry) => {
                    let idx = entries.len();
                    entries_by_name.insert(cog_entry.name.clone(), idx);
                    entries.push(cog_entry);
                    debug!(key = %key, "Discovered S3 COG");
                    count += 1;
                }
                Err(e) => {
                    warn!(key = %key, error = %e, "Failed to read S3 COG metadata");
                }
            }
        }

        Ok(Self {
            bucket: bucket.to_string(),
            entries,
            entries_by_name,
        })
    }

    /// Read metadata from a single S3 COG file.
    async fn read_cog_entry(
        s3_url: &str,
        meta: &object_store::ObjectMeta,
    ) -> Result<CogEntry, Box<dyn std::error::Error + Send + Sync>> {
        // Extract name from key
        let key = meta.location.as_ref();
        let name = key
            .rsplit('/')
            .next()
            .and_then(|f| f.rsplit('.').last())
            .unwrap_or("unknown")
            .to_string();

        // Parse bucket and key from URL
        let parsed = url::Url::parse(s3_url)?;
        let bucket = parsed.host_str().ok_or("Missing bucket")?.to_string();
        let key_str = parsed.path().trim_start_matches('/').to_string();

        // Open and read COG using sync reader (we're in an async context but CogReader is sync)
        let reader = S3RangeReaderSync::new(s3_url)?;
        let cog_reader = CogReader::from_reader(Arc::new(reader))?;

        let size_bytes = Some(meta.size as u64);
        let last_modified = Some(meta.last_modified.into());

        CogEntry::from_reader(
            &cog_reader,
            name,
            CogLocation::S3 { bucket, key: key_str },
            size_bytes,
            last_modified,
        )
        .map_err(|e| e.into())
    }

    /// Get the bucket name
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// Get statistics about discovered COGs.
    pub fn stats(&self) -> S3SourceStats {
        let total_size: u64 = self.entries.iter().filter_map(|e| e.size_bytes).sum();
        let tiled_count = self.entries.iter().filter(|e| e.is_tiled).count();
        let with_overviews = self.entries.iter().filter(|e| e.has_overviews).count();

        S3SourceStats {
            file_count: self.entries.len(),
            total_size_bytes: total_size,
            tiled_count,
            with_overviews,
        }
    }
}

impl CogSource for S3CogSource {
    fn entries(&self) -> &[CogEntry] {
        &self.entries
    }

    fn get(&self, name: &str) -> Option<&CogEntry> {
        self.entries_by_name.get(name).map(|&idx| &self.entries[idx])
    }
}

/// Statistics about an S3 COG source.
#[derive(Debug, Clone)]
pub struct S3SourceStats {
    /// Number of COG files discovered
    pub file_count: usize,
    /// Total size of all files in bytes
    pub total_size_bytes: u64,
    /// Number of properly tiled COGs
    pub tiled_count: usize,
    /// Number of COGs with overviews
    pub with_overviews: usize,
}

impl S3SourceStats {
    /// Get total size in megabytes
    pub fn total_size_mb(&self) -> f64 {
        self.total_size_bytes as f64 / 1024.0 / 1024.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = S3ScanOptions::default();
        assert!(opts.prefix.is_none());
        assert!(opts.extensions.contains(&"tif".to_string()));
    }

    #[test]
    fn test_with_prefix() {
        let opts = S3ScanOptions::with_prefix("data/layers/");
        assert_eq!(opts.prefix, Some("data/layers/".to_string()));
    }

    #[test]
    fn test_options_builder() {
        let opts = S3ScanOptions::default()
            .with_endpoint("http://localhost:9000")
            .with_region("eu-west-1")
            .with_allow_http(true)
            .with_max_objects(100);

        assert_eq!(opts.endpoint_url, Some("http://localhost:9000".to_string()));
        assert_eq!(opts.region, Some("eu-west-1".to_string()));
        assert!(opts.allow_http);
        assert_eq!(opts.max_objects, Some(100));
    }
}
