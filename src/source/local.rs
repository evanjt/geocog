//! Local filesystem COG source.
//!
//! Scans directories for GeoTIFF files and extracts metadata.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use tracing::{debug, warn};
use walkdir::WalkDir;

use crate::cog_reader::CogReader;
use crate::range_reader::LocalRangeReader;

use super::{CogEntry, CogLocation, CogSource};

/// Options for scanning local directories.
#[derive(Debug, Clone)]
pub struct LocalScanOptions {
    /// Minimum directory depth to scan (0 = include root files)
    pub min_depth: usize,
    /// Maximum directory depth to scan (None = unlimited)
    pub max_depth: Option<usize>,
    /// File extensions to consider as GeoTIFFs (case-insensitive)
    pub extensions: Vec<String>,
    /// Whether to follow symbolic links
    pub follow_links: bool,
}

impl Default for LocalScanOptions {
    fn default() -> Self {
        Self {
            min_depth: 0,
            max_depth: None,
            extensions: vec![
                "tif".to_string(),
                "tiff".to_string(),
                "geotiff".to_string(),
                "geotif".to_string(),
            ],
            follow_links: false,
        }
    }
}

impl LocalScanOptions {
    /// Create options that only scan immediate subdirectories (like tileyolo's layer structure)
    pub fn layers() -> Self {
        Self {
            min_depth: 2, // layer/file.tif pattern
            ..Default::default()
        }
    }

    /// Create options for recursive scanning with no depth limit
    pub fn recursive() -> Self {
        Self::default()
    }

    /// Set minimum depth
    pub fn with_min_depth(mut self, depth: usize) -> Self {
        self.min_depth = depth;
        self
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }
}

/// COG source that scans local directories.
///
/// Discovers GeoTIFF files in a directory tree and extracts metadata from each.
///
/// # Example
///
/// ```rust,ignore
/// use geocog::source::{LocalCogSource, LocalScanOptions};
///
/// // Scan with default options (recursive)
/// let source = LocalCogSource::scan("/path/to/data", LocalScanOptions::default())?;
///
/// // Or scan layer-style directories (depth=2)
/// let source = LocalCogSource::scan("/path/to/layers", LocalScanOptions::layers())?;
///
/// for entry in source.entries() {
///     println!("Found: {} ({} bands)", entry.name, entry.bands);
/// }
/// ```
pub struct LocalCogSource {
    entries: Vec<CogEntry>,
    entries_by_name: HashMap<String, usize>,
}

impl LocalCogSource {
    /// Scan a directory for COG files.
    ///
    /// # Arguments
    /// * `root` - Root directory to scan
    /// * `options` - Scanning options (depth, extensions, etc.)
    ///
    /// # Returns
    /// A `LocalCogSource` containing metadata for all discovered COGs.
    pub fn scan<P: AsRef<Path>>(
        root: P,
        options: LocalScanOptions,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let root = root.as_ref();
        if !root.exists() {
            return Err(format!("Directory does not exist: {}", root.display()).into());
        }

        let mut entries = Vec::new();
        let mut entries_by_name = HashMap::new();

        // Build walker with options
        let mut walker = WalkDir::new(root)
            .min_depth(options.min_depth)
            .follow_links(options.follow_links);

        if let Some(max) = options.max_depth {
            walker = walker.max_depth(max);
        }

        // Scan for files
        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();

            // Check extension
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_lowercase());

            let is_geotiff = ext
                .as_ref()
                .map(|e| options.extensions.iter().any(|x| x.eq_ignore_ascii_case(e)))
                .unwrap_or(false);

            if !is_geotiff {
                continue;
            }

            // Try to read COG metadata
            match Self::read_cog_entry(path) {
                Ok(cog_entry) => {
                    let idx = entries.len();
                    entries_by_name.insert(cog_entry.name.clone(), idx);
                    entries.push(cog_entry);
                    debug!(path = %path.display(), "Discovered COG");
                }
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "Failed to read COG metadata");
                }
            }
        }

        Ok(Self {
            entries,
            entries_by_name,
        })
    }

    /// Read metadata from a single COG file.
    fn read_cog_entry(path: &Path) -> Result<CogEntry, Box<dyn std::error::Error + Send + Sync>> {
        // Get file metadata
        let file_meta = std::fs::metadata(path)?;
        let size_bytes = file_meta.len();
        let last_modified = file_meta.modified().ok();

        // Derive name from filename
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Open and read COG
        let reader = LocalRangeReader::new(path)?;
        let cog_reader = CogReader::from_reader(Arc::new(reader))?;

        CogEntry::from_reader(
            &cog_reader,
            name,
            CogLocation::Local(path.to_path_buf()),
            Some(size_bytes),
            last_modified,
        )
        .map_err(|e| e.into())
    }

    /// Get the root directory that was scanned.
    pub fn root(&self) -> Option<&Path> {
        self.entries.first().and_then(|e| match &e.location {
            CogLocation::Local(p) => p.parent(),
            _ => None,
        })
    }

    /// Get statistics about discovered COGs.
    pub fn stats(&self) -> LocalSourceStats {
        let total_size: u64 = self.entries.iter().filter_map(|e| e.size_bytes).sum();
        let tiled_count = self.entries.iter().filter(|e| e.is_tiled).count();
        let with_overviews = self.entries.iter().filter(|e| e.has_overviews).count();

        LocalSourceStats {
            file_count: self.entries.len(),
            total_size_bytes: total_size,
            tiled_count,
            with_overviews,
        }
    }
}

impl CogSource for LocalCogSource {
    fn entries(&self) -> &[CogEntry] {
        &self.entries
    }

    fn get(&self, name: &str) -> Option<&CogEntry> {
        self.entries_by_name.get(name).map(|&idx| &self.entries[idx])
    }
}

/// Statistics about a local COG source.
#[derive(Debug, Clone)]
pub struct LocalSourceStats {
    /// Number of COG files discovered
    pub file_count: usize,
    /// Total size of all files in bytes
    pub total_size_bytes: u64,
    /// Number of properly tiled COGs
    pub tiled_count: usize,
    /// Number of COGs with overviews
    pub with_overviews: usize,
}

impl LocalSourceStats {
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
        let opts = LocalScanOptions::default();
        assert_eq!(opts.min_depth, 0);
        assert!(opts.max_depth.is_none());
        assert!(opts.extensions.contains(&"tif".to_string()));
    }

    #[test]
    fn test_layers_options() {
        let opts = LocalScanOptions::layers();
        assert_eq!(opts.min_depth, 2);
    }

    #[test]
    fn test_scan_nonexistent_dir() {
        let result = LocalCogSource::scan("/nonexistent/path", LocalScanOptions::default());
        assert!(result.is_err());
    }
}
