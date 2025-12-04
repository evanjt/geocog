use lru::LruCache;
use std::cmp::max;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

const CACHE_CAPACITY_BYTES: usize = 512 * 1024 * 1024; // 512 MB upper bound

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TileKind {
    Chunked,
    Lzw,
}

/// Key for cached decompressed tiles
/// Includes source identifier, tile index, and optional overview index
#[derive(Clone, Eq, PartialEq)]
struct TileKey {
    /// Source identifier (file path or URL)
    source: Arc<str>,
    /// Tile index within the IFD
    tile_index: u32,
    /// Overview index (None = full resolution, Some(n) = overview n)
    overview_idx: Option<u16>,
}

impl Hash for TileKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.source.hash(state);
        self.tile_index.hash(state);
        self.overview_idx.hash(state);
    }
}

impl TileKey {
    fn new(source: &str, tile_index: usize, overview_idx: Option<usize>) -> Self {
        TileKey {
            source: Arc::from(source),
            tile_index: tile_index as u32,
            overview_idx: overview_idx.map(|i| i as u16),
        }
    }
}

struct CacheEntry {
    data: Arc<Vec<f32>>,
    size_bytes: usize,
}

pub struct TileCache {
    current_bytes: usize,
    capacity_bytes: usize,
    entries: LruCache<TileKey, CacheEntry>,
}

impl TileCache {
    fn new(capacity_bytes: usize) -> Self {
        TileCache {
            current_bytes: 0,
            capacity_bytes,
            entries: LruCache::unbounded(),
        }
    }

    fn get(&mut self, key: &TileKey) -> Option<Arc<Vec<f32>>> {
        self.entries.get(key).map(|entry| Arc::clone(&entry.data))
    }

    fn contains(&mut self, key: &TileKey) -> bool {
        self.entries.contains(key)
    }

    fn insert(&mut self, key: TileKey, data: Arc<Vec<f32>>, size_bytes: usize) {
        if size_bytes > self.capacity_bytes {
            return;
        }

        if let Some(old) = self.entries.pop(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(old.size_bytes);
        }

        while self.current_bytes + size_bytes > self.capacity_bytes {
            if let Some((_key, entry)) = self.entries.pop_lru() {
                self.current_bytes = self.current_bytes.saturating_sub(entry.size_bytes);
            } else {
                break;
            }
        }

        self.current_bytes = self.current_bytes.saturating_add(size_bytes);
        self.entries.put(key, CacheEntry { data, size_bytes });
    }
}

static TILE_CACHE: std::sync::LazyLock<Mutex<TileCache>> = std::sync::LazyLock::new(|| {
    let cap = max(CACHE_CAPACITY_BYTES, 64 * 1024 * 1024); // never below 64MB
    Mutex::new(TileCache::new(cap))
});

fn make_key(source: &str, tile_index: usize, overview_idx: Option<usize>) -> TileKey {
    TileKey::new(source, tile_index, overview_idx)
}

/// Get a cached tile by source identifier, tile index, and optional overview index
/// - `source`: File path or URL identifying the COG
/// - `tile_index`: Tile index within the IFD
/// - `overview_idx`: None for full resolution, Some(n) for overview n
pub fn get(source: &str, tile_index: usize, overview_idx: Option<usize>) -> Option<Arc<Vec<f32>>> {
    let key = make_key(source, tile_index, overview_idx);
    TILE_CACHE.lock().unwrap().get(&key)
}

/// Check if a tile is cached
pub fn contains(source: &str, tile_index: usize, overview_idx: Option<usize>) -> bool {
    let key = make_key(source, tile_index, overview_idx);
    TILE_CACHE.lock().unwrap().contains(&key)
}

/// Insert a decompressed tile into the cache
pub fn insert(source: &str, tile_index: usize, overview_idx: Option<usize>, data: Arc<Vec<f32>>) {
    let size_bytes = data.len() * std::mem::size_of::<f32>();
    let key = make_key(source, tile_index, overview_idx);
    TILE_CACHE.lock().unwrap().insert(key, data, size_bytes);
}

// ============================================================================
// Legacy API for backward compatibility with tiff_chunked.rs and lzw_fallback.rs
// These will be phased out once those modules are updated
// ============================================================================

use std::path::Path;

/// Legacy get function for backward compatibility
pub fn get_legacy(path: &Path, _kind: TileKind, index: usize) -> Option<Arc<Vec<f32>>> {
    let source = path.to_string_lossy();
    get(&source, index, None)
}

/// Legacy contains function for backward compatibility
pub fn contains_legacy(path: &Path, _kind: TileKind, index: usize) -> bool {
    let source = path.to_string_lossy();
    contains(&source, index, None)
}

/// Legacy insert function for backward compatibility
pub fn insert_legacy(path: &Path, _kind: TileKind, index: usize, data: Arc<Vec<f32>>) {
    let source = path.to_string_lossy();
    insert(&source, index, None, data);
}
