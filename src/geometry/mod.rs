pub mod projection;

/// A simple 2D point with x and y coordinates.
///
/// This struct is useful for representing geographic coordinates,
/// pixel coordinates, or any 2D point in various operations.
///
/// # Example
///
/// ```rust
/// use cogrs::Point;
///
/// // Create a point from coordinates
/// let p = Point::new(-122.4, 37.8);
/// assert_eq!(p.x, -122.4);
/// assert_eq!(p.y, 37.8);
///
/// // Create from tuple
/// let p2: Point = (-122.4, 37.8).into();
/// assert_eq!(p, p2);
///
/// // Use lonlat constructor for geographic coordinates
/// let sf = Point::lonlat(-122.4, 37.8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// X coordinate (longitude for geographic points)
    pub x: f64,
    /// Y coordinate (latitude for geographic points)
    pub y: f64,
}

impl Point {
    /// Create a new point from x and y coordinates.
    #[inline]
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Create a point from longitude and latitude (EPSG:4326).
    ///
    /// This is a semantic alias for `new()` that makes it clear
    /// the coordinates represent geographic lon/lat.
    #[inline]
    #[must_use]
    pub fn lonlat(lon: f64, lat: f64) -> Self {
        Self { x: lon, y: lat }
    }

    /// Create a point at the origin (0, 0).
    #[inline]
    #[must_use]
    pub fn origin() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Get longitude (alias for x).
    #[inline]
    #[must_use]
    pub fn lon(&self) -> f64 {
        self.x
    }

    /// Get latitude (alias for y).
    #[inline]
    #[must_use]
    pub fn lat(&self) -> f64 {
        self.y
    }

    /// Convert to a tuple (x, y).
    #[inline]
    #[must_use]
    pub fn to_tuple(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    /// Calculate Euclidean distance to another point.
    #[inline]
    #[must_use]
    pub fn distance_to(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl Default for Point {
    fn default() -> Self {
        Self::origin()
    }
}

impl From<(f64, f64)> for Point {
    #[inline]
    fn from((x, y): (f64, f64)) -> Self {
        Self::new(x, y)
    }
}

impl From<Point> for (f64, f64) {
    #[inline]
    fn from(p: Point) -> Self {
        (p.x, p.y)
    }
}

impl From<[f64; 2]> for Point {
    #[inline]
    fn from([x, y]: [f64; 2]) -> Self {
        Self::new(x, y)
    }
}

impl From<Point> for [f64; 2] {
    #[inline]
    fn from(p: Point) -> Self {
        [p.x, p.y]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_new() {
        let p = Point::new(10.0, 20.0);
        assert_eq!(p.x, 10.0);
        assert_eq!(p.y, 20.0);
    }

    #[test]
    fn test_point_lonlat() {
        let p = Point::lonlat(-122.4, 37.8);
        assert_eq!(p.lon(), -122.4);
        assert_eq!(p.lat(), 37.8);
    }

    #[test]
    fn test_point_origin() {
        let p = Point::origin();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
    }

    #[test]
    fn test_point_from_tuple() {
        let p: Point = (5.0, 10.0).into();
        assert_eq!(p.x, 5.0);
        assert_eq!(p.y, 10.0);
    }

    #[test]
    fn test_point_to_tuple() {
        let p = Point::new(5.0, 10.0);
        let (x, y) = p.to_tuple();
        assert_eq!(x, 5.0);
        assert_eq!(y, 10.0);
    }

    #[test]
    fn test_point_from_array() {
        let p: Point = [3.0, 4.0].into();
        assert_eq!(p.x, 3.0);
        assert_eq!(p.y, 4.0);
    }

    #[test]
    fn test_point_to_array() {
        let p = Point::new(3.0, 4.0);
        let arr: [f64; 2] = p.into();
        assert_eq!(arr, [3.0, 4.0]);
    }

    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_equality() {
        let p1 = Point::new(1.0, 2.0);
        let p2 = Point::new(1.0, 2.0);
        let p3 = Point::new(1.0, 3.0);
        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_point_default() {
        let p = Point::default();
        assert_eq!(p, Point::origin());
    }
}
