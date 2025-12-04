/// Project a point from one CRS to another using pure Rust (proj4rs + crs-definitions).
///
/// This function handles coordinate transformations between any EPSG codes supported
/// by the crs-definitions database (~thousands of codes including UTM zones, national grids, etc).
///
/// # Arguments
/// * `source_epsg` - Source CRS EPSG code
/// * `target_epsg` - Target CRS EPSG code
/// * `x` - X coordinate in source CRS
/// * `y` - Y coordinate in source CRS
///
/// # Returns
/// Tuple of (x, y) in target CRS, or an error if the EPSG code is not supported.
pub fn project_point(source_epsg: i32, target_epsg: i32, x: f64, y: f64) -> Result<(f64, f64), String> {
    // No-op if same CRS
    if source_epsg == target_epsg {
        return Ok((x, y));
    }

    project_with_proj4rs(source_epsg, target_epsg, x, y)
}

/// Convenience function: longitude/latitude (EPSG:4326) to Web Mercator (EPSG:3857)
pub fn lon_lat_to_mercator(lon: f64, lat: f64) -> (f64, f64) {
    project_point(4326, 3857, lon, lat).unwrap_or((lon, lat))
}

/// Convenience function: Web Mercator (EPSG:3857) to longitude/latitude (EPSG:4326)
pub fn mercator_to_lon_lat(x: f64, y: f64) -> (f64, f64) {
    project_point(3857, 4326, x, y).unwrap_or((x, y))
}

/// Get PROJ4 string for an EPSG code using the crs-definitions database
pub fn get_proj_string(epsg: i32) -> Option<&'static str> {
    u16::try_from(epsg).ok()
        .and_then(crs_definitions::from_code)
        .map(|def| def.proj4)
}

/// Check if an EPSG code represents a geographic (lon/lat) CRS
pub fn is_geographic_crs(epsg: i32) -> bool {
    // Geographic CRS codes are typically in the 4000-4999 range
    // but we check the proj string to be sure
    if let Some(proj_str) = get_proj_string(epsg) {
        proj_str.contains("+proj=longlat")
    } else {
        // Fallback: assume 4326 and similar are geographic
        epsg == 4326 || (epsg >= 4000 && epsg < 5000)
    }
}

/// Project using proj4rs with EPSG codes from crs-definitions
fn project_with_proj4rs(source_epsg: i32, target_epsg: i32, x: f64, y: f64) -> Result<(f64, f64), String> {
    use proj4rs::proj::Proj;
    use proj4rs::transform::transform;

    let source_str = get_proj_string(source_epsg)
        .ok_or_else(|| format!("EPSG:{} is not in the crs-definitions database", source_epsg))?;
    let target_str = get_proj_string(target_epsg)
        .ok_or_else(|| format!("EPSG:{} is not in the crs-definitions database", target_epsg))?;

    let source_proj = Proj::from_proj_string(source_str)
        .map_err(|e| format!("Invalid source projection EPSG:{}: {:?}", source_epsg, e))?;
    let target_proj = Proj::from_proj_string(target_str)
        .map_err(|e| format!("Invalid target projection EPSG:{}: {:?}", target_epsg, e))?;

    // proj4rs uses radians for geographic coordinates
    let source_is_geographic = is_geographic_crs(source_epsg);
    let (x_in, y_in) = if source_is_geographic {
        (x.to_radians(), y.to_radians())
    } else {
        (x, y)
    };

    let mut point = (x_in, y_in, 0.0);
    transform(&source_proj, &target_proj, &mut point)
        .map_err(|e| format!("Transform from EPSG:{} to EPSG:{} failed: {:?}", source_epsg, target_epsg, e))?;

    // Convert back from radians if target is geographic
    let target_is_geographic = is_geographic_crs(target_epsg);
    let (out_x, out_y) = if target_is_geographic {
        (point.0.to_degrees(), point.1.to_degrees())
    } else {
        (point.0, point.1)
    };

    Ok((out_x, out_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_lon_lat_to_mercator_origin() {
        let (x, y) = lon_lat_to_mercator(0.0, 0.0);
        assert!(approx_eq(x, 0.0));
        assert!(approx_eq(y, 0.0));
    }

    #[test]
    fn test_mercator_to_lon_lat_origin() {
        let (lon, lat) = mercator_to_lon_lat(0.0, 0.0);
        assert!(approx_eq(lon, 0.0));
        assert!(approx_eq(lat, 0.0));
    }

    #[test]
    fn test_roundtrip_4326_3857() {
        let test_points = [
            (0.0, 0.0),
            (10.0, 51.5),   // London-ish
            (-122.4, 37.8), // San Francisco
            (139.7, 35.7),  // Tokyo
        ];

        for (lon, lat) in test_points {
            let (x, y) = lon_lat_to_mercator(lon, lat);
            let (lon2, lat2) = mercator_to_lon_lat(x, y);
            assert!(approx_eq(lon, lon2), "lon: {} != {}", lon, lon2);
            assert!(approx_eq(lat, lat2), "lat: {} != {}", lat, lat2);
        }
    }

    #[test]
    fn test_extreme_latitudes() {
        // proj4rs handles extreme latitudes - verify we get finite values
        let (_, y1) = lon_lat_to_mercator(0.0, 85.0);
        let (_, y2) = lon_lat_to_mercator(0.0, -85.0);
        assert!(y1.is_finite(), "85 deg should produce finite y");
        assert!(y2.is_finite(), "-85 deg should produce finite y");
        assert!(y1 > 0.0, "positive lat should give positive y");
        assert!(y2 < 0.0, "negative lat should give negative y");
    }

    #[test]
    fn test_project_point_same_crs() {
        let result = project_point(4326, 4326, 10.0, 51.5);
        assert!(result.is_ok());
        let (x, y) = result.unwrap();
        assert!(approx_eq(x, 10.0));
        assert!(approx_eq(y, 51.5));
    }

    #[test]
    fn test_project_point_4326_to_3857() {
        let result = project_point(4326, 3857, 0.0, 0.0);
        assert!(result.is_ok());
        let (x, y) = result.unwrap();
        assert!(approx_eq(x, 0.0));
        assert!(approx_eq(y, 0.0));
    }

    #[test]
    fn test_project_point_3857_to_4326() {
        let result = project_point(3857, 4326, 0.0, 0.0);
        assert!(result.is_ok());
        let (lon, lat) = result.unwrap();
        assert!(approx_eq(lon, 0.0));
        assert!(approx_eq(lat, 0.0));
    }

    #[test]
    fn test_project_point_via_proj4rs() {
        // Test a transformation that goes through proj4rs
        // EPSG:32633 is UTM zone 33N
        let result = project_point(4326, 32633, 15.0, 52.0);
        assert!(result.is_ok(), "Should support UTM zones: {:?}", result);
        let (x, y) = result.unwrap();
        // UTM coordinates should be in meters, roughly 500000 for easting near zone center
        assert!(x > 400000.0 && x < 600000.0, "UTM easting: {}", x);
        assert!(y > 5000000.0 && y < 6000000.0, "UTM northing: {}", y);
    }

    #[test]
    fn test_project_point_roundtrip_utm() {
        // Roundtrip through UTM
        let lon = 15.0;
        let lat = 52.0;

        let to_utm = project_point(4326, 32633, lon, lat);
        assert!(to_utm.is_ok());
        let (x, y) = to_utm.unwrap();

        let back = project_point(32633, 4326, x, y);
        assert!(back.is_ok());
        let (lon2, lat2) = back.unwrap();

        assert!((lon - lon2).abs() < 1e-5, "lon roundtrip: {} -> {}", lon, lon2);
        assert!((lat - lat2).abs() < 1e-5, "lat roundtrip: {} -> {}", lat, lat2);
    }

    #[test]
    fn test_get_proj_string_common_codes() {
        assert!(get_proj_string(4326).is_some(), "4326 should be in database");
        assert!(get_proj_string(3857).is_some(), "3857 should be in database");
        assert!(get_proj_string(32633).is_some(), "UTM 33N should be in database");
    }

    #[test]
    fn test_is_geographic_crs() {
        assert!(is_geographic_crs(4326), "4326 is geographic");
        assert!(!is_geographic_crs(3857), "3857 is projected");
        assert!(!is_geographic_crs(32633), "UTM is projected");
    }

    #[test]
    fn test_unsupported_epsg_code() {
        // Use an EPSG code that definitely doesn't exist
        let result = project_point(4326, 999999, 0.0, 0.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not in the crs-definitions database"));
    }
}
