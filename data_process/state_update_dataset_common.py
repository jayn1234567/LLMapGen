#!/usr/bin/env python3
import argparse
import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon, box, shape
except ModuleNotFoundError:
    gpd = None
    rasterio = None
    from_origin = None
    GeometryCollection = LineString = MultiLineString = MultiPolygon = Polygon = box = shape = None


TASK_TEXT = "Please construct the complete road map in the current BEV (Bird's Eye View) image patch."
INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8


def require_geo_dependencies():
    missing = []
    if gpd is None:
        missing.append("geopandas")
    if rasterio is None:
        missing.append("rasterio")
    if Polygon is None:
        missing.append("shapely")
    if missing:
        raise ModuleNotFoundError(
            "Missing geospatial dependencies: "
            + ", ".join(missing)
            + ". Install them in the data-processing environment before running dataset generation."
        )


@dataclass(frozen=True)
class RawSample:
    sample_id: str
    root: Path
    lane_geojson: Path
    intersection_geojson: Path
    image_tiff: Path
    mask_tiff: Path


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def json_safe(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def sample_id_from_root(root: Path) -> str:
    return root.name


def find_geojson(label_dir: Path, preferred_names, stem_keywords):
    for name in preferred_names:
        path = label_dir / name
        if path.exists():
            return path
    if not label_dir.exists():
        return label_dir / preferred_names[0]
    geojson_files = sorted(label_dir.glob("*.geojson"))
    preferred_lower = {name.lower() for name in preferred_names}
    for path in geojson_files:
        if path.name.lower() in preferred_lower:
            return path
    for path in geojson_files:
        stem = path.stem.lower()
        if any(keyword in stem for keyword in stem_keywords):
            return path
    return label_dir / preferred_names[0]


def required_paths(root: Path) -> RawSample:
    label_dir = root / "label_check_crop"
    return RawSample(
        sample_id=sample_id_from_root(root),
        root=root,
        lane_geojson=find_geojson(label_dir, ("Lane.geojson", "lane.geojson"), ("lane",)),
        intersection_geojson=find_geojson(
            label_dir,
            ("intersection.geojson", "Intersection.geojson"),
            ("intersection",),
        ),
        image_tiff=root / "inter_patch_tif" / "0_inter.tif",
        mask_tiff=root / "patch_tif" / "0_edit_poly.tif",
    )


def is_valid_sample_root(root: Path, require_intersection: bool = False) -> bool:
    sample = required_paths(root)
    required = [sample.lane_geojson, sample.image_tiff, sample.mask_tiff]
    if require_intersection:
        required.append(sample.intersection_geojson)
    return all(path.exists() for path in required)


def safe_extract_tar_gz(archive_path: Path, delete_archive: bool) -> Path:
    target_dir = archive_path.with_suffix("").with_suffix("")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_resolved = target_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (target_dir / member.name).resolve()
            try:
                member_path.relative_to(target_resolved)
            except ValueError:
                raise ValueError(f"unsafe archive member path: {member.name}")
        tar.extractall(path=target_dir)
    if delete_archive and find_sample_roots(target_dir, require_intersection=False):
        archive_path.unlink()
    return target_dir


def extract_archives(input_root: Path, delete_archive: bool):
    archives = sorted(input_root.rglob("*.tar.gz"))
    for archive in archives:
        safe_extract_tar_gz(archive, delete_archive=delete_archive)


def find_sample_roots(input_root: Path, require_intersection: bool = False):
    roots = set()
    for inter_dir in input_root.rglob("inter_patch_tif"):
        root = inter_dir.parent
        if is_valid_sample_root(root, require_intersection=require_intersection):
            roots.add(root)
    return sorted(roots, key=lambda path: str(path))


def discover_samples(input_root: Path, include_intersections: bool, delete_archives: bool, limit_samples=None):
    extract_archives(input_root, delete_archive=delete_archives)
    roots = find_sample_roots(input_root, require_intersection=include_intersections)
    samples = [required_paths(root) for root in roots]
    samples = sorted(samples, key=lambda sample: (sample.sample_id, str(sample.root)))
    if limit_samples is not None:
        samples = samples[:limit_samples]
    return samples


def split_samples(samples, train_ratio: float, seed: int):
    ordered = list(samples)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if len(ordered) <= 1:
        return ordered, []
    n_train = int(len(ordered) * train_ratio)
    n_train = max(1, min(len(ordered) - 1, n_train))
    return ordered[:n_train], ordered[n_train:]


def read_masked_image(image_path: Path, mask_path: Path):
    with rasterio.open(image_path) as src:
        image = src.read()
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
    with rasterio.open(mask_path) as src:
        mask = src.read()
    mask_any = (mask > 0).any(axis=0, keepdims=True)
    image = np.where(mask_any, image, 0)
    return image, meta, transform, crs


def image_chunk_to_pil(chunk: np.ndarray) -> Image.Image:
    if chunk.shape[0] == 1:
        arr = np.repeat(chunk, 3, axis=0)
    else:
        arr = chunk[:3]
    arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pad_image_to_patch_grid(image: np.ndarray, patch_size: int):
    _, height, width = image.shape
    pad_height = (-height) % patch_size
    pad_width = (-width) % patch_size
    if pad_height == 0 and pad_width == 0:
        return image, [width, height]
    padded = np.pad(
        image,
        ((0, 0), (0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )
    return padded, [width, height]


def coord_to_pixel(coord, inverse_transform):
    x, y = float(coord[0]), float(coord[1])
    px, py = inverse_transform * (x, y)
    return [float(px), float(py)]


def line_to_pixel_coords(line: LineString, inverse_transform):
    return [coord_to_pixel(coord, inverse_transform) for coord in line.coords]


def polygon_to_pixel_polygon(poly: Polygon, inverse_transform):
    exterior = [coord_to_pixel(coord, inverse_transform) for coord in poly.exterior.coords]
    interiors = [
        [coord_to_pixel(coord, inverse_transform) for coord in ring.coords]
        for ring in poly.interiors
    ]
    return Polygon(exterior, interiors)


def load_line_geometries(path: Path, crs, transform, simplify_tolerance: float):
    if not path.exists():
        return []
    gdf = gpd.read_file(path).to_crs(crs)
    lines = []
    for index, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        geoms = []
        if isinstance(geom, LineString):
            geoms = [geom]
        elif isinstance(geom, MultiLineString):
            geoms = list(geom.geoms)
        for part_idx, line in enumerate(geoms):
            if len(line.coords) >= 2:
                lines.append({
                    "category": "centerline",
                    "geometry": line,
                    "_source_line_index": int(index),
                    "_source_part_index": part_idx,
                })
    return lines


def geojson_crs_name(payload):
    crs_info = payload.get("crs") if isinstance(payload, dict) else None
    if not isinstance(crs_info, dict):
        return "EPSG:4326"
    properties = crs_info.get("properties")
    if not isinstance(properties, dict):
        return "EPSG:4326"
    name = properties.get("name")
    if not name:
        return "EPSG:4326"
    if str(name).upper().endswith("CRS84"):
        return "EPSG:4326"
    return name


def read_geojson_features_as_gdf(path: Path, dst_crs):
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list) or not features:
        return gpd.GeoDataFrame(geometry=[], crs=dst_crs)

    rows = []
    geometries = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        try:
            geom = shape(geometry)
        except Exception:
            continue
        if geom is None or geom.is_empty:
            continue
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        rows.append(dict(properties))
        geometries.append(geom)

    src_crs = geojson_crs_name(payload)
    fallback_gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs=src_crs)
    if fallback_gdf.empty:
        return fallback_gdf.to_crs(dst_crs)
    return fallback_gdf.to_crs(dst_crs)


def load_intersection_geometries(path: Path, crs, transform, simplify_tolerance: float):
    if not path.exists():
        return []
    gdf = gpd.read_file(path).to_crs(crs)
    if len(gdf) == 0:
        gdf = read_geojson_features_as_gdf(path, crs)
    polygons = []
    for index, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        geoms = polygon_parts(geom)
        properties = {
            key: value
            for key, value in row.items()
            if key != "geometry" and not (isinstance(value, float) and np.isnan(value))
        }
        for part_idx, poly in enumerate(geoms):
            if not poly.is_empty and poly.area > 0:
                polygons.append({
                    "geometry": poly,
                    "source_properties": properties,
                    "source_index": int(index),
                    "source_part_index": part_idx,
                })
    return polygons


def patch_window_polygon(transform, x0, y0, patch_size):
    return Polygon([
        transform * (x0, y0),
        transform * (x0 + patch_size, y0),
        transform * (x0 + patch_size, y0 + patch_size),
        transform * (x0, y0 + patch_size),
        transform * (x0, y0),
    ])


def patch_window_transform(transform, x0, y0):
    return from_origin(
        transform.xoff + x0 * transform.a,
        transform.yoff + y0 * transform.e,
        transform.a,
        abs(transform.e),
    )


def map_coord_to_local_point(coord, window_transform, patch_size):
    px, py = ~window_transform * (float(coord[0]), float(coord[1]))
    x = clamp(abs(int(round(px))), 0, patch_size - 1)
    y = clamp(abs(int(round(py))), 0, patch_size - 1)
    return [x, y]


def line_parts(geom):
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        parts = []
        for sub in geom.geoms:
            parts.extend(line_parts(sub))
        return parts
    return []


def endpoint_type_from_map_line(original_line, clipped_endpoint, tol=1e-6):
    point = np.array([float(clipped_endpoint[0]), float(clipped_endpoint[1])])
    original_start = np.array([float(original_line.coords[0][0]), float(original_line.coords[0][1])])
    original_end = np.array([float(original_line.coords[-1][0]), float(original_line.coords[-1][1])])
    if np.linalg.norm(point - original_start) <= tol or np.linalg.norm(point - original_end) <= tol:
        return "inside"
    return "cut"


def clip_lanes_to_patch(lines, transform, x0, y0, patch_size):
    window_polygon = patch_window_polygon(transform, x0, y0, patch_size)
    window_transform = patch_window_transform(transform, x0, y0)
    results = []
    for idx, line in enumerate(lines):
        geom = line["geometry"]
        if not geom.intersects(window_polygon):
            continue
        clipped = geom.intersection(window_polygon)
        for part_idx, clipped_line in enumerate(line_parts(clipped)):
            if clipped_line.is_empty or len(clipped_line.coords) < 2:
                continue
            local_points = [map_coord_to_local_point(coord, window_transform, patch_size) for coord in clipped_line.coords]
            local_points = dedupe_points(local_points)
            if len(local_points) < 2:
                continue
            if np.linalg.norm(np.array(local_points[0]) - np.array(local_points[-1])) < 1:
                continue
            global_pixel_points = [[point[0] + x0, point[1] + y0] for point in local_points]
            results.append({
                "category": "centerline",
                "start_type": endpoint_type_from_map_line(geom, clipped_line.coords[0]),
                "end_type": endpoint_type_from_map_line(geom, clipped_line.coords[-1]),
                "points": local_points,
                "_source_line_index": line.get("_source_line_index", idx),
                "_source_part_index": line.get("_source_part_index", part_idx),
                "_source_points": global_pixel_points,
                "_patch_x0": x0,
                "_patch_y0": y0,
            })
    return results


def region_code(x, y, xmin, ymin, xmax, ymax):
    code = INSIDE
    if x < xmin:
        code |= LEFT
    elif x > xmax:
        code |= RIGHT
    if y < ymin:
        code |= BOTTOM
    elif y > ymax:
        code |= TOP
    return code


def clip_segment(p0, p1, xmin, ymin, xmax, ymax):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    c0 = region_code(x0, y0, xmin, ymin, xmax, ymax)
    c1 = region_code(x1, y1, xmin, ymin, xmax, ymax)
    p0_cut = False
    p1_cut = False

    while True:
        if not (c0 | c1):
            return {"p0": [x0, y0], "p1": [x1, y1], "p0_cut": p0_cut, "p1_cut": p1_cut}
        if c0 & c1:
            return None

        out = c0 or c1
        if out & TOP:
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
            y = ymax
        elif out & BOTTOM:
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
            y = ymin
        elif out & RIGHT:
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
            x = xmax
        else:
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
            x = xmin

        if out == c0:
            x0, y0 = x, y
            p0_cut = True
            c0 = region_code(x0, y0, xmin, ymin, xmax, ymax)
        else:
            x1, y1 = x, y
            p1_cut = True
            c1 = region_code(x1, y1, xmin, ymin, xmax, ymax)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def round_local_point(point, x0, y0, patch_size):
    x = clamp(int(round(point[0] - x0)), 0, patch_size - 1)
    y = clamp(int(round(point[1] - y0)), 0, patch_size - 1)
    return [x, y]


def local_point_with_cut(point, is_cut, x0, y0, patch_size):
    return {"point": round_local_point(point, x0, y0, patch_size), "cut": bool(is_cut)}


def dedupe_points(points):
    out = []
    for point in points:
        if not out or point != out[-1]:
            out.append(point)
    return out


def dedupe_flagged_points(items):
    out = []
    for item in items:
        if out and item["point"] == out[-1]["point"]:
            out[-1]["cut"] = out[-1]["cut"] or item["cut"]
            continue
        out.append(item)
    return out


def clip_polyline_to_patch(line, x0, y0, patch_size, source_line_index=None):
    points = line.get("points") or []
    if len(points) < 2:
        return []
    xmin, ymin = x0, y0
    xmax, ymax = x0 + patch_size - 1, y0 + patch_size - 1
    clipped_lines = []
    current = []

    for p0, p1 in zip(points[:-1], points[1:]):
        clipped = clip_segment(p0, p1, xmin, ymin, xmax, ymax)
        if clipped is None:
            if len(current) >= 2:
                clipped_lines.append(current)
            current = []
            continue

        fp0 = local_point_with_cut(clipped["p0"], clipped["p0_cut"], x0, y0, patch_size)
        fp1 = local_point_with_cut(clipped["p1"], clipped["p1_cut"], x0, y0, patch_size)
        if fp0["point"] == fp1["point"]:
            continue

        if current and current[-1]["point"] == fp0["point"]:
            current[-1]["cut"] = current[-1]["cut"] or fp0["cut"]
            current.append(fp1)
        else:
            if len(current) >= 2:
                clipped_lines.append(current)
            current = [fp0, fp1]

        if clipped["p1_cut"]:
            if len(current) >= 2:
                clipped_lines.append(current)
            current = []

    if len(current) >= 2:
        clipped_lines.append(current)

    results = []
    for flagged_pts in clipped_lines:
        flagged_pts = dedupe_flagged_points(flagged_pts)
        if len(flagged_pts) < 2:
            continue
        pts = [item["point"] for item in flagged_pts]
        results.append({
            "category": "centerline",
            "start_type": "cut" if flagged_pts[0]["cut"] else "inside",
            "end_type": "cut" if flagged_pts[-1]["cut"] else "inside",
            "points": pts,
            "_source_line_index": source_line_index if source_line_index is not None else line.get("_source_line_index"),
            "_source_points": points,
            "_patch_x0": x0,
            "_patch_y0": y0,
        })
    return results


def polygon_parts(geom):
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        parts = []
        for sub in geom.geoms:
            parts.extend(polygon_parts(sub))
        return parts
    return []


def local_ring_points(poly: Polygon, x0, y0, patch_size, window_transform=None):
    if window_transform is None:
        points = [round_local_point(point, x0, y0, patch_size) for point in poly.exterior.coords]
    else:
        points = [map_coord_to_local_point(point, window_transform, patch_size) for point in poly.exterior.coords]
    points = dedupe_points(points)
    if len(points) >= 2 and points[0] != points[-1]:
        points.append(points[0])
    return points


def clip_intersections_to_patch(intersections, x0, y0, patch_size, transform=None):
    if transform is None:
        bbox = box(x0, y0, x0 + patch_size - 1, y0 + patch_size - 1)
        window_transform = None
    else:
        bbox = patch_window_polygon(transform, x0, y0, patch_size)
        window_transform = patch_window_transform(transform, x0, y0)
    results = []
    for idx, item in enumerate(intersections):
        geom = item["geometry"]
        if not geom.intersects(bbox):
            continue
        clipped = geom.intersection(bbox)
        is_cut = not geom.difference(bbox).is_empty
        for part_idx, poly in enumerate(polygon_parts(clipped)):
            if poly.is_empty or poly.area <= 0:
                continue
            pts = local_ring_points(poly, x0, y0, patch_size, window_transform=window_transform)
            if len(pts) < 4:
                continue
            results.append({
                "category": "intersection",
                "is_cut": bool(is_cut),
                "points": pts,
                "_source_intersection_index": item.get("source_index", idx),
                "_source_part_index": item.get("source_part_index", part_idx),
                "_source_properties": item.get("source_properties", {}),
                "_patch_x0": x0,
                "_patch_y0": y0,
            })
    return results


def is_near(value, target, tol):
    return abs(float(value) - float(target)) <= tol


def squared_distance(a, b):
    return (float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2


def shift_neighbor_point_to_current(point, side, patch_size):
    x, y = int(round(point[0])), int(round(point[1]))
    if side == "left":
        return [x - patch_size, y]
    if side == "top":
        return [x, y - patch_size]
    return [x, y]


def source_trace_points(line, side, boundary_local, boundary_at_end, patch_size, max_points):
    source_points = line.get("_source_points") or []
    if len(source_points) < 2:
        return []
    patch_x0 = line.get("_patch_x0", 0)
    patch_y0 = line.get("_patch_y0", 0)
    boundary_global = [boundary_local[0] + patch_x0, boundary_local[1] + patch_y0]
    nearest_idx = min(range(len(source_points)), key=lambda idx: squared_distance(source_points[idx], boundary_global))
    if boundary_at_end:
        start_idx = max(0, nearest_idx - (max_points - 1))
        selected = source_points[start_idx:nearest_idx + 1]
        if not selected or squared_distance(selected[-1], boundary_global) > 1.0:
            selected = selected[-(max_points - 1):] + [boundary_global]
    else:
        end_idx = min(len(source_points), nearest_idx + max_points)
        selected = source_points[nearest_idx:end_idx]
        if not selected or squared_distance(selected[0], boundary_global) > 1.0:
            selected = [boundary_global] + selected[1:max_points]
        selected = list(reversed(selected))
    local_points = [[point[0] - patch_x0, point[1] - patch_y0] for point in selected[-max_points:]]
    return [shift_neighbor_point_to_current(point, side, patch_size) for point in local_points]


def make_trace_from_line(line, side, patch_size, max_points, boundary_tol):
    points = line.get("points") or []
    if len(points) < 2:
        return None
    if side == "left":
        boundary = lambda p: p[0] >= patch_size - 1 - boundary_tol
    elif side == "top":
        boundary = lambda p: p[1] >= patch_size - 1 - boundary_tol
    else:
        return None

    if line.get("end_type") == "cut" and boundary(points[-1]):
        trace_points = source_trace_points(line, side, points[-1], True, patch_size, max_points)
        if not trace_points:
            trace_points = [shift_neighbor_point_to_current(point, side, patch_size) for point in points[-max_points:]]
    elif line.get("start_type") == "cut" and boundary(points[0]):
        trace_points = source_trace_points(line, side, points[0], False, patch_size, max_points)
        if not trace_points:
            trace_points = [shift_neighbor_point_to_current(point, side, patch_size) for point in reversed(points[:max_points])]
    else:
        return None
    trace_points = dedupe_points(trace_points[-max_points:])
    if not trace_points:
        return None
    return {"side": side, "points": trace_points}


def build_incoming_traces(patch_lines_by_rc, row, col, patch_size, max_traces, trace_points, boundary_tol):
    traces = []
    for side, lines in [
        ("left", patch_lines_by_rc.get((row, col - 1), [])),
        ("top", patch_lines_by_rc.get((row - 1, col), [])),
    ]:
        side_count = 0
        for line in lines:
            trace = make_trace_from_line(line, side, patch_size, trace_points, boundary_tol)
            if trace is None:
                continue
            trace["id"] = f"{'L' if side == 'left' else 'T'}{side_count}"
            traces.append(trace)
            side_count += 1
            if side_count >= max_traces:
                break
    return traces


def boundary_points_for_intersection(intersection, side, patch_size, boundary_tol, max_points):
    if not intersection.get("is_cut"):
        return []
    points = dedupe_points(intersection.get("points") or [])
    if side == "left":
        selected = [point for point in points if is_near(point[0], patch_size - 1, boundary_tol)]
    elif side == "top":
        selected = [point for point in points if is_near(point[1], patch_size - 1, boundary_tol)]
    else:
        return []
    selected = dedupe_points(selected)
    if len(selected) > max_points:
        if max_points == 1:
            selected = [selected[len(selected) // 2]]
        else:
            step = (len(selected) - 1) / (max_points - 1)
            selected = [selected[round(i * step)] for i in range(max_points)]
    return [shift_neighbor_point_to_current(point, side, patch_size) for point in selected]


def build_incoming_intersections(patch_lines_by_rc, row, col, patch_size, max_hints, hint_points, boundary_tol):
    hints = []
    for side, lines in [
        ("left", patch_lines_by_rc.get((row, col - 1), [])),
        ("top", patch_lines_by_rc.get((row - 1, col), [])),
    ]:
        side_count = 0
        for line in lines:
            if line.get("category") != "intersection":
                continue
            points = boundary_points_for_intersection(line, side, patch_size, boundary_tol, hint_points)
            if not points:
                continue
            prefix = "IL" if side == "left" else "IT"
            hints.append({"id": f"{prefix}{side_count}", "side": side, "points": points})
            side_count += 1
            if side_count >= max_hints:
                break
    return hints


def line_side_priority(line, patch_size, boundary_tol):
    points = line.get("points") or []
    if line.get("category") != "centerline" or len(points) < 2:
        return 2, line
    left_start = line.get("start_type") == "cut" and is_near(points[0][0], 0, boundary_tol)
    left_end = line.get("end_type") == "cut" and is_near(points[-1][0], 0, boundary_tol)
    top_start = line.get("start_type") == "cut" and is_near(points[0][1], 0, boundary_tol)
    top_end = line.get("end_type") == "cut" and is_near(points[-1][1], 0, boundary_tol)
    if left_start or left_end:
        return 0, orient_line_from_endpoint(line, "start" if left_start else "end")
    if top_start or top_end:
        return 1, orient_line_from_endpoint(line, "start" if top_start else "end")
    return 2, line


def orient_line_from_endpoint(line, endpoint):
    if endpoint == "start":
        return line
    reversed_line = dict(line)
    reversed_line["points"] = list(reversed(line["points"]))
    reversed_line["start_type"], reversed_line["end_type"] = line["end_type"], line["start_type"]
    if "_source_points" in line:
        reversed_line["_source_points"] = list(reversed(line["_source_points"]))
    return reversed_line


def intersection_side_priority(line, patch_size, boundary_tol):
    points = line.get("points") or []
    if line.get("category") != "intersection" or not line.get("is_cut"):
        return 5
    if any(is_near(point[0], 0, boundary_tol) for point in points):
        return 3
    if any(is_near(point[1], 0, boundary_tol) for point in points):
        return 4
    return 5


def public_line(line):
    return {key: value for key, value in line.items() if not key.startswith("_")}


def sort_target_lines(lines, patch_size, boundary_tol):
    ordered = []
    for idx, line in enumerate(lines):
        if line.get("category") == "centerline":
            priority, oriented = line_side_priority(line, patch_size, boundary_tol)
        else:
            priority, oriented = intersection_side_priority(line, patch_size, boundary_tol), line
        pts = oriented.get("points") or [[999999, 999999]]
        first = pts[0]
        ordered.append((priority, first[1], first[0], idx, oriented))
    return [item[-1] for item in sorted(ordered)]


def make_prompt(include_intersections: bool, incoming_traces, incoming_intersections=None, phase="a"):
    trace_json = json.dumps(incoming_traces, ensure_ascii=False, separators=(",", ":"))
    parts = [
        "<image>",
        TASK_TEXT,
        "",
        "Incoming traces JSON:",
        trace_json,
    ]
    if include_intersections:
        inter_json = json.dumps(incoming_intersections or [], ensure_ascii=False, separators=(",", ":"))
        parts.extend(["", "Incoming intersections JSON:", inter_json])
    if phase == "b":
        parts.extend([
            "",
            "Each incoming trace has 1 to 3 points. If multiple points are present, they are ordered from the previous patch interior toward the current patch boundary.",
            "Incoming traces are continuity hints only; they may be incomplete or absent.",
        ])
        if include_intersections:
            parts.append("Each incoming intersection has 1 to 3 boundary points from neighboring patches.")
    return "\n".join(parts)


def build_sft_record(row, patch_size, include_intersections, phase):
    incoming_traces = row["incoming_traces"] if phase == "b" else []
    incoming_intersections = row.get("incoming_intersections", []) if phase == "b" else []
    prompt = make_prompt(include_intersections, incoming_traces, incoming_intersections, phase=phase)
    meta = dict(row["meta"])
    meta.update({
        "scan_order": "row_major_top_to_bottom_left_to_right",
        "available_neighbors": ["left", "top"],
        "train_shuffle_allowed": True,
        "trace_source_train": "gt_left_top_neighbors" if phase == "b" else "none",
        "trace_source_infer": "predicted_left_top_neighbors",
        "phase": f"phase_{phase}",
    })
    if include_intersections:
        meta["intersection_hint_source_train"] = "gt_left_top_neighbors" if phase == "b" else "none"
    target_text = json.dumps({"lines": row["target_lines"]}, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": row["id"],
        "image": row["image"],
        "meta": meta,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": target_text},
        ],
    }


def target_has_lines(row):
    return bool(row.get("target_lines"))


def cap_empty_rows(rows, max_empty_ratio):
    if max_empty_ratio is None or max_empty_ratio < 0:
        return rows
    nonempty = [row for row in rows if target_has_lines(row)]
    empty = [row for row in rows if not target_has_lines(row)]
    if not nonempty:
        return []
    max_empty = int((max_empty_ratio / max(1e-8, 1.0 - max_empty_ratio)) * len(nonempty))
    return nonempty + empty[:max_empty]


def process_sample(sample: RawSample, output_root: Path, split_name: str, include_intersections: bool, args):
    image_arr, meta, transform, crs = read_masked_image(sample.image_tiff, sample.mask_tiff)
    image_arr, original_image_size = pad_image_to_patch_grid(image_arr, args.patch_size)
    lines = load_line_geometries(sample.lane_geojson, crs, transform, args.simplify_tolerance)
    intersections = []
    if include_intersections and sample.intersection_geojson.exists():
        intersections = load_intersection_geometries(sample.intersection_geojson, crs, transform, args.simplify_tolerance)

    _, height, width = image_arr.shape
    patch_lines_by_rc = {}
    patch_source_meta = {}
    for y0 in range(0, height - args.patch_size + 1, args.stride):
        for x0 in range(0, width - args.patch_size + 1, args.stride):
            chunk = image_arr[:, y0:y0 + args.patch_size, x0:x0 + args.patch_size]
            if np.all(chunk == 0):
                continue
            row = y0 // args.stride
            col = x0 // args.stride
            local_lines = []
            local_lines.extend(clip_lanes_to_patch(lines, transform, x0, y0, args.patch_size))
            if include_intersections:
                local_lines.extend(clip_intersections_to_patch(intersections, x0, y0, args.patch_size, transform=transform))
            local_lines = sort_target_lines(local_lines, args.patch_size, args.boundary_tol)
            patch_lines_by_rc[(row, col)] = local_lines
            patch_source_meta[(row, col)] = {
                "intersection_sources": [
                    {
                        "is_cut": line.get("is_cut"),
                        "source_properties": line.get("_source_properties", {}),
                    }
                    for line in local_lines
                    if line.get("category") == "intersection"
                ]
            }

    rows = []
    patch_count = 0
    for (row, col), local_lines in sorted(patch_lines_by_rc.items()):
        if args.max_patches_per_sample is not None and patch_count >= args.max_patches_per_sample:
            break
        if not local_lines and args.max_empty_ratio == 0:
            continue
        x0 = col * args.stride
        y0 = row * args.stride
        patch_id = f"{sample.sample_id}_r{row:03d}_c{col:03d}"
        rel_image = Path("images") / split_name / sample.sample_id / f"{patch_id}.png"

        incoming_traces = build_incoming_traces(
            patch_lines_by_rc, row, col, args.patch_size,
            args.max_traces_per_side, args.trace_points, args.boundary_tol,
        )
        incoming_intersections = []
        if include_intersections:
            incoming_intersections = build_incoming_intersections(
                patch_lines_by_rc, row, col, args.patch_size,
                args.max_intersections_per_side, args.intersection_hint_points, args.boundary_tol,
            )

        meta_payload = {
            "tile_id": sample.sample_id,
            "log_id": sample.sample_id,
            "patch_row": row,
            "patch_col": col,
            "row": row,
            "col": col,
            "x0": x0,
            "y0": y0,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "source_image_size": [width, height],
            "original_source_image_size": original_image_size,
            "coord_system": f"patch_local_{args.patch_size}",
            "task_mode": "state_update_centerline_intersection" if include_intersections else "state_update_centerline",
            "raw_sample_root": str(sample.root),
        }
        rows.append({
            "id": patch_id,
            "image": str(rel_image),
            "tile_id": sample.sample_id,
            "patch_row": row,
            "patch_col": col,
            "base_patch_box_full": [x0, y0, x0 + args.patch_size, y0 + args.patch_size],
            "incoming_traces": incoming_traces,
            "incoming_intersections": incoming_intersections,
            "target_lines": [public_line(line) for line in local_lines],
            "meta": meta_payload,
            **patch_source_meta[(row, col)],
        })
        patch_count += 1
    rows = cap_empty_rows(rows, args.max_empty_ratio)
    for row in rows:
        x0 = row["meta"]["x0"]
        y0 = row["meta"]["y0"]
        out_image = output_root / row["image"]
        out_image.parent.mkdir(parents=True, exist_ok=True)
        chunk = image_arr[:, y0:y0 + args.patch_size, x0:x0 + args.patch_size]
        image_chunk_to_pil(chunk).save(out_image)
    return rows


def validate_rows(rows, include_intersections, patch_size):
    errors = []
    for row in rows:
        for line in row.get("target_lines", []):
            category = line.get("category")
            points = line.get("points") or []
            for point in points:
                if len(point) != 2 or not all(isinstance(v, int) for v in point):
                    errors.append(f"{row['id']}: invalid point {point}")
                elif not (0 <= point[0] < patch_size and 0 <= point[1] < patch_size):
                    errors.append(f"{row['id']}: out-of-range point {point}")
            if category == "centerline":
                if line.get("start_type") not in {"cut", "inside"} or line.get("end_type") not in {"cut", "inside"}:
                    errors.append(f"{row['id']}: invalid centerline endpoint type")
            elif category == "intersection":
                if include_intersections and not isinstance(line.get("is_cut"), bool):
                    errors.append(f"{row['id']}: intersection missing boolean is_cut")
                if len(points) < 4 or points[0] != points[-1]:
                    errors.append(f"{row['id']}: intersection is not closed")
            else:
                errors.append(f"{row['id']}: unsupported category {category}")
        for trace in row.get("incoming_traces", []):
            if len(trace.get("points", [])) < 2:
                errors.append(f"{row['id']}: centerline trace has fewer than 2 points")
        for hint in row.get("incoming_intersections", []):
            if len(hint.get("points", [])) < 1:
                errors.append(f"{row['id']}: intersection hint has no points")
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"dataset validation failed with {len(errors)} errors:\n{preview}")


def build_dataset(include_intersections: bool, args):
    require_geo_dependencies()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    samples = discover_samples(
        input_root,
        include_intersections=include_intersections,
        delete_archives=not args.keep_archives,
        limit_samples=args.limit_samples,
    )
    if not samples:
        raise FileNotFoundError(f"no valid samples found under {input_root}")
    train_samples, test_samples = split_samples(samples, args.train_ratio, args.split_seed)
    split_manifest = {
        "split_unit": "raw_sample_folder",
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "include_intersections": include_intersections,
        "train_ids": [sample.sample_id for sample in train_samples],
        "test_ids": [sample.sample_id for sample in test_samples],
    }
    write_json(output_root / "split_manifest.json", split_manifest)

    split_rows = {}
    for split_name, split_samples_list in [("train", train_samples), ("test", test_samples)]:
        rows = []
        for sample in split_samples_list:
            rows.extend(process_sample(sample, output_root, split_name, include_intersections, args))
        validate_rows(rows, include_intersections, args.patch_size)
        split_rows[split_name] = rows

    for phase in ["a", "b"]:
        phase_dir = output_root / f"phase_{phase}"
        for split_name, rows in split_rows.items():
            sft_rows = [build_sft_record(row, args.patch_size, include_intersections, phase) for row in rows]
            write_jsonl(phase_dir / f"{split_name}.jsonl", sft_rows)
            write_jsonl(phase_dir / f"meta_{split_name}.jsonl", rows)

    info = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "task": "lane_intersection" if include_intersections else "lane_only",
        "num_raw_samples": len(samples),
        "num_train_raw_samples": len(train_samples),
        "num_test_raw_samples": len(test_samples),
        "num_train_patches": len(split_rows["train"]),
        "num_test_patches": len(split_rows["test"]),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "max_empty_ratio": args.max_empty_ratio,
        "phase_a_train_jsonl": str(output_root / "phase_a" / "train.jsonl"),
        "phase_a_test_jsonl": str(output_root / "phase_a" / "test.jsonl"),
        "phase_b_train_jsonl": str(output_root / "phase_b" / "train.jsonl"),
        "phase_b_test_jsonl": str(output_root / "phase_b" / "test.jsonl"),
    }
    write_json(output_root / "dataset_info.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=2))


def add_common_args(parser):
    parser.add_argument("--input-root", required=True, help="Directory containing raw sample folders or .tar.gz archives.")
    parser.add_argument("--output-root", required=True, help="Output dataset directory.")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-empty-ratio", type=float, default=0.1)
    parser.add_argument("--boundary-tol", type=float, default=1.0)
    parser.add_argument("--simplify-tolerance", type=float, default=0.5)
    parser.add_argument("--trace-points", type=int, default=3)
    parser.add_argument("--intersection-hint-points", type=int, default=3)
    parser.add_argument("--max-traces-per-side", type=int, default=8)
    parser.add_argument("--max-intersections-per-side", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--max-patches-per-sample", type=int, default=None)
    parser.add_argument("--keep-archives", action="store_true", help="Do not delete .tar.gz archives after successful extraction.")


def run_cli(include_intersections: bool, description: str):
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    args = parser.parse_args()
    build_dataset(include_intersections, args)
