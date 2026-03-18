from typing import Dict, List, Sequence, Tuple

import numpy as np


SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<state>",
    "<txt_xy>",
    "<txt_trace>",
    "<txt_end>",
    "<line>",
    "<pts>",
    "<eol>",
]

START_TYPES = ["start", "cut"]
END_TYPES = ["end", "cut"]
DEFAULT_LINE_TYPES = ["solid", "thick_solid", "dashed", "short_dashed", "others"]


class MapSequenceTokenizer:
    def __init__(
        self,
        image_size: int,
        categories: Sequence[str],
        max_seq_len: int,
        coord_num_bins: int = None,
        angle_num_bins: int = 360,
        line_types: Sequence[str] = (),
    ) -> None:
        self.image_size = int(image_size)
        self.categories = list(categories)
        self.line_types = [str(x) for x in line_types]
        self.max_seq_len = int(max_seq_len)
        self.coord_num_bins = int(coord_num_bins) if coord_num_bins is not None else int(image_size)
        self.coord_num_bins = max(2, self.coord_num_bins)
        self.angle_num_bins = max(2, int(angle_num_bins))

        self.itos = list(SPECIAL_TOKENS)
        self.itos.extend([f"<cat_{c}>" for c in self.categories])
        self.itos.extend([f"<lt_{lt}>" for lt in self.line_types])
        self.itos.extend([f"<s_{t}>" for t in START_TYPES])
        self.itos.extend([f"<e_{t}>" for t in END_TYPES])
        # Coordinate/angle token spaces are configurable for paper-scale quantization.
        self.itos.extend([f"<x_{i}>" for i in range(self.coord_num_bins)])
        self.itos.extend([f"<y_{i}>" for i in range(self.coord_num_bins)])
        self.itos.extend([f"<a_{i}>" for i in range(self.angle_num_bins)])

        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.state_id = self.stoi["<state>"]
        self.txt_xy_id = self.stoi["<txt_xy>"]
        self.txt_trace_id = self.stoi["<txt_trace>"]
        self.txt_end_id = self.stoi["<txt_end>"]
        self.line_id = self.stoi["<line>"]
        self.pts_id = self.stoi["<pts>"]
        self.eol_id = self.stoi["<eol>"]
        self.category_ids = [self.stoi[f"<cat_{c}>"] for c in self.categories if f"<cat_{c}>" in self.stoi]
        self.line_type_ids = [self.stoi[f"<lt_{lt}>"] for lt in self.line_types if f"<lt_{lt}>" in self.stoi]
        self.start_type_ids = [self.stoi[f"<s_{t}>"] for t in START_TYPES if f"<s_{t}>" in self.stoi]
        self.end_type_ids = [self.stoi[f"<e_{t}>"] for t in END_TYPES if f"<e_{t}>" in self.stoi]
        self.x_ids = [self.stoi[f"<x_{i}>"] for i in range(self.coord_num_bins)]
        self.y_ids = [self.stoi[f"<y_{i}>"] for i in range(self.coord_num_bins)]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode_lines(self, lines: List[Dict[str, np.ndarray]]) -> List[int]:
        toks = [self.bos_id]
        for line in lines:
            cat = line["category"]
            pts = line["points"]
            cat_tok = f"<cat_{cat}>"
            if cat_tok not in self.stoi:
                continue
            toks.append(self.stoi["<line>"])
            toks.append(self.stoi[cat_tok])
            line_type = normalize_line_type(line.get("line_type", ""))
            lt_tok = f"<lt_{line_type}>"
            if line_type and lt_tok in self.stoi:
                toks.append(self.stoi[lt_tok])
            s_tok = f"<s_{line.get('start_type', 'start')}>"
            e_tok = f"<e_{line.get('end_type', 'end')}>"
            if s_tok in self.stoi:
                toks.append(self.stoi[s_tok])
            if e_tok in self.stoi:
                toks.append(self.stoi[e_tok])
            toks.append(self.stoi["<pts>"])
            for x, y in pts:
                xq = self._quantize_coord(float(x))
                yq = self._quantize_coord(float(y))
                toks.append(self.stoi[f"<x_{xq}>"])
                toks.append(self.stoi[f"<y_{yq}>"])
            toks.append(self.stoi["<eol>"])
            if len(toks) >= self.max_seq_len - 1:
                break

        toks = toks[: self.max_seq_len - 1]
        toks.append(self.eos_id)
        return toks

    def decode_to_line_records(self, token_ids: Sequence[int]) -> List[Dict[str, np.ndarray]]:
        id_to_tok = self.itos
        lines: List[Dict[str, np.ndarray]] = []
        cur_cat = None
        cur_line_type = ""
        cur_start_type = "start"
        cur_end_type = "end"
        cur_points: List[Tuple[float, float]] = []
        reading_points = False
        x_buf = None
        cur_line_start = None
        cur_cat_pos = None

        for pos, idx in enumerate(token_ids):
            if idx < 0 or idx >= len(id_to_tok):
                continue
            tok = id_to_tok[idx]
            if tok == "<eos>":
                break
            if tok == "<line>":
                if cur_cat is not None and cur_points:
                    lines.append(
                        {
                            "category": cur_cat,
                            "line_type": cur_line_type,
                            "start_type": cur_start_type,
                            "end_type": cur_end_type,
                            "points": np.asarray(cur_points, dtype=np.float32),
                            "token_start": int(cur_line_start if cur_line_start is not None else pos),
                            "token_end": int(pos - 1),
                            "category_token_pos": None if cur_cat_pos is None else int(cur_cat_pos),
                        }
                    )
                cur_line_start = pos
                cur_cat = None
                cur_line_type = ""
                cur_start_type = "start"
                cur_end_type = "end"
                cur_points = []
                reading_points = False
                x_buf = None
                cur_cat_pos = None
                continue
            if tok.startswith("<cat_"):
                cur_cat = tok[len("<cat_") : -1]
                cur_cat_pos = pos
                continue
            if tok.startswith("<lt_"):
                cur_line_type = tok[len("<lt_") : -1]
                continue
            if tok.startswith("<s_"):
                cur_start_type = tok[len("<s_") : -1]
                continue
            if tok.startswith("<e_"):
                cur_end_type = tok[len("<e_") : -1]
                continue
            if tok == "<pts>":
                reading_points = True
                continue
            if tok == "<eol>":
                if cur_cat is not None and cur_points:
                    lines.append(
                        {
                            "category": cur_cat,
                            "line_type": cur_line_type,
                            "start_type": cur_start_type,
                            "end_type": cur_end_type,
                            "points": np.asarray(cur_points, dtype=np.float32),
                            "token_start": int(cur_line_start if cur_line_start is not None else 0),
                            "token_end": int(pos),
                            "category_token_pos": None if cur_cat_pos is None else int(cur_cat_pos),
                        }
                    )
                cur_cat = None
                cur_line_type = ""
                cur_start_type = "start"
                cur_end_type = "end"
                cur_points = []
                reading_points = False
                x_buf = None
                cur_line_start = None
                cur_cat_pos = None
                continue
            if not reading_points:
                continue
            if tok.startswith("<x_"):
                x_raw = int(tok[3:-1])
                x_raw = int(np.clip(x_raw, 0, self.coord_num_bins - 1))
                x_buf = self._dequantize_coord(x_raw)
                continue
            if tok.startswith("<y_") and x_buf is not None:
                y_raw = int(tok[3:-1])
                y_raw = int(np.clip(y_raw, 0, self.coord_num_bins - 1))
                y = self._dequantize_coord(y_raw)
                cur_points.append((x_buf, y))
                x_buf = None
        return lines

    def decode_to_lines(self, token_ids: Sequence[int]) -> List[Dict[str, np.ndarray]]:
        records = self.decode_to_line_records(token_ids)
        out: List[Dict[str, np.ndarray]] = []
        for rec in records:
            out.append(
                {
                    "category": rec["category"],
                    "line_type": rec["line_type"],
                    "start_type": rec["start_type"],
                    "end_type": rec["end_type"],
                    "points": rec["points"],
                }
            )
        return out

    def encode_text_target_xy(self, start_xy: Sequence[float], end_xy: Sequence[float]) -> List[int]:
        sx = self._quantize_coord(float(start_xy[0]))
        sy = self._quantize_coord(float(start_xy[1]))
        ex = self._quantize_coord(float(end_xy[0]))
        ey = self._quantize_coord(float(end_xy[1]))
        return [
            self.txt_xy_id,
            self.stoi[f"<x_{sx}>"],
            self.stoi[f"<y_{sy}>"],
            self.stoi[f"<x_{ex}>"],
            self.stoi[f"<y_{ey}>"],
            self.txt_end_id,
        ]

    def encode_text_trace_points(self, points_xy: np.ndarray, angles_deg: np.ndarray) -> List[int]:
        if points_xy.ndim != 2 or points_xy.shape[0] == 0:
            return [self.txt_trace_id, self.txt_end_id]
        out = [self.txt_trace_id]
        n = min(points_xy.shape[0], angles_deg.shape[0])
        for i in range(n):
            x = self._quantize_coord(float(points_xy[i, 0]))
            y = self._quantize_coord(float(points_xy[i, 1]))
            a = self._quantize_angle(float(angles_deg[i]))
            out.extend([self.stoi[f"<x_{x}>"], self.stoi[f"<y_{y}>"], self.stoi[f"<a_{a}>"]])
            if len(out) >= self.max_seq_len - 2:
                break
        out.append(self.txt_end_id)
        return out[: self.max_seq_len - 1]

    def _quantize_coord(self, value: float) -> int:
        value = float(np.clip(value, 0.0, float(self.image_size - 1)))
        if self.coord_num_bins <= 1 or self.image_size <= 1:
            return 0
        scale = float(self.coord_num_bins - 1) / float(self.image_size - 1)
        return int(np.clip(round(value * scale), 0, self.coord_num_bins - 1))

    def _dequantize_coord(self, token_idx: int) -> float:
        token_idx = int(np.clip(token_idx, 0, self.coord_num_bins - 1))
        if self.coord_num_bins <= 1:
            return 0.0
        scale = float(self.image_size - 1) / float(self.coord_num_bins - 1)
        return float(token_idx * scale)

    def _quantize_angle(self, angle_deg: float) -> int:
        angle = float(angle_deg) % 360.0
        scale = float(self.angle_num_bins - 1) / 360.0
        return int(np.clip(round(angle * scale), 0, self.angle_num_bins - 1))

    def valid_next_token_ids(
        self,
        prefix_ids: Sequence[int],
        min_points_per_line: int = 2,
        max_lines: int = None,
    ) -> List[int]:
        min_points = max(1, int(min_points_per_line))
        max_lines = None if max_lines is None else max(1, int(max_lines))
        toks = [int(x) for x in prefix_ids]
        if not toks:
            return [self.bos_id]
        if self.eos_id in toks:
            return [self.eos_id]
        if toks[0] != self.bos_id:
            return [self.bos_id]

        stage = "between_lines"
        point_pairs = 0
        num_lines = 0
        for tok in toks[1:]:
            if stage == "between_lines":
                if tok == self.line_id:
                    stage = "after_line"
                elif tok == self.eos_id:
                    return [self.eos_id]
                else:
                    return [self.line_id, self.eos_id]
            elif stage == "after_line":
                if tok in self.category_ids:
                    stage = "after_cat"
                else:
                    return list(self.category_ids)
            elif stage == "after_cat":
                if self.line_type_ids and tok in self.line_type_ids:
                    stage = "after_line_type"
                elif tok in self.start_type_ids:
                    stage = "after_start"
                elif self.line_type_ids:
                    return list(self.line_type_ids) + list(self.start_type_ids)
                else:
                    return list(self.start_type_ids)
            elif stage == "after_line_type":
                if tok in self.start_type_ids:
                    stage = "after_start"
                else:
                    return list(self.start_type_ids)
            elif stage == "after_start":
                if tok in self.end_type_ids:
                    stage = "after_end"
                else:
                    return list(self.end_type_ids)
            elif stage == "after_end":
                if tok == self.pts_id:
                    stage = "expect_x"
                    point_pairs = 0
                else:
                    return [self.pts_id]
            elif stage == "expect_x":
                if tok in self.x_ids:
                    stage = "expect_y"
                else:
                    return list(self.x_ids)
            elif stage == "expect_y":
                if tok in self.y_ids:
                    point_pairs += 1
                    stage = "after_xy"
                else:
                    return list(self.y_ids)
            elif stage == "after_xy":
                if tok in self.x_ids:
                    stage = "expect_y"
                elif tok == self.eol_id and point_pairs >= min_points:
                    num_lines += 1
                    stage = "between_lines"
                    point_pairs = 0
                else:
                    allowed = list(self.x_ids)
                    if point_pairs >= min_points:
                        allowed.append(self.eol_id)
                    return allowed

        if stage == "between_lines":
            if max_lines is not None and num_lines >= max_lines:
                return [self.eos_id]
            return [self.line_id, self.eos_id]
        if stage == "after_line":
            return list(self.category_ids)
        if stage == "after_cat":
            if self.line_type_ids:
                return list(self.line_type_ids) + list(self.start_type_ids)
            return list(self.start_type_ids)
        if stage == "after_line_type":
            return list(self.start_type_ids)
        if stage == "after_start":
            return list(self.end_type_ids)
        if stage == "after_end":
            return [self.pts_id]
        if stage == "expect_x":
            return list(self.x_ids)
        if stage == "expect_y":
            return list(self.y_ids)
        if stage == "after_xy":
            allowed = list(self.x_ids)
            if point_pairs >= min_points:
                allowed.append(self.eol_id)
            return allowed
        return [self.eos_id]


def _resample_polyline(points_xy: np.ndarray, interval_meter: float, max_points: int) -> np.ndarray:
    if len(points_xy) <= 1:
        return points_xy
    seg = np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=1)
    total = float(np.sum(seg))
    if total < 1e-6:
        return points_xy[:1]

    n = int(np.floor(total / max(interval_meter, 1e-6))) + 1
    n = max(2, min(n, max_points))
    target = np.linspace(0.0, total, n, dtype=np.float32)

    cum = np.concatenate(([0.0], np.cumsum(seg)))
    sampled = []
    for t in target:
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        t0, t1 = cum[j], cum[j + 1]
        ratio = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        p = points_xy[j] * (1.0 - ratio) + points_xy[j + 1] * ratio
        sampled.append(p)
    return np.asarray(sampled, dtype=np.float32)


def world_to_pixel(points_xy: np.ndarray, image_size: int, meter_range_half: float = 30.0) -> np.ndarray:
    # MapTR-like local BEV coordinates are roughly in [-30m, 30m] each axis.
    x = (points_xy[:, 0] + meter_range_half) / (2.0 * meter_range_half) * (image_size - 1)
    y = (meter_range_half - points_xy[:, 1]) / (2.0 * meter_range_half) * (image_size - 1)
    return np.stack([x, y], axis=-1)


def pixel_to_world(points_xy: np.ndarray, image_size: int, meter_range_half: float = 30.0) -> np.ndarray:
    x = (points_xy[:, 0] / (image_size - 1)) * (2.0 * meter_range_half) - meter_range_half
    y = meter_range_half - (points_xy[:, 1] / (image_size - 1)) * (2.0 * meter_range_half)
    return np.stack([x, y], axis=-1)


def serialize_annotation(
    annotation: Dict[str, List[np.ndarray]],
    categories: Sequence[str],
    image_size: int,
    interval_meter: float,
    max_lines: int,
    max_points_per_line: int,
    meter_range_half: float = 30.0,
) -> List[Dict[str, np.ndarray]]:
    lines: List[Dict[str, np.ndarray]] = []
    border_tol = max(2.0, float(image_size) * 0.02)
    for cat in categories:
        for poly in annotation.get(cat, []):
            if poly is None:
                continue
            arr = np.asarray(poly, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
                continue
            sampled = _resample_polyline(arr, interval_meter=interval_meter, max_points=max_points_per_line)
            pix = world_to_pixel(sampled, image_size=image_size, meter_range_half=float(meter_range_half))
            start_is_cut = _is_border_point(pix[0], image_size=image_size, tol=border_tol)
            end_is_cut = _is_border_point(pix[-1], image_size=image_size, tol=border_tol)
            lines.append(
                {
                    "category": cat,
                    "start_type": "cut" if start_is_cut else "start",
                    "end_type": "cut" if end_is_cut else "end",
                    "points": pix,
                }
            )

    # Reorder by distance of first point to origin (paper Sec.3.2).
    lines.sort(key=lambda d: float(np.linalg.norm(d["points"][0])) if len(d["points"]) > 0 else 1e9)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines


def _is_border_point(p: np.ndarray, image_size: int, tol: float) -> bool:
    x = float(p[0])
    y = float(p[1])
    lo = tol
    hi = (image_size - 1) - tol
    return x <= lo or x >= hi or y <= lo or y >= hi


def normalize_opensatmap_category(name: str) -> str:
    x = str(name).strip().lower()
    if x == "lane line":
        return "lane_line"
    if x == "virtual line":
        return "virtual_line"
    if x == "curb":
        return "curb"
    return x.replace(" ", "_")


def normalize_line_type(name: str) -> str:
    x = str(name).strip().lower()
    if not x:
        return ""
    x = x.replace("-", " ").replace("/", " ").replace("_", " ")
    x = " ".join(x.split())
    mapping = {
        "solid": "solid",
        "thick solid": "thick_solid",
        "dashed": "dashed",
        "short dashed": "short_dashed",
        "others": "others",
        "other": "others",
        "none": "",
    }
    return mapping.get(x, "others")


def serialize_opensatmap_lines(
    raw_lines: Sequence[Dict],
    categories: Sequence[str],
    line_types: Sequence[str],
    src_w: int,
    src_h: int,
    image_size: int,
    interval_meter: float,
    meter_per_pixel: float,
    max_lines: int,
    max_points_per_line: int,
) -> List[Dict[str, np.ndarray]]:
    lines: List[Dict[str, np.ndarray]] = []
    cat_set = set(str(c) for c in categories)
    line_type_set = set(str(x) for x in line_types if str(x))
    sx = float(max(1, image_size - 1)) / float(max(1, src_w - 1))
    sy = float(max(1, image_size - 1)) / float(max(1, src_h - 1))
    interval_px = float(interval_meter) / max(float(meter_per_pixel), 1e-6)
    border_tol = max(2.0, float(image_size) * 0.02)

    for rec in raw_lines:
        cat = normalize_opensatmap_category(rec.get("category", ""))
        if cat not in cat_set:
            continue
        arr = np.asarray(rec.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        pix = np.stack([arr[:, 0] * sx, arr[:, 1] * sy], axis=-1)
        sampled = _resample_polyline(pix, interval_meter=interval_px, max_points=max_points_per_line)
        if sampled.ndim != 2 or sampled.shape[0] < 2:
            continue
        line_type = normalize_line_type(rec.get("line_type", ""))
        if line_type:
            if line_type_set and line_type not in line_type_set:
                line_type = "others" if "others" in line_type_set else ""
        start_is_cut = _is_border_point(sampled[0], image_size=image_size, tol=border_tol)
        end_is_cut = _is_border_point(sampled[-1], image_size=image_size, tol=border_tol)
        lines.append(
            {
                "category": cat,
                "line_type": line_type,
                "start_type": "cut" if start_is_cut else "start",
                "end_type": "cut" if end_is_cut else "end",
                "points": sampled,
            }
        )

    lines.sort(key=lambda d: float(np.linalg.norm(d["points"][0])) if len(d["points"]) > 0 else 1e9)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines
