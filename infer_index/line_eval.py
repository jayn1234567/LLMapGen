from pathlib import Path
import json
import sys
from dataclasses import fields
from dataclasses import asdict

from tqdm import tqdm
from shapely.geometry import LineString

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from .param import Parameter
    from .eval_report_format import LineMatchRes, LineEvalRes
    from .utils import hungarian_match, read_jsonl, convert_QA_data, convert_img_coord_to_meter
except ImportError:
    from param import Parameter
    from eval_report_format import LineMatchRes, LineEvalRes
    from utils import hungarian_match, read_jsonl, convert_QA_data, convert_img_coord_to_meter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mllm.coord_utils import COORD_MODE_PIXEL, convert_payload_text, record_coord_config
from infer_index.map_semantic_eval import evaluate_intersection_iou_records, evaluate_lane_type_records


def line_match_matric(line1: LineString, line2: LineString, buffer) -> float:
    ''' 都已经变成米制单位了
    '''
    poly1 = line1.buffer(buffer)
    poly2 = line2.buffer(buffer)
    return poly1.intersection(poly2).area / poly1.union(poly2).area # 计算交并比


def generate_line_eval_res(gt: list[LineString],
                           pred: list[LineString],
                           gt_match_indices: list[int]) -> LineMatchRes:
    res = LineMatchRes()
    res.gt_line_length_sum = sum([i.length for i in gt])
    res.gt_line_num = len(gt)
    if not pred:
        return res
    res.pred_line_num = len(pred)
    res.pred_line_length_sum = sum([i.length for i in pred])

    res.matched_line_num = len(gt_match_indices)
    res.matched_line_length_sum = sum([gt[i].length for i in gt_match_indices])
    return res


def generate_eval_summary(samples_eval_res: list[LineMatchRes]) -> LineEvalRes:

    # 将单个sample的结果相加
    eval_summary = LineMatchRes()
    eval_summary.sample_num = 0
    eval_summary.valid_string_format = 0
    for one_eval in samples_eval_res:
        for field in fields(LineMatchRes):
            name = field.name
            setattr(eval_summary, name, getattr(eval_summary, name) + getattr(one_eval, name))

    res = LineEvalRes()
    res.instance_pre = safe_div(eval_summary.matched_line_num, eval_summary.pred_line_num)
    res.instance_recall = safe_div(eval_summary.matched_line_num, eval_summary.gt_line_num)
    res.instance_f1 = 2 * res.instance_pre * res.instance_recall / (res.instance_pre + res.instance_recall + 1e-6)

    res.length_pre = safe_div(eval_summary.matched_line_length_sum, eval_summary.pred_line_length_sum)
    res.length_recall = safe_div(eval_summary.matched_line_length_sum, eval_summary.gt_line_length_sum)
    res.length_f1 = 2 * res.length_pre * res.length_recall / (res.length_pre + res.length_recall + 1e-6)

    res.valid_string_format = eval_summary.valid_string_format
    res.samples_num = eval_summary.sample_num

    for field in fields(LineEvalRes):
        name = field.name
        if field.type is float:
            setattr(res, name, round(getattr(res, name), 4)) # 保留四位小数
    return res


def safe_div(num, den):
    return num / den if den else 0.0


def convert_str_2_linestring(data: str, categories=None) -> list[LineString]:
    res = convert_QA_data(data, categories=categories) # 把QA标签和大模型infer结果转换成几何
    res = convert_img_coord_to_meter(res, Parameter.METER_PER_PIXEL) # 像素单位转成米
    res = [LineString(i) for i in res] # 转成linestring
    return res


def _first_text(record: dict, keys: list[str], default: str = "[]") -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return value
    return default


def _text_in_pixel_coords(record: dict, pixel_keys: list[str], raw_keys: list[str]) -> str:
    pixel_text = _first_text(record, pixel_keys, default="")
    if pixel_text:
        return pixel_text
    raw_text = _first_text(record, raw_keys)
    coord_cfg = record_coord_config(record, default_mode=COORD_MODE_PIXEL)
    if coord_cfg["coord_mode"] == COORD_MODE_PIXEL:
        return raw_text
    try:
        return convert_payload_text(
            raw_text,
            coord_cfg["coord_mode"],
            COORD_MODE_PIXEL,
            coord_cfg["patch_width"],
            coord_cfg["patch_height"],
            coord_range=coord_cfg["coord_range"],
            clamp=True,
        )
    except Exception:
        return raw_text


def eval_one_line_sample(one_sample, buffer_size: float, match_thres: float, categories=None) -> LineMatchRes:
    ''' buffer_size: 对linestring取buffer时要取多大
        match_thres: 两个linestring的交并比大于该阈值时才认为他们匹配上了
    '''
    valid_string_format = True
    gt_text = _text_in_pixel_coords(one_sample, ["labels_pixel", "ground_truth_pixel"], ["labels", "ground_truth"])
    pred_text = _text_in_pixel_coords(
        one_sample,
        ["response_pixel", "prediction_json_pixel", "prediction_pixel"],
        ["response", "prediction_json", "prediction"],
    )
    parse_ok = one_sample.get('parse_ok', True)
    try:
        gt = convert_str_2_linestring(gt_text, categories=categories)
    except Exception as e:
        logger.debug(e)
        gt = []
        valid_string_format = False

    try:
        if not parse_ok:
            raise ValueError(one_sample.get('parse_error') or 'prediction parse_ok is false')
        pred = convert_str_2_linestring(pred_text, categories=categories)
    except Exception as e:
        logger.debug(e)
        pred = []
        valid_string_format = False

    gt_match_indices, pred_match_indices = hungarian_match(gt, pred, line_match_matric, buffer_size, match_thres)
    res = generate_line_eval_res(gt, pred, gt_match_indices)
    res.valid_string_format = valid_string_format
    res.sample_num = 1

    # print(res)
    # from vis.plot import plot
    # plot([gt, pred])

    return res


def evaluate_records(
    records,
    meter_per_pixel: float = Parameter.METER_PER_PIXEL,
    buffer_size: float = 1.0,
    match_threshold: float = 0.33,
    include_samples: bool = False,
    categories=None,
    eval_name: str = "Line Evaluation Results",
    **kwargs,
):
    old_meter_per_pixel = Parameter.METER_PER_PIXEL
    Parameter.METER_PER_PIXEL = meter_per_pixel
    try:
        sample_results = []
        sample_payloads = []
        for idx, record in enumerate(records):
            if not isinstance(record, dict) or not any(key in record for key in ("ground_truth", "labels", "ground_truth_pixel", "labels_pixel")):
                continue
            one = eval_one_line_sample(record, buffer_size, match_threshold, categories=categories)
            sample_results.append(one)
            if include_samples:
                payload = asdict(one)
                payload["idx"] = idx
                payload["record_id"] = record.get("record_id", record.get("id", f"sample_{idx}"))
                sample_payloads.append(payload)
        eval_res = generate_eval_summary(sample_results)
        summary = asdict(eval_res)
        summary.update({
            "backend": "infer_index.line_eval",
            "eval_name": eval_name,
            "category_filter": categories,
            "meter_per_pixel": meter_per_pixel,
            "buffer_size": buffer_size,
            "match_threshold": match_threshold,
            "table": eval_res.table_text(title=eval_name),
        })
        if include_samples:
            return {"summary": summary, "samples": sample_payloads}
        return summary
    finally:
        Parameter.METER_PER_PIXEL = old_meter_per_pixel


def line_eval_res_from_summary(summary) -> LineEvalRes:
    if isinstance(summary, LineEvalRes):
        return summary
    res = LineEvalRes()
    if not isinstance(summary, dict):
        return res
    for field in fields(LineEvalRes):
        if field.name in summary:
            setattr(res, field.name, summary[field.name])
    return res


def print_eval_table(summary, logger=None, title: str = None) -> None:
    if isinstance(summary, dict) and isinstance(summary.get("summary"), dict):
        summary = summary["summary"]
    if isinstance(summary, dict) and summary.get("backend") == "infer_index.polygon_iou":
        text = summary.get("table", "")
        if logger is None:
            print(text)
        else:
            logger.info("\n" + text)
        return
    if title is None and isinstance(summary, dict):
        title = summary.get("eval_name")
    line_eval_res_from_summary(summary).show_res(logger, title=title or "Line Evaluation Results")


def format_eval_table(summary, title: str = None) -> str:
    if isinstance(summary, dict) and isinstance(summary.get("summary"), dict):
        summary = summary["summary"]
    if isinstance(summary, dict) and summary.get("backend") == "infer_index.polygon_iou":
        return summary.get("table", "")
    if title is None and isinstance(summary, dict):
        title = summary.get("eval_name")
    return line_eval_res_from_summary(summary).table_text(title=title or "Line Evaluation Results")


def evaluate_lane_intersection_records(
    records,
    meter_per_pixel: float = Parameter.METER_PER_PIXEL,
    buffer_size: float = 1.0,
    match_threshold: float = 0.33,
    intersection_iou_threshold: float = 0.5,
    include_samples: bool = False,
    **kwargs,
):
    lane = evaluate_records(
        records,
        meter_per_pixel=meter_per_pixel,
        buffer_size=buffer_size,
        match_threshold=match_threshold,
        include_samples=include_samples,
        categories="lane",
        eval_name="Lane Evaluation Results",
    )
    intersection = evaluate_intersection_iou_records(
        records,
        iou_threshold=intersection_iou_threshold,
        include_samples=include_samples,
    )
    lane_intersection = evaluate_records(
        records,
        meter_per_pixel=meter_per_pixel,
        buffer_size=buffer_size,
        match_threshold=match_threshold,
        include_samples=include_samples,
        categories="all",
        eval_name="Lane + Intersection Legacy Line Evaluation Results",
    )
    lane_type = evaluate_lane_type_records(
        records,
        meter_per_pixel=meter_per_pixel,
        buffer_size=buffer_size,
        match_threshold=match_threshold,
    )
    intersection_summary = intersection.get("summary", intersection)
    return {
        "lane": lane,
        "intersection": intersection,
        "lane_intersection": lane_intersection,
        "lane_type": lane_type,
        "intersection_type": intersection_summary.get("type_accuracy", {}),
    }


def print_lane_intersection_eval_tables(payload, logger=None) -> None:
    for key in ("lane", "intersection", "lane_intersection"):
        if isinstance(payload, dict) and key in payload:
            print_eval_table(payload[key], logger=logger)
    for key in ("lane_type", "intersection_type"):
        summary = payload.get(key) if isinstance(payload, dict) else None
        if not isinstance(summary, dict):
            continue
        text = summary.get("table", "")
        if not text:
            continue
        if logger is None:
            print(text)
        else:
            logger.info("\n" + text)


def evaluate_one_sample(
    ground_truth: str,
    prediction: str,
    parse_ok: bool = True,
    meter_per_pixel: float = Parameter.METER_PER_PIXEL,
    buffer_size: float = 1.0,
    match_threshold: float = 0.33,
    coord_mode: str = COORD_MODE_PIXEL,
    coord_range: int = 1000,
    patch_size: int = 256,
    **kwargs,
) -> LineMatchRes:
    old_meter_per_pixel = Parameter.METER_PER_PIXEL
    Parameter.METER_PER_PIXEL = meter_per_pixel
    try:
        return eval_one_line_sample(
            {
                "ground_truth": ground_truth,
                "prediction_json": prediction,
                "parse_ok": parse_ok,
                "coord_mode": coord_mode,
                "coord_range": coord_range,
                "patch_size": patch_size,
            },
            buffer_size,
            match_threshold,
            categories=None,
        )
    finally:
        Parameter.METER_PER_PIXEL = old_meter_per_pixel


def line_eval_main(infer_jsonl: str, logger=None) -> LineMatchRes:
    buffer_size = 1
    match_thres = 0.33

    samples = read_jsonl(infer_jsonl)

    samples_eval_res: list[LineMatchRes] = [] # 分别记录各个sample的评估结果
    for one_sample in tqdm(samples, desc='Evaluating samples'):
        # 计算单个sample的评估结果
        samples_eval_res.append(eval_one_line_sample(one_sample, buffer_size, match_thres))

    # 汇总计算实例级和里程级准召及F1-score
    line_eval_res = generate_eval_summary(samples_eval_res)
    line_eval_res.show_res(logger)
    return line_eval_res



if __name__ == '__main__':
    jsonl = r'd:\实验记录\simplified_linestring_10000数据集\infer_result\20260417-232452.jsonl'
    # jsonl = r'd:\实验记录\34000数据集-简化提示词\infer_result\20260420-024444.jsonl'
    # jsonl = r'd:\实验记录\34000数据集-中文提示词\infer_result\20260419-233556.jsonl'
    jsonl = str(Path(jsonl))
    line_eval_main(jsonl, logger)
    print(jsonl)
