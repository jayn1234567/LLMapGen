import re
import ast
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from scipy.optimize import linear_sum_assignment


def read_jsonl(path):
    # 兼容 JSONL、JSON array，以及 state-update 的 {"patch_results": [...]}。
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("共读取 0 条数据")
        return []

    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            data_list = payload
        elif isinstance(payload, dict) and isinstance(payload.get("patch_results"), list):
            data_list = payload["patch_results"]
        else:
            data_list = [payload]
    except json.JSONDecodeError:
        data_list = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                data_list.append(json.loads(line))

    print(f"共读取 {len(data_list)} 条数据")
    return data_list


def hungarian_match(gt_list, pred_list, metric_func, buffer_size: float, match_thres: float):
    num_gt = len(gt_list)
    num_pred = len(pred_list)
    
    if num_gt == 0 or num_pred == 0:
        return [], []

    cost_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
    
    def compute_row(gt_id):
        gt = gt_list[gt_id]
        row = np.zeros(num_pred, dtype=np.float32)
        for pred_id in range(num_pred):
            row[pred_id] = metric_func(gt, pred_list[pred_id], buffer_size)
        return gt_id, row

    # 线程池：轻量、无fork、不与外部多进程冲突 ✅
    with ThreadPoolExecutor(max_workers=8) as executor:
        for gt_id, row in executor.map(compute_row, range(num_gt)):
            cost_matrix[gt_id] = row

    # 匈牙利最优匹配
    gt_indices, pred_indices = linear_sum_assignment(-cost_matrix)

    gt_res, pred_res = [], []
    # 滤除匹配数值不满足阈值的
    for idx in range(len(gt_indices)):
        if cost_matrix[gt_indices[idx], pred_indices[idx]] < match_thres:
            continue
        gt_res.append(gt_indices[idx])
        pred_res.append(pred_indices[idx])

    return gt_res, pred_res


def _extract_json_payload(data: str) -> str:
    text = str(data or "").strip()
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return text
    start = min(starts)
    stack = []
    in_string = False
    escape = False
    pairs = {"{": "}", "[": "]"}
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return text[start:idx + 1]
    return text[start:]


def _load_prediction_payload(data):
    if isinstance(data, (list, dict)):
        return data
    text = _extract_json_payload(data)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r'(\w+):', r'"\1":', text)
        return ast.literal_eval(text)


def _iter_line_items(payload):
    if isinstance(payload, dict):
        payload = payload.get("lines", payload.get("road_map", payload.get("centerlines", [])))
    if not isinstance(payload, list):
        return []
    return payload


def _is_centerline_item(item):
    if not isinstance(item, dict):
        return True
    category = str(item.get("category", "centerline")).strip().lower()
    return category in {"centerline", "center_line", "lane", ""}


def convert_QA_data(data: str) -> list[list]:
    ''' 将QA数据集/大模型的infer结果，转换成点的列表

        data: 单条QA数据集或大模型infer结果
    '''
    json_res = []
    payload = _load_prediction_payload(data)
    for one_sample in _iter_line_items(payload):
        if not _is_centerline_item(one_sample):
            continue
        try:
            json_res.append(one_sample['points'])
        except Exception as e:
            json_res.append(one_sample)
    
    # 定义正则表达式模式，提取数字
    res = []
    for sublist in json_res:
        cur_res = []
        for pairs in sublist:
            cur_pairs = pairs
            if pairs and isinstance(pairs[0], str):
                cur_pairs = [int(re.search(r'\d+', pt).group()) for pt in pairs] # 提取出每个字符串中的数字，并转换为整数
            elif pairs and not isinstance(pairs[0], int):
                cur_pairs = pairs
            cur_res.append(cur_pairs)
        res.append(cur_res)
    return res


def convert_img_coord_to_meter(coords: list[list], resolution) -> list[list]:
    res = []
    for item in coords:
        new_item = []
        for onept in item:
            new_item.append([i * resolution for i in onept])
        res.append(new_item)
    return res
