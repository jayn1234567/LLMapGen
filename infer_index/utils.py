import re
import ast
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from scipy.optimize import linear_sum_assignment


def read_jsonl(path):
    # 读取 jsonl 文件
    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # 去掉空行、换行符
            line = line.strip()
            if not line:
                continue
            # 解析每行 JSON
            json_obj = json.loads(line)
            data_list.append(json_obj)

    # 查看结果
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


def convert_QA_data(data: str) -> list[list]:
    ''' 将QA数据集/大模型的infer结果，转换成点的列表

        data: 单条QA数据集或大模型infer结果
    '''
    json_res = []
    data = re.sub(r'(\w+):', r'"\1":', data) # 把所有的  单词+冒号  替换成 "单词"+冒号
    data = ast.literal_eval(data)
    for one_sample in data:
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
