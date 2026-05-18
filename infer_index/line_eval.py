from pathlib import Path
from dataclasses import fields

from tqdm import tqdm
from loguru import logger
from shapely.geometry import LineString

from param import Parameter
from eval_report_format import LineMatchRes, LineEvalRes
from utils import hungarian_match, read_jsonl, convert_QA_data, convert_img_coord_to_meter


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
    for one_eval in samples_eval_res:
        for field in fields(LineMatchRes):
            name = field.name
            setattr(eval_summary, name, getattr(eval_summary, name) + getattr(one_eval, name))

    res = LineEvalRes()
    res.instance_pre = eval_summary.matched_line_num / eval_summary.pred_line_num
    res.instance_recall = eval_summary.matched_line_num / eval_summary.gt_line_num
    res.instance_f1 = 2 * res.instance_pre * res.instance_recall / (res.instance_pre + res.instance_recall + 1e-6)

    res.length_pre = eval_summary.matched_line_length_sum / eval_summary.pred_line_length_sum
    res.length_recall = eval_summary.matched_line_length_sum / eval_summary.gt_line_length_sum
    res.length_f1 = 2 * res.length_pre * res.length_recall / (res.length_pre + res.length_recall + 1e-6)

    res.valid_string_format = eval_summary.valid_string_format
    res.samples_num = eval_summary.sample_num

    for field in fields(LineEvalRes):
        name = field.name
        if field.type is float:
            setattr(res, name, round(getattr(res, name), 4)) # 保留四位小数
    return res


def convert_str_2_linestring(data: str) -> list[LineString]:
    res = convert_QA_data(data) # 把QA标签和大模型infer结果转换成几何
    res = convert_img_coord_to_meter(res, Parameter.METER_PER_PIXEL) # 像素单位转成米
    res = [LineString(i) for i in res] # 转成linestring
    return res



def eval_one_line_sample(one_sample, buffer_size: float, match_thres: float) -> LineMatchRes:
    ''' buffer_size: 对linestring取buffer时要取多大
        match_thres: 两个linestring的交并比大于该阈值时才认为他们匹配上了
    '''
    valid_string_format = True
    gt = convert_str_2_linestring(one_sample['labels'])
    try:
        pred = convert_str_2_linestring(one_sample['response'])
    except Exception as e:
        logger.error(e)
        pred = []
        valid_string_format = False
        
    gt_match_indices, pred_match_indices = hungarian_match(gt, pred, line_match_matric, buffer_size, match_thres)
    res = generate_line_eval_res(gt, pred, gt_match_indices)
    res.valid_string_format = valid_string_format

    # print(res)
    # from vis.plot import plot
    # plot([gt, pred])

    return res


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
