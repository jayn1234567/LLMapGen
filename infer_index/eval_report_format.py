from dataclasses import dataclass


@dataclass
class LineMatchRes:
    gt_line_num: int = 0
    gt_line_length_sum: float = 0

    pred_line_num: int = 0
    pred_line_length_sum: float = 0
    
    matched_line_num: int = 0
    matched_line_length_sum: float = 0
    
    sample_num: int = 1 # 对应的样本数量
    valid_string_format: bool = True # 该sample的格式是否可以被解析
    

@dataclass
class LineEvalRes:
    instance_pre: float = 0.0
    instance_recall: float = 0.0
    instance_f1: float = 0.0
    length_pre: float = 0.0
    length_recall: float = 0.0
    length_f1: float = 0.0

    valid_string_format: int = 0 # 有多少样本格式是可被解析的，越高越好
    samples_num: int = 0

    def show_res(self, logger=None):
        def out(msg):
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

        out("=" * 58)
        out(f"{' Line Evaluation Results ':^58}")
        out("=" * 58)
        out(f"{'Metric':<18} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        out("-" * 58)
        out(f"{'Instance Level':<18} {self.instance_pre:<12.4f} {self.instance_recall:<12.4f} {self.instance_f1:<12.4f}")
        out(f"{'Length Level':<18} {self.length_pre:<12.4f} {self.length_recall:<12.4f} {self.length_f1:<12.4f}")
        out("=" * 58)

        prec = self.valid_string_format / self.samples_num if self.samples_num > 1e-6 else 0

        out(f'格式合法的推理结果占比: {prec:.4f}({self.valid_string_format}/{self.samples_num})')
