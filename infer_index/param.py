from dataclasses import dataclass


@dataclass
class Parameter:
    INTERPOLATE_DIS = 5 # 单位，米，表示对lane均匀差值的距离
    METER_PER_PIXEL = 0.2 # TIFF图中每个像素表示多少米
