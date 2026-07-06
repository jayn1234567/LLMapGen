import json
import re
import math
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.transform import from_origin
from rasterio.windows import Window
from collections import defaultdict
import os
from shapely.geometry import LineString, Polygon, MultiLineString
from pyproj import Transformer
import matplotlib.pyplot as plt
from validate import visualize_tiff_window_lines

from constants import EndpointType, Role

CHUNK_SIZE = 512
def pad_to_multiple_of_CHUNK_SIZE(image):
    """
    将图像补全为CHUNK_SIZE的倍数。
    """
    height, width = image.shape[1], image.shape[2]
    pad_width = CHUNK_SIZE - (width % CHUNK_SIZE) if width % CHUNK_SIZE != 0 else 0
    pad_height = CHUNK_SIZE - (height % CHUNK_SIZE) if height % CHUNK_SIZE != 0 else 0

    # 使用np.pad进行补全，补全值为0
    padded_image = np.pad(image, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    return padded_image


class DatasetCreator:
    def __init__(self, geojson_path, original_tiff_path, mask_tiff_path, dataset_output_dir, cur_data_type='train', chunk_size=CHUNK_SIZE, debug=False, max_total_pts=None):
        self.geojson_path = geojson_path
        self.original_tiff_path = original_tiff_path
        self.mask_tiff_path = mask_tiff_path
        self.chunk_size = chunk_size
        self.patch_id = self.get_folder_with_many_numbers(original_tiff_path)
        self.debug = debug
        self.dataset_output_dir = dataset_output_dir  # 总的输出目录
        self.output_dir = os.path.join(dataset_output_dir, cur_data_type, self.patch_id)  # 当前处理类型下的输出目录
        self.max_total_pts = max_total_pts  # patch内最大点数，超出则抛弃当前patch
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def get_folder_with_many_numbers(self, path):
        # 分割路径为各个部分
        parts = path.split(os.sep)
        # 遍历每个部分，检查是否包含多个数字
        for part in parts:
            # 使用正则表达式匹配数字
            numbers = re.findall(r'\d{6}', part)
            if len(numbers) > 1:  # 如果包含多个数字
                return part
        return None  # 如果没有找到符合条件的文件夹

    def read_tiff_metadata(self, original_path, mask_path = '', output_dir = ''):
        # 读取原TIFF图和Mask TIFF图
        with rasterio.open(original_path) as src:
            original = src.read()
            meta = src.meta
            transform = src.transform  # tiff图左上角为原点

        if mask_path:
            with rasterio.open(mask_path) as src:
                mask = src.read()
            # 将Mask二值化，假设Mask区域为1，外部为0
            mask = (mask > 0).astype(np.uint8)
            masked_image = np.where(mask == 1, original, 0)
        else:
            print('mask为空')
            mask = original
            masked_image = np.where(mask > 0, original, 0)

        # 应用Mask：将Mask区域外的原TIFF图置黑
        image = pad_to_multiple_of_CHUNK_SIZE(masked_image)
        meta.update({
                'width': image.shape[2],
                'height': image.shape[1]
            })

        # 保存Mask后的TIFF图
        meta.update({"dtype": "uint8"})
        if self.debug:
            masked_tiff_path = os.path.join(output_dir, "masked_image.tiff")
            with rasterio.open(masked_tiff_path, 'w', **meta) as dst:
                dst.write(masked_image)
            masked_tiff_path = os.path.join(output_dir, "masked_paded_image.tiff")
            with rasterio.open(masked_tiff_path, 'w', **meta) as dst:
                dst.write(image)
        return image, meta, transform
    

    def load_lane(self, geojson_path, meta):
        # 读取GeoJSON文件
        gdf = gpd.read_file(geojson_path)
        # 将GeoJSON的坐标系转换为TIFF的坐标系
        gdf = gdf.to_crs(meta['crs'])
        if self.debug:
            utm_path = os.path.join(self.output_dir, "utm_lane.geojson")
            gdf.to_file(utm_path, driver='GeoJSON')
        return gdf

    def split_tiff(self, masked_image, i, j, transform, meta, chunk_size=CHUNK_SIZE):
        # 切分TIFF图
        chunk = masked_image[:, j:j+chunk_size, i:i+chunk_size]
        # 检查窗口内是否全为黑色
        if np.all(chunk == 0):
            return None, None  # 如果全为黑色，跳过该窗口
        # 像素坐标到UTM坐标系的转换器
        window_transform = from_origin(transform.xoff + i * transform.a,
                                    transform.yoff + j * transform.e,
                                    transform.a, abs(transform.e))  # 传入的是x/y方向缩放因子，所以需要取绝对值

        output_image_path = os.path.join(self.output_dir, f"chunk_image_{i}_{j}.tiff")
        chunk_meta = meta.copy()
        chunk_meta.update({
            'transform': window_transform,
            'width': chunk.shape[2],
            'height': chunk.shape[1]
        })
        if self.debug:
            with rasterio.open(output_image_path, 'w', **chunk_meta) as dst:
                dst.write(chunk)
        return chunk, window_transform

    def get_endpoint_type(self, original_line, chunk_line):
        """
        判断线的起点和终点类型
        original_line: 原始线
        chunk_line: 切分后的线
        """
        original_start = original_line.coords[0]
        original_end = original_line.coords[-1]
        chunk_start = chunk_line.coords[0]
        chunk_end = chunk_line.coords[-1]

        # 判断起点类型
        if chunk_start == original_start:
            start_type = EndpointType.START
        elif chunk_start == original_end:
            start_type = EndpointType.END
        else:
            start_type = EndpointType.CUT

        # 判断终点类型
        if chunk_end == original_start:
            end_type = EndpointType.START
        elif chunk_end == original_end:
            end_type = EndpointType.END
        else:
            end_type = EndpointType.CUT
        return start_type.value, end_type.value
        
    def get_lane_in_window(self, gdf, transform, i, j, chunk_size=CHUNK_SIZE):
        # 获取窗口内的lane
        chunk_rows = []
        for _, row in gdf.iterrows():
            line = row['geometry']
            # 将窗口的像素坐标转换到UTM坐标系
            window_polygon = Polygon([
                transform * (i, j),
                transform * (i + chunk_size, j),
                transform * (i + chunk_size, j + chunk_size),
                transform * (i, j + chunk_size),
                transform * (i, j)
            ])
            if line.intersects(window_polygon):
                chunk_rows.append(row)
        return chunk_rows, window_polygon

    def convert2pixel(self, coords, transform):
        pixel_coords = []
        for coord in coords:
            x, y, _ = coord
            # 将投影坐标转换到像素坐标
            pixel_x, pixel_y = ~transform * (x, y)
            abs_x, abs_y = abs(round(pixel_x)), abs(round(pixel_y))
            if abs_x >= self.chunk_size:
                abs_x = self.chunk_size - 1
            if abs_y >= self.chunk_size:
                abs_y = self.chunk_size - 1
            pixel_coords.append((abs_x, abs_y))
        return pixel_coords

    def split_lane(self, gdf, transform, i, j, chunk_size, window_transform, meta):
        orgin_line_id_2_pixel_lines = defaultdict(list)  # key: 线的ID，value: 切分后的线
        chunk_rows, window_polygon = self.get_lane_in_window(gdf, transform, i, j, chunk_size)
        if not chunk_rows:
            return orgin_line_id_2_pixel_lines, None, window_polygon

        # 创建新的GeoDataFrame保存切分后的线
        chunk_gdf = gpd.GeoDataFrame(chunk_rows, crs=gdf.crs)
        chunk_gdf['Id_copy'] = chunk_gdf['Id']
        chunk_gdf.set_index('Id_copy', inplace=True)  # 按照Id索引
        # 将线转换到像素坐标系
        pixel_lines = []
        for _, cur_row in chunk_gdf.iterrows():
            line = cur_row['geometry']
            line_intersect = line.intersection(window_polygon)
            if line_intersect.is_empty:
                continue
            if isinstance(line_intersect, LineString):
                pixel_coords = self.convert2pixel(line_intersect.coords, window_transform)
                pixel_line = LineString(pixel_coords)
                pixel_lines.append(pixel_line)
                orgin_line_id_2_pixel_lines[cur_row.Id].append(pixel_line)  # name就是id
            elif isinstance(line_intersect, MultiLineString):
                lines = [LineString(self.convert2pixel(part.coords, window_transform)) for part in line_intersect.geoms]
                pixel_lines.extend(lines)
                orgin_line_id_2_pixel_lines[cur_row.Id].extend(lines)
        
        # 保存切分后的GeoJSON
        if self.debug:
            # 创建新的GeoDataFrame保存像素坐标系下的线
            pixel_gdf = gpd.GeoDataFrame({'geometry': pixel_lines}, crs=meta['crs'])
            output_geojson_path = os.path.join(self.output_dir, f"chunk_geojson_{i}_{j}.geojson")
            pixel_gdf.to_file(output_geojson_path, driver='GeoJSON')
        return orgin_line_id_2_pixel_lines, chunk_gdf, window_polygon

    def sort_key(self, line):
        points = line
        if not points:
            return (float('inf'), ) # 如果没有点，返回无穷大
        # 计算第一个点到原点的距离
        first_point_distance = math.sqrt(points[0][0]**2 + points[0][1]**2)
        # 如果有多个点，依次计算距离
        for point in points[1:]:
            current_distance = math.sqrt(point[0]**2 + point[1]**2)
            if current_distance != first_point_distance:
                return (first_point_distance, current_distance)
        return (first_point_distance, )
    
    def generate_messages(self, orgin_line_id_2_pixel_lines, chunk_gdf, max_total_pts=None):
        def get_json_content(orgin_line_id_2_pixel_lines):
            res = []
            for orgin_line_id, pixel_lines in orgin_line_id_2_pixel_lines.items():
                origin_line = chunk_gdf.loc[orgin_line_id]
                for pixel_line in pixel_lines:
                    start_type, end_type = self.get_endpoint_type(origin_line['geometry'], pixel_line)
                    # 首尾点像素距离小于1，则认为是空线
                    if abs(math.sqrt((pixel_line.coords[0][0] - pixel_line.coords[-1][0]) ** 2 + (pixel_line.coords[0][1] - pixel_line.coords[-1][1]) ** 2)) < 1:
                        continue
                    if math.sqrt(pixel_line.coords[0][0] ** 2 + pixel_line.coords[0][1] ** 2) \
                        > math.sqrt(pixel_line.coords[-1][0] ** 2 + pixel_line.coords[-1][1] ** 2):
                        pixel_line = LineString(list(pixel_line.coords)[::-1])
                    res.append([[int(x), int(y)] for x, y in pixel_line.coords])
            return res
        assistant_content = get_json_content(orgin_line_id_2_pixel_lines)
        sorted_content = sorted(assistant_content, key=self.sort_key)
        sorted_content.reverse()  # 反转列表，使距离较小的在前面
        total_pts = sum(len(sublist) for sublist in assistant_content)
        if max_total_pts and total_pts > max_total_pts:
            return []
        # sorted_content = [[item[0], item[-1]] for _, item in enumerate(sorted_content)]

        for i, sublist in enumerate(sorted_content):
            for j, item in enumerate(sublist):
                sorted_content[i][j] = ['<|'+str(item[0])+'|>', '<|'+str(item[1])+'|>']

        messages = [
            # {
            #     'from': Role.User.value,
            #     'value': Role.UserPrompt1.value
            # },
            # {
            #     'from': Role.Assistant.value,
            #     'value': json.dumps(len(sorted_content))
            # },
            {
                'from': Role.User.value,
                'value': Role.UserPrompt2.value
            },
            {
                'from': Role.Assistant.value,
                'value': json.dumps(sorted_content)
            }
        ]
        return messages

    def preprocess_lane_by_douglas(self, gdf, tolerance=0.5):
        """
        对 GeoDataFrame 中的车道线几何应用道格拉斯-普克算法进行简化。

        Parameters:
            gdf (geopandas.GeoDataFrame): 包含车道线几何的 GeoDataFrame，几何列名为 'geometry'。
            tolerance (float): 简化容差（与坐标系单位一致，例如米）。值越大简化越剧烈。

        Returns:
            geopandas.GeoDataFrame: 简化后的 GeoDataFrame。
        """
        # 对每一行几何进行简化，保留拓扑结构
        gdf['geometry'] = gdf['geometry'].apply(
            lambda geom: geom.simplify(tolerance, preserve_topology=True)
        )
        return gdf

    def run(self):
        masked_image, meta, transform = self.read_tiff_metadata(self.original_tiff_path, self.mask_tiff_path, self.output_dir)
        # 切分TIFF图和GeoJSON中的线
        raw_gdf = self.load_lane(self.geojson_path, meta)
        gdf = self.preprocess_lane_by_douglas(raw_gdf)

        width, height = meta['width'], meta['height'] 
        img_cnt = 0
        res = []
        for i in range(0, width, self.chunk_size):
            for j in range(0, height, self.chunk_size):
                chunk_tiff, window_transform = self.split_tiff(masked_image, i, j, transform, meta, self.chunk_size)
                if chunk_tiff is None:
                    continue
                id = f'r{i//self.chunk_size}_c{j//self.chunk_size}_p{img_cnt:02d}'
                output_path = os.path.join(self.output_dir, id+'.png')
                chunk_tiff = np.transpose(chunk_tiff, (1, 2, 0))
                if not os.path.exists(output_path):
                    plt.imsave(output_path, chunk_tiff)
                orgin_line_id_2_pixel_lines, chunk_gdf, window_polygon = self.split_lane(gdf, transform, i, j, self.chunk_size, window_transform, meta)
                if self.debug:
                    output_image_path = os.path.join(self.output_dir, f"chunk_image_{i}_{j}.tiff")
                    visualize_tiff_window_lines(output_image_path, window_polygon, orgin_line_id_2_pixel_lines.values(), self.output_dir, i, j)
                messages = self.generate_messages(orgin_line_id_2_pixel_lines, chunk_gdf, self.max_total_pts)
                if not messages:
                    continue
                img_cnt += 1
                rel_path = os.path.relpath(output_path, self.dataset_output_dir)
                res.append({
                        'id': f'{self.patch_id}_{id}',
                        'conversations': messages,
                        'image': rel_path.replace('\\', '/')
                    }
                )
        output_file = os.path.join(self.output_dir, 'output.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        print(f"处理后的图像和GeoJSON已保存至 {self.output_dir}")
        return res

if __name__ == '__main__':
    # 示例使用
    import os
    folder = r'D:\Workspace\基模型\训练\Lane数据集\A0_2532083105204964_1741093085335_82269825\patch_tif'
    original_path = os.path.join(folder, '..', 'inter_patch_tif', '0_inter.tif')
    mask_path = os.path.join(folder, '0_edit_poly.tif')
    geojson_path = r"D:\Workspace\基模型\训练\Lane数据集\A0_2532083105204964_1741093085335_82269825\label_check_crop\Lane.geojson"
    output_path = r'D:\Workspace\基模型\训练\生产数据集\output'
    os.makedirs(output_path, exist_ok=True)
    tool = DatasetCreator(geojson_path, original_path, mask_path, output_path, debug=True)
    tool.run()
