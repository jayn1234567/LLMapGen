import os
import math
import rasterio
from shapely.geometry import LineString
from PIL import Image

def visualize_tiff_window_lines(chunk_image_path, window_polygon, pixel_lines, output_dir, i, j):
    """
    可视化TIFF图、窗口和裁剪后的线。
    """
    # 读取TIFF图
    with rasterio.open(chunk_image_path) as src:
        chunk_image = src.read()

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制TIFF图
    ax.imshow(chunk_image[0], cmap='gray')  # 假设是单波段图像

    # 绘制窗口多边形
    min_x, min_y, max_x, max_y = window_polygon.bounds
    windowPolygon = LineString([(min_x, min_y), (max_x, min_y),
                                (max_x, max_y), (min_x, max_y), (min_x, min_y)])
    ax.plot(*windowPolygon.xy, color='red', label='Window Polygon')

    # 绘制裁剪后的线
    for line in pixel_lines:
        if isinstance(line, list):
            line = line[0]
        # 将y坐标取反，以匹配matplotlib的坐标系
        x_coords = [coord[0] for coord in line.coords]
        y_coords = [-coord[1] for coord in line.coords]  # y坐标取反
        ax.plot(x_coords, y_coords, color='blue', label='Cropped Line')

    # 设置坐标轴范围
    ax.set_xlim(0, chunk_image.shape[2])
    ax.set_ylim(-chunk_image.shape[1], 0)  # y轴范围取反

    # 添加图例
    ax.legend()

    # 保存图像
    output_image_path = os.path.join(output_dir, f"visualization_{i}_{j}.png")
    plt.savefig(output_image_path)
    plt.close()


import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_lines(image_path, lines_data):
    """
    可视化线信息
    :param image_path: 图像路径
    :param lines_data: 线信息列表
    """
    # 读取图像
    image_pil = Image.open(image_path)  # cv2 imread函数无法处理中文路径
    image_rgb = np.array(image_pil)
    if image_rgb is None:
        print(f"无法读取图像: {image_path}")
        return

    # 创建一个空白画布用于绘制线
    canvas = image_rgb.copy()

    # 遍历每条线
    for idx, line in enumerate(lines_data):
        category = line.get("category", "unknown")
        start_type = line.get("start_type", "unknown")
        end_type = line.get("end_type", "unknown")
        points = line.get("points", [])

        # 绘制线
        if len(points) >= 2:
            for i in range(len(points) - 1):
                p1 = tuple(map(int, points[i]))
                p2 = tuple(map(int, points[i + 1]))
                cv2.line(canvas, p1, p2, (0, 255, 0), 2)
                # 在每段线上添加箭头
                cv2.arrowedLine(canvas, p1, p2, (0, 255, 0), 2, tipLength=0.5)

            # 标记起点和终点
            start_point = tuple(map(int, points[0]))
            end_point = tuple(map(int, points[-1]))

            cv2.circle(canvas, start_point, 5, (255, 0, 0), -1)  # 蓝色标记起点
            cv2.circle(canvas, end_point, 5, (255, 0, 0), -1)  # 蓝色标记终点
            if start_type == "cut":
                cv2.putText(canvas, f'c_{idx + 1}', start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 白色文本标记
            else:
                cv2.putText(canvas, f's_{idx + 1}', start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 白色文本标记

            if end_type == "cut":
                cv2.putText(canvas, f'c_{idx + 1}', end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 白色文本标记
            else:
                cv2.putText(canvas, f'e_{idx + 1}', end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 白色文本标记
            mid_point = tuple(map(int, points[len(points) // 2]))
            dis = math.sqrt(start_point[0] ** 2 + start_point[1] ** 2)
            cv2.putText(canvas, f'{dis:.2f}', mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # 白色文本标记

    # 显示图像
    cv2.imshow("Image", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_json_file(json_file):
    """
    处理 JSON 文件
    :param json_file: JSON 文件路径
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        image_path = os.path.join(r'D:\Workspace\基模型\训练\生产数据集\output', item.get("images", ""))
        messages = item.get("messages", [])

        # 找到 assistant 的 content
        assistant_content = None
        for message in messages:
            if message.get("role") == "assistant":
                assistant_content = message.get("content", "")
                break

        if assistant_content:
            # 解析 assistant 的 content
            try:
                assistant_data = json.loads(assistant_content)
                lines_data = assistant_data.get("lines", [])
                if lines_data:
                    visualize_lines(image_path, lines_data)
            except json.JSONDecodeError:
                print("无法解析 assistant 的 content")

if __name__ == "__main__":
    # JSON 文件路径
    json_file = r'D:\Workspace\基模型\训练\生产数据集\output\output.json'

    # 处理 JSON 文件
    process_json_file(json_file)
