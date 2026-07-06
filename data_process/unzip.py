import os
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor

def extract_tar_gz(file_path):
    """
    解压单个 .tar.gz 文件到与压缩包同名的文件夹内，并在解压完成后删除文件
    :param file_path: 文件路径
    """
    try:
        # 获取文件名（不带扩展名）
        base_name = os.path.splitext(os.path.splitext(file_path)[0])[0]

        # 创建与压缩包同名的文件夹
        target_dir = os.path.join(os.path.dirname(file_path), base_name)
        os.makedirs(target_dir, exist_ok=True)

        # 解压到目标文件夹
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        print(f"解压完成: {file_path} -> {target_dir}")

        # 解压完成后删除文件
        os.remove(file_path)
        print(f"已删除: {file_path}")
    except Exception as e:
        print(f"处理失败: {file_path} - {e}")

def find_tar_gz_files(folder_path):
    """
    查找文件夹中所有 .tar.gz 文件
    :param folder_path: 文件夹路径
    :return: .tar.gz 文件列表
    """
    tar_gz_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".tar.gz"):
                tar_gz_files.append(os.path.join(root, file))
    return tar_gz_files

def main(folder_path, max_workers=4):
    """
    主函数
    :param folder_path: 文件夹路径
    :param max_workers: 最大线程数
    """
    # 查找所有 .tar.gz 文件
    tar_gz_files = find_tar_gz_files(folder_path)
    if not tar_gz_files:
        print("未找到 .tar.gz 文件")
        return

    # 使用线程池解压文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(extract_tar_gz, tar_gz_files)

if __name__ == "__main__":
    # 指定文件夹路径
    folder_path = r"D:\Workspace\基模型\训练\Lane数据集\rc_airflow_task_0426_1639\train"  # 替换为你的文件夹路径

    # 调用主函数
    main(folder_path)
