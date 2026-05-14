import  os
import git
from dataset_creator import DatasetCreator
import concurrent.futures

def find_parent_folders_of_inter_patch_tif(root_folder):
    parent_folders = []

    # 递归遍历文件夹
    for root, dirs, files in os.walk(root_folder):
        for dir_name in dirs:
            if dir_name == 'inter_patch_tif':
                # 获取父文件夹路径
                parent_folders.append(root)
    return parent_folders

def get_current_commit_id():
    try:
        # 获取当前 Git 仓库
        repo = git.Repo(search_parent_directories=True)
        
        # 获取当前分支的最新 commit ID
        commit_id = repo.head.commit.hexsha
        
        return commit_id
    except git.exc.InvalidGitRepositoryError:
        return "当前目录不是一个 Git 仓库"
    except Exception as e:
        return f"发生错误: {e}"

def process_folder(folder, dataset_output_dir, cur_data_type):
    geojson_path = os.path.join(folder, 'label_check_crop', 'Lane.geojson')
    tiff_path = os.path.join(folder, 'inter_patch_tif', '0_inter.tif')
    mask_tiff_path = os.path.join(folder, 'patch_tif', '0_edit_poly.tif')
    tool = DatasetCreator(geojson_path, tiff_path, mask_tiff_path, dataset_output_dir, cur_data_type, chunk_size=256, max_total_pts=0)
    return tool.run()

if __name__ == "__main__":
    import json
    from tqdm import tqdm
    # 输入文件夹路径
    input_folder = r"D:\Workspace\基模型\数据集\Lane数据集\rc_airflow_task_0426_1639\raw_input"
    # 查找所有 'inter_patch_tif' 的父文件夹路径
    parent_folders = find_parent_folders_of_inter_patch_tif(input_folder)
    output_folder = r"D:\Workspace\基模型\数据集\Lane数据集\rc_airflow_task_0426_1639\output\special_token"
    dataset_output_dir  = output_folder
    train_output_file = os.path.join(output_folder, 'train.jsonl')
    val_output_file = os.path.join(output_folder, 'val.jsonl')
    info_output_file = os.path.join(output_folder, 'dataset_info.json')

    train_res, val_res = [], []
    empty_train_res_cnt, empty_val_res_cnt = 0, 0
    patch_cnt = 0
    all_message_cnt = 0
    empty_message_cnt = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for idx, folder in enumerate(parent_folders):
            if idx < 0.9 * len(parent_folders):
                cur_data_type = 'train'
            else:
                cur_data_type = 'val'
        # process_folder(folder, dataset_output_dir, cur_data_type)
            futures.append(executor.submit(process_folder, folder, dataset_output_dir, cur_data_type))
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(parent_folders)):
            cur_res = future.result()
            cur_empty_res, cur_nonempty_res = [], []
            for item in cur_res:
                line_cnt = item['conversations'][1]['value']
                if len(line_cnt) < 1:
                    cur_empty_res.append(item)
                else:
                    cur_nonempty_res.append(item)

            if 'train' in item['image']:
                train_res.extend(cur_nonempty_res)
            else:
                val_res.extend(cur_nonempty_res)

            if empty_train_res_cnt < 0.1 * len(train_res):
                train_res.extend(cur_empty_res)
                empty_train_res_cnt += len(cur_empty_res)
            elif empty_val_res_cnt < 0.1 * len(val_res):
                val_res.extend(cur_empty_res)
                empty_val_res_cnt += len(cur_empty_res)
            patch_cnt += 1
            

    with open(train_output_file, 'w', encoding='utf-8') as f:
        for item in train_res:
            # 每行写入一个 JSON 对象
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    with open(val_output_file, 'w', encoding='utf-8') as f:
        for item in val_res:
            # 每行写入一个 JSON 对象
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    if not len(val_res):
        val_empty_message_ratio = 0.0
    else:
        val_empty_message_ratio = empty_val_res_cnt / len(val_res)
    with open(info_output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'patch_size': patch_cnt,
            'all_dataset_cnt': len(train_res)+len(val_res),
            'train_all_dataset_size': len(train_res),
            'train_empty_dataset_size': empty_train_res_cnt,
            'train_empty_message_ratio': empty_train_res_cnt / len(train_res),
            'val_all_dataset_size': len(val_res),
            'val_empty_dataset_size': empty_val_res_cnt,
            'val_empty_message_ratio': val_empty_message_ratio,
            'git_branch': 'nyj_data_513302892c03',
            'git_commit': get_current_commit_id()
        }, f, ensure_ascii=False)
