import cv2
import numpy as np
from multiprocessing import Pool, Manager, Queue
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import torch
import os
from tqdm import tqdm

def is_symmetric(image_path, threshold=0.06):
    # 加载图像
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        b, g, r, alpha = cv2.split(img)
        img = alpha
        if img is None:
            print("Image not found or unable to open")
            return False
        # 获得图像的宽度和高度
        height, width = img.shape
        center_x, center_y = width // 2, height // 2
        vertical_scores = []
        horizontal_scores = []
        # for angle in range(0, 360, 5):
        for angle in [0]:
            # 旋转图像至给定角度
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            rotated_image = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REFLECT)
            img = rotated_image
            
            # 检查垂直对称
            vertical_center = width // 2
            left_side = img[:, :vertical_center]
            right_side = img[:, vertical_center:]
            right_side_flipped = cv2.flip(right_side, 1)
            vertical_difference = cv2.absdiff(left_side, right_side_flipped)
            vertical_score = np.sum(vertical_difference)
            vertical_scores.append(vertical_score)
            
            # 检查水平对称
            horizontal_center = height // 2
            top_side = img[:horizontal_center, :]
            bottom_side = img[horizontal_center:, :]
            bottom_side_flipped = cv2.flip(bottom_side, 0)
            horizontal_difference = cv2.absdiff(top_side, bottom_side_flipped)
            horizontal_score = np.sum(horizontal_difference)
            horizontal_scores.append(horizontal_score)
            
        vertical_score = min(vertical_scores)
        horizontal_score = min(horizontal_scores)

        # 设置阈值来判断是否对称
        # vertical_threshold = threshold * height * vertical_center * 255  # 调整阈值
        # horizontal_threshold = threshold * width * horizontal_center * 255  # 调整阈值
        vertical_threshold = threshold * np.sum(img)
        horizontal_threshold = threshold * np.sum(img)

        # print(vertical_score, vertical_threshold, sum(img) )
        
        is_vertical_symmetric = vertical_score < vertical_threshold
        is_horizontal_symmetric = horizontal_score < horizontal_threshold
        
        return is_vertical_symmetric, is_horizontal_symmetric
    except Exception as e:
        print(f'Error: {e}')
        return False, False


def symmetric_task_list(task_list, i, prefix=None):
    result = {}
    for uuid in tqdm(task_list):
        symmetric_tag = False
        symmetric_num = 0                    
        for index in range(6):
            image_path = os.path.join(f'{prefix}/{uuid[:2]}/{uuid}/normal_{index:04d}.webp')
            is_vertical_symmetric, is_horizontal_symmetric = is_symmetric(image_path)
            if is_vertical_symmetric or is_horizontal_symmetric:
                symmetric_num += 1
            if symmetric_num >= 1:
                symmetric_tag = True
                break
        result[uuid] = symmetric_tag
        symmetric_tag = 'symmetric' if symmetric_tag else 'asymmetric'
        with open(f'log/{i}.txt','a')as fp:
            fp.write(f'{uuid} {symmetric_tag}\n')
        fp.close()
    return result


if __name__ == '__main__':
    input_file = '/mnt/pfs/users/wangdehu/tmp_ids/v4_lrm_success.txt'
    output_file = 'symmetric_tag.txt'
    task_list = [line.strip() for line in open(input_file, 'r')]
    prefix = '/mnt/pfs/data/v4_36view_lrm/'
    os.makedirs('log', exist_ok=True)
    exists_data = {}
    for line in os.listdir('log'):
        with open(f'log/{line}','r') as f:
            for line in f:
                uuid, symmetric_tag = line.strip().split(' ')
                exists_data[uuid] = symmetric_tag
    work_list = []
    for line in task_list:
        uuid = line.strip()
        if uuid in exists_data:
            continue
        else:
            work_list.append(uuid)
    print(f'work_list: {len(work_list)} task_list: {len(task_list)}')

    # task_list = work_list
    task_num = max(min(32, len(work_list)),1)
    num_per_task = len(work_list) // task_num + 1
    procs = []
    pool = Pool(processes=task_num)
    for i in range(task_num):
        start = i * num_per_task
        end = min((i + 1) * num_per_task, len(work_list))
        sub_task_list = work_list[start:end]
        proc = pool.apply_async(symmetric_task_list, (sub_task_list, i, prefix))
        procs.append(proc)
    final_results = {}
    for p in procs:
        final_results.update(p.get())
    pool.close()
    pool.join()
    
    exists_data = {}
    for line in os.listdir('log'):
        with open(f'log/{line}','r') as f:
            for line in f:
                uuid, symmetric_tag = line.strip().split(' ')
                symmetric_tag = 'symmetric' if symmetric_tag in ['True', 'symmetric'] else 'asymmetric'
                exists_data[uuid] = symmetric_tag

    with open(output_file, 'w') as f:
        for line in task_list:
            uuid = line.strip()
            if uuid in final_results:
                f.write(f'{uuid} {final_results[uuid]}\n')
            else:
                f.write(f'{uuid} {exists_data[uuid]}\n')
    f.close()
