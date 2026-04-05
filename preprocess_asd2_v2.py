"""
数据预处理脚本 V2 - 适配真实数据路径
将AU时序数据转换为GASF编码，每个范式独立保存
"""
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from pyts.image import GramianAngularField

# ================= 配置 =================
SOURCE_DIR = "/home/lhj/桌面/NewData1013/data/AU_dataset_ALL_paradigm"
OUTPUT_DIR = "AU-ASD-TD-GASF-V2"
IMAGE_SIZE = 64

AU_COLS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]

REQUIRED_TASKS = ['A1', 'A2', 'C', 'D']  # 4个范式

def interpolate_seq(data, target_size):
    """时间序列插值到固定长度"""
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
    interpolated = F.interpolate(data_tensor, size=target_size, mode='linear', align_corners=False)
    return interpolated.squeeze(0).transpose(0, 1).numpy()


def process_subject_files(subject_id, group_dir, group_name):
    """
    处理单个被试的4个范式文件

    Args:
        subject_id: 被试编号，如 'S01001'
        group_dir: 组目录路径
        group_name: 'ASD' 或 'TD'

    Returns:
        dict: {task_name: tensor(17, 64, 64)} 或 None
    """
    transformer = GramianAngularField(image_size=IMAGE_SIZE, method='summation')
    scaler = MinMaxScaler(feature_range=(-1, 1))

    task_tensors = {}

    for task in REQUIRED_TASKS:
        file_path = os.path.join(group_dir, f"{subject_id}_{task}.csv")

        if not os.path.exists(file_path):
            print(f"⚠️  缺失文件: {subject_id}_{task}.csv")
            return None

        try:
            # 读取CSV
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            # 过滤有效数据
            valid_df = df[(df['success'] == 1) & (df['confidence'] >= 0.8)]

            if len(valid_df) == 0:
                print(f"⚠️  {subject_id}_{task}.csv 无有效数据")
                task_tensors[task] = torch.zeros((len(AU_COLS), IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
                continue

            # 提取AU数据
            data = valid_df[AU_COLS].values.astype(np.float32)

            # 插值到固定长度
            data = interpolate_seq(data, IMAGE_SIZE)

            # 归一化
            seq_normalized = scaler.fit_transform(data)

            # GASF转换
            images = transformer.fit_transform(seq_normalized.T)
            task_tensors[task] = torch.tensor(images, dtype=torch.float32)

        except Exception as e:
            print(f"⚠️  处理失败 ({subject_id}_{task}): {e}")
            task_tensors[task] = torch.zeros((len(AU_COLS), IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)

    # 检查是否4个任务都处理完成
    if len(task_tensors) != len(REQUIRED_TASKS):
        return None

    return task_tensors


def preprocess():
    """主预处理函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    groups = {
        'ASD': os.path.join(SOURCE_DIR, 'ASD_AUdata'),
        'TD': os.path.join(SOURCE_DIR, 'TD_AUdata')
    }

    for group_name, group_dir in groups.items():
        print(f"\n{'='*60}")
        print(f"📦 正在处理 {group_name} 组")
        print(f"{'='*60}")

        if not os.path.exists(group_dir):
            print(f"❌ 目录不存在: {group_dir}")
            continue

        # 创建输出目录
        group_out = os.path.join(OUTPUT_DIR, group_name)
        os.makedirs(group_out, exist_ok=True)

        # 获取所有被试ID（从文件名提取）
        all_files = [f for f in os.listdir(group_dir) if f.endswith('.csv')]
        subject_ids = set()
        for f in all_files:
            # 文件名格式: S01001_A1.csv
            subject_id = f.split('_')[0]
            subject_ids.add(subject_id)

        subject_ids = sorted(list(subject_ids))
        print(f"找到 {len(subject_ids)} 个被试")

        success_count = 0

        # 处理每个被试
        for subject_id in tqdm(subject_ids, desc=f"处理{group_name}"):
            task_tensors = process_subject_files(subject_id, group_dir, group_name)

            if task_tensors is None:
                continue

            # 保存：每个范式单独保存，方便后续分范式编码
            # 数据结构: {
            #   'A1': (17, 64, 64),
            #   'A2': (17, 64, 64),
            #   'C': (17, 64, 64),
            #   'D': (17, 64, 64)
            # }
            save_path = os.path.join(group_out, f"{subject_id}.pt")
            torch.save(task_tensors, save_path)
            success_count += 1

        print(f"✅ {group_name} 组完成: {success_count}/{len(subject_ids)} 个被试")

    print(f"\n{'='*60}")
    print(f"✅ 预处理完成！GASF 数据已保存在: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    preprocess()
