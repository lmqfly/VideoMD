#!/usr/bin/env python3
"""
裁剪视频并提取下采样帧（11帧，包含第一帧和最后一帧）
处理三个不同的数据集：
- MSR_cath1: cath1_1b43A02/run001_protein.cmprsd/output_lossless.mp4 -> cath1_1b43A02_001_1.png 到 cath1_1b43A02_001_11.png
- MSR_cath2: cath2_1a1wA00/run001_protein.cmprsd/output_lossless.mp4 -> cath2_1a1wA00_001_1.png 到 cath2_1a1wA00_001_11.png
- ATLAS: 1a62_A/R1/output_lossless.mp4 -> 1a62_A_R1_1.png 到 1a62_A_R1_11.png
"""
import os
import cv2
import numpy as np
import time
import re
from multiprocessing import Pool
from pathlib import Path

# ================= 配置区域 =================

# 输入目录
INPUT_DIRS = {
    'msr_cath1': '/root/autodl-tmp/video_msr_cath1',
    'msr_cath2': '/root/autodl-tmp/video_msr_cath2',
    'atlas': '/root/autodl-tmp/video_atlas'
}

# 输出目录
OUTPUT_DIR = '/root/autodl-tmp/video_frames_cropped'

# "接近白色"亮度阈值（越小越严格，240-250通常合适）
WHITE_THRESHOLD = 240

# 1. 安全边距 (Safe Margin) - 每一帧裁剪时保留的边缘像素
SAFE_MARGIN = 10

# 2. 补白颜色 (Pad Color) - 格式 0xRRGGBB
PAD_COLOR_HEX = "0xfcfffc"

# 输出分辨率
TARGET_W = 224
TARGET_H = 224

# 下采样帧数（包含第一帧和最后一帧）
NUM_SAMPLED_FRAMES = 11

# 并行进程数
PROCESS_NUM = 90

# ===========================================


def hex_to_bgr(hex_str):
    """将 0xRRGGBB 转换为 OpenCV 需要的 (B, G, R) 元组"""
    color_int = int(hex_str, 16)
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    return (b, g, r)


# 预计算背景颜色
BG_COLOR = hex_to_bgr(PAD_COLOR_HEX)


def process_single_frame_logic(frame):
    """
    核心图像处理逻辑：
    1. 识别内容区域
    2. 裁剪
    3. 等比缩放
    4. 居中贴图到目标画布
    """
    h_src, w_src = frame.shape[:2]

    # 1. 转灰度并二值化寻找内容
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray < WHITE_THRESHOLD

    # 如果全是白色（没内容），直接返回全白画布
    if not np.any(mask):
        canvas = np.full((TARGET_H, TARGET_W, 3), BG_COLOR, dtype=np.uint8)
        return canvas

    # 2. 获取内容坐标
    coords_y, coords_x = np.where(mask)
    y_min, y_max = coords_y.min(), coords_y.max()
    x_min, x_max = coords_x.min(), coords_x.max()

    # 3. 添加安全边距 (Margin) 并防止越界
    x_min = max(0, x_min - SAFE_MARGIN)
    y_min = max(0, y_min - SAFE_MARGIN)
    x_max = min(w_src, x_max + SAFE_MARGIN)
    y_max = min(h_src, y_max + SAFE_MARGIN)

    # 裁剪出内容
    crop = frame[y_min:y_max, x_min:x_max]
    h_crop, w_crop = crop.shape[:2]

    if h_crop == 0 or w_crop == 0:
        return np.full((TARGET_H, TARGET_W, 3), BG_COLOR, dtype=np.uint8)

    # 4. 计算缩放比例 (保持长宽比，适应目标分辨率)
    scale_w = TARGET_W / w_crop
    scale_h = TARGET_H / h_crop
    scale = min(scale_w, scale_h)  # 取较小值，确保能完整放入

    new_w = int(w_crop * scale)
    new_h = int(h_crop * scale)

    # 缩放内容
    resized_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5. 创建目标画布并居中粘贴
    canvas = np.full((TARGET_H, TARGET_W, 3), BG_COLOR, dtype=np.uint8)

    # 计算粘贴位置 (居中)
    x_offset = (TARGET_W - new_w) // 2
    y_offset = (TARGET_H - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_crop

    return canvas


def get_sampled_frame_indices(total_frames, num_samples):
    """
    计算下采样帧索引，从第一帧开始，间隔10帧采样
    
    Args:
        total_frames: 总帧数
        num_samples: 需要采样的帧数（此参数保留以兼容调用，实际按间隔10采样）
    
    Returns:
        list: 帧索引列表（0-indexed，例如：0, 10, 20, 30, ...）
    """
    # 从第0帧（第一帧）开始，每10帧采样一次
    interval = 10
    indices = []
    
    idx = 0
    while idx < total_frames:
        indices.append(idx)
        idx += interval
    
    return indices


def generate_output_filename(video_path):
    """
    根据视频路径生成输出文件名前缀
    
    Args:
        video_path: 视频文件的完整路径
    
    Returns:
        (output_prefix, dataset_type) 或 (None, None) 如果无法识别
    """
    path_parts = Path(video_path).parts

    # 检查是否是 MSR_cath1 格式: .../video_msr_cath1/cath1_xxx/run001_protein.cmprsd/output_lossless.mp4
    if 'video_msr_cath1' in path_parts:
        idx = path_parts.index('video_msr_cath1')
        if idx + 2 < len(path_parts):
            protein_name = path_parts[idx + 1]  # cath1_1b43A02
            run_dir = path_parts[idx + 2]  # run001_protein.cmprsd
            # 提取 run 编号
            match = re.search(r'run(\d+)', run_dir)
            if match:
                run_num = match.group(1)
                return f"{protein_name}_{run_num}", 'msr_cath1'

    # 检查是否是 MSR_cath2 格式: .../video_msr_cath2/cath2_xxx/run001_protein.cmprsd/output_lossless.mp4
    if 'video_msr_cath2' in path_parts:
        idx = path_parts.index('video_msr_cath2')
        if idx + 2 < len(path_parts):
            protein_name = path_parts[idx + 1]  # cath2_1a1wA00
            run_dir = path_parts[idx + 2]  # run001_protein.cmprsd
            # 提取 run 编号
            match = re.search(r'run(\d+)', run_dir)
            if match:
                run_num = match.group(1)
                return f"{protein_name}_{run_num}", 'msr_cath2'

    # 检查是否是 ATLAS 格式: .../video_atlas/1a62_A/R1/output_lossless.mp4
    if 'video_atlas' in path_parts:
        idx = path_parts.index('video_atlas')
        if idx + 2 < len(path_parts):
            protein_name = path_parts[idx + 1]  # 1a62_A
            run_dir = path_parts[idx + 2]  # R1
            # 提取 R 编号
            match = re.search(r'R(\d+)', run_dir)
            if match:
                run_num = match.group(1)
                return f"{protein_name}_R{run_num}", 'atlas'

    return None, None


def process_single_video(args):
    """
    处理单个视频文件，提取下采样帧并裁剪
    
    Args:
        args: (video_path, output_prefix) 元组
    """
    video_path, output_prefix = args
    video_name = os.path.basename(video_path)

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[{output_prefix}] ❌ 无法打开视频: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"[{output_prefix}] ⚠️  视频帧数为0，跳过")
            cap.release()
            return

        # 计算下采样帧索引
        sampled_indices = get_sampled_frame_indices(total_frames, NUM_SAMPLED_FRAMES)
        
        print(f"[{output_prefix}] 🚀 开始处理 {total_frames} 帧，采样 {len(sampled_indices)} 帧 (动态裁剪)...")

        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 只处理采样帧
            if frame_idx in sampled_indices:
                try:
                    # 处理这一帧
                    final_img = process_single_frame_logic(frame)

                    # 计算在采样帧中的序号（1-indexed）
                    seq_num = sampled_indices.index(frame_idx) + 1

                    # 生成输出文件名
                    output_filename = f"{output_prefix}_{seq_num}.png"
                    save_path = os.path.join(OUTPUT_DIR, output_filename)

                    # 如果文件已存在，跳过
                    if os.path.exists(save_path):
                        print(f"[{output_prefix}] ⏭️  {output_filename} 已存在，跳过")
                    else:
                        # 保存图片 (压缩等级3，平衡速度和大小)
                        cv2.imwrite(save_path, final_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                        saved_count += 1

                except Exception as e:
                    print(f"[{output_prefix}] ⚠️ 第 {frame_idx} 帧处理失败: {e}")

            frame_idx += 1

        cap.release()
        print(f"[{output_prefix}] ✨ 完成。保存了 {saved_count}/{len(sampled_indices)} 帧")

    except Exception as e:
        print(f"[{output_prefix}] ❌ 发生异常: {str(e)}")


def collect_video_tasks():
    """
    收集所有需要处理的视频任务
    
    Returns:
        list: [(video_path, output_prefix), ...] 元组列表
    """
    tasks = []

    for dataset_name, input_dir in INPUT_DIRS.items():
        if not os.path.exists(input_dir):
            print(f"⚠️  警告：输入目录不存在: {input_dir}")
            continue

        print(f"📂 扫描目录: {input_dir}")

        # 递归查找所有 output_lossless.mp4 文件
        for root, dirs, files in os.walk(input_dir):
            if 'output_lossless.mp4' in files:
                video_path = os.path.join(root, 'output_lossless.mp4')
                output_prefix, dataset_type = generate_output_filename(video_path)

                if output_prefix:
                    tasks.append((video_path, output_prefix))
                    print(f"  ✅ 找到: {video_path} -> {output_prefix}_*.png")
                else:
                    print(f"  ⚠️  无法识别路径格式: {video_path}")

    return tasks


def main():
    # 1. 准备输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 收集任务
    print("=" * 60)
    print("开始收集视频文件...")
    print("=" * 60)

    tasks = collect_video_tasks()
    total_files = len(tasks)

    if total_files == 0:
        print("没有找到需要处理的视频文件")
        return

    print("=" * 60)
    print(f"扫描到 {total_files} 个视频，准备使用 {PROCESS_NUM} 个进程并行处理。")
    print(f"模式: 下采样提取 {NUM_SAMPLED_FRAMES} 帧 + 逐帧动态裁剪 + 居中补全")
    print(f"目标分辨率: {TARGET_W}x{TARGET_H}")
    print(f"背景颜色(BGR): {BG_COLOR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    start_time = time.time()

    # 3. 启动多进程
    with Pool(processes=PROCESS_NUM) as pool:
        pool.map(process_single_video, tasks)

    end_time = time.time()
    print("=" * 60)
    print(f"🎬 全部完成！总耗时: {end_time - start_time:.2f} 秒")
    print(f"处理了 {total_files} 个视频文件")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()

