#!/usr/bin/env python3
"""
裁剪视频并重命名，处理三个不同的数据集：
- MSR_cath1: cath1_1b43A02/run001_protein.cmprsd/output_lossless.mp4 -> cath1_1b43A02_run001.mp4
- MSR_cath2: cath2_1a1wA00/run001_protein.cmprsd/output_lossless.mp4 -> cath2_1a1wA00_run001.mp4
- ATLAS: 1a62_A/R1/output_lossless.mp4 -> 1a62_A_R1.mp4
"""
import os
import cv2
import numpy as np
import subprocess
import time
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ================= 配置区域 =================

# 输入目录
INPUT_DIRS = {
    'msr_cath1': '/root/autodl-tmp/video_msr_cath1',
    'msr_cath2': '/root/autodl-tmp/video_msr_cath2',
    'atlas': '/root/autodl-tmp/video_atlas'
}

# 输出目录
OUTPUT_DIR = '/root/autodl-tmp/video_cropped_renamed'

# "接近白色"亮度阈值（越小越严格，240-250通常合适）
WHITE_THRESHOLD = 240

# 1. 安全边距 (Safe Margin)
SAFE_MARGIN = 30

# 2. 补白颜色 (Pad Color)
PAD_COLOR = "0xfcfffc"

# 输出分辨率
TARGET_W = 1280
TARGET_H = 720
TARGET_RATIO = TARGET_W / TARGET_H

# 并行进程数 (默认使用 CPU 核心数 - 1，防止电脑卡死)
PROCESS_NUM = 90

# ===========================================


def get_global_crop_region(video_path, margin=0):
    """
    扫描视频的【每一帧】，计算所有帧内容的并集区域（最大包围盒）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化全局边界
    global_x_min = w_frame
    global_y_min = h_frame
    global_x_max = 0
    global_y_max = 0

    found_content = False

    # 逐帧读取
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 二值化：小于阈值的为内容
        mask = gray < WHITE_THRESHOLD

        if not np.any(mask):
            continue

        # 获取这一帧内容的坐标范围
        coords_y, coords_x = np.where(mask)

        y_min_curr, y_max_curr = coords_y.min(), coords_y.max()
        x_min_curr, x_max_curr = coords_x.min(), coords_x.max()

        # 更新全局最大边界
        if x_min_curr < global_x_min:
            global_x_min = x_min_curr
        if y_min_curr < global_y_min:
            global_y_min = y_min_curr
        if x_max_curr > global_x_max:
            global_x_max = x_max_curr
        if y_max_curr > global_y_max:
            global_y_max = y_max_curr

        found_content = True

    cap.release()

    if not found_content:
        return None

    # === 添加安全边距 (Safe Margin) 并确保不越界 ===
    x_min = max(0, global_x_min - margin)
    y_min = max(0, global_y_min - margin)
    x_max = min(w_frame, global_x_max + margin)
    y_max = min(h_frame, global_y_max + margin)

    w = x_max - x_min
    h = y_max - y_min

    return x_min, y_min, w, h


def generate_output_filename(video_path):
    """
    根据视频路径生成输出文件名
    
    Args:
        video_path: 视频文件的完整路径
    
    Returns:
        (output_filename, dataset_type) 或 (None, None) 如果无法识别
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
                return f"{protein_name}_run{run_num}.mp4", 'msr_cath1'
    
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
                return f"{protein_name}_run{run_num}.mp4", 'msr_cath2'
    
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
                return f"{protein_name}_R{run_num}.mp4", 'atlas'
    
    return None, None


def process_single_video(args):
    """
    单个视频的处理逻辑，供进程池调用
    
    Args:
        args: (video_path, output_filename) 元组
    """
    video_path, output_filename = args
    file_name = os.path.basename(video_path)
    
    try:
        dst_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # 如果输出文件已存在，跳过
        if os.path.exists(dst_path):
            print(f"[{output_filename}] ⏭️  文件已存在，跳过")
            return

        print(f"[{output_filename}] 🔍 开始分析内容区域...")

        # 获取带 Margin 的全局裁剪区域
        region = get_global_crop_region(video_path, margin=SAFE_MARGIN)

        if region is None:
            print(f"[{output_filename}] ⚠️  似乎是全白的或未检测到内容，跳过。")
            return

        x, y, w, h = region

        # 裁剪后实际比例
        ratio = w / h

        # 计算补边(pad)以适应目标比例
        pad_top = pad_bottom = pad_left = pad_right = 0

        if abs(ratio - TARGET_RATIO) > 1e-3:
            if ratio > TARGET_RATIO:
                # 当前更宽 → 需要上下补色
                new_h = int(w / TARGET_RATIO)
                diff = new_h - h
                pad_top = diff // 2
                pad_bottom = diff - pad_top
            else:
                # 当前更窄 → 需要左右补色
                new_w = int(h * TARGET_RATIO)
                diff = new_w - w
                pad_left = diff // 2
                pad_right = diff - pad_left

        # 构建 FFmpeg 滤镜链
        # 1. crop: 裁剪出带margin的内容
        vf_filters = f"crop={w}:{h}:{x}:{y}"

        # 2. pad: 如果需要，补上指定颜色的边
        if pad_top or pad_bottom or pad_left or pad_right:
            vf_filters += f",pad=width={w+pad_left+pad_right}:height={h+pad_top+pad_bottom}:x={pad_left}:y={pad_top}:color={PAD_COLOR}"

        # 3. scale: 缩放到最终分辨率
        vf_filters += f",scale={TARGET_W}:{TARGET_H}"

        # 执行命令
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vf", vf_filters,
            "-c:v", "libx264",  # 显式指定编码器，保证兼容性
            "-preset", "fast",  # 多进程下可以用 fast 平衡速度
            "-c:a", "copy",     # 音频直接复制
            "-y",               # 覆盖输出
            dst_path
        ]

        print(f"[{output_filename}] 🚀 正在导出 FFmpeg...")
        # 隐藏 FFmpeg 的输出，只在出错时显示
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"[{output_filename}] ❌ FFmpeg 错误: {result.stderr}")
        else:
            print(f"[{output_filename}] ✨ 处理完成!")

    except Exception as e:
        print(f"[{output_filename}] ❌ 发生异常: {str(e)}")


def collect_video_tasks():
    """
    收集所有需要处理的视频任务
    
    Returns:
        list: [(video_path, output_filename), ...] 元组列表
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
                output_filename, dataset_type = generate_output_filename(video_path)
                
                if output_filename:
                    tasks.append((video_path, output_filename))
                    print(f"  ✅ 找到: {video_path} -> {output_filename}")
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
    print(f"补边颜色: {PAD_COLOR}")
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

