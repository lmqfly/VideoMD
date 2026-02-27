import os
import cv2
import numpy as np
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count

# ================= 配置区域 =================

# 输入与输出目录
INPUT_DIR = "all_data"
OUTPUT_DIR = "all_data_cropped"

# “接近白色”亮度阈值（越小越严格，240-250通常合适）
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
PROCESS_NUM = 20

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
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

    # 初始化全局边界
    global_x_min = w_frame
    global_y_min = h_frame
    global_x_max = 0
    global_y_max = 0

    found_content = False
    
    # 注意：多进程模式下，尽量减少 print，否则控制台会乱码
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
        if x_min_curr < global_x_min: global_x_min = x_min_curr
        if y_min_curr < global_y_min: global_y_min = y_min_curr
        if x_max_curr > global_x_max: global_x_max = x_max_curr
        if y_max_curr > global_y_max: global_y_max = y_max_curr
        
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

def process_single_video(file_name):
    """
    单个视频的处理逻辑，供进程池调用
    """
    try:
        src_path = os.path.join(INPUT_DIR, file_name)
        dst_path = os.path.join(OUTPUT_DIR, file_name)

        print(f"[{file_name}] 🔍 开始分析内容区域...")
        
        # 获取带 Margin 的全局裁剪区域
        region = get_global_crop_region(src_path, margin=SAFE_MARGIN)
        
        if region is None:
            print(f"[{file_name}] ⚠️ 似乎是全白的或未检测到内容，跳过。")
            return

        x, y, w, h = region
        # print(f"[{file_name}] ✅ 裁剪区域: x={x}, y={y}, w={w}, h={h}")

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
            "-i", src_path,
            "-vf", vf_filters,
            "-c:v", "libx264", # 显式指定编码器，保证兼容性
            "-preset", "fast", # 多进程下可以用 fast 平衡速度
            "-c:a", "copy",    # 音频直接复制
            "-y",              # 覆盖输出
            dst_path
        ]

        print(f"[{file_name}] 🚀 正在导出 FFmpeg...")
        # 隐藏 FFmpeg 的输出，只在出错时显示
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"[{file_name}] ❌ FFmpeg 错误: {result.stderr}")
        else:
            print(f"[{file_name}] ✨ 处理完成!")

    except Exception as e:
        print(f"[{file_name}] ❌ 发生异常: {str(e)}")

def main():
    # 1. 准备目录
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录 '{INPUT_DIR}' 不存在")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 收集任务
    video_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp4")]
    total_files = len(video_files)
    
    if total_files == 0:
        print("没有找到 .mp4 文件")
        return

    print(f"扫描到 {total_files} 个视频，准备使用 {PROCESS_NUM} 个进程并行处理。")
    print(f"补边颜色: {PAD_COLOR}")
    print("-" * 30)

    start_time = time.time()

    # 3. 启动多进程
    # Windows 下必须使用 if __name__ == '__main__': 保护
    with Pool(processes=PROCESS_NUM) as pool:
        pool.map(process_single_video, video_files)

    end_time = time.time()
    print("-" * 30)
    print(f"🎬 全部完成！总耗时: {end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    main()