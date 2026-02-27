import os
import cv2
import numpy as np
import time
from multiprocessing import Pool

# ================= 配置区域 =================

# 输入目录
INPUT_DIR = "all_data"
# 输出目录
OUTPUT_DIR = "all_data_frames_dynamic"

# “接近白色”亮度阈值（越小越严格，240-250通常合适）
WHITE_THRESHOLD = 240

# 1. 安全边距 (Safe Margin) - 每一帧裁剪时保留的边缘像素
SAFE_MARGIN = 10 

# 2. 补白颜色 (Pad Color) - 格式 0xRRGGBB
PAD_COLOR_HEX = "0xfcfffc" 

# 输出分辨率
TARGET_W = 224
TARGET_H = 224

# 并行进程数 (建议设置为 CPU 核心数 - 2)
PROCESS_NUM = 20

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
    scale = min(scale_w, scale_h) # 取较小值，确保能完整放入

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

def process_single_video(file_name):
    """
    处理单个视频文件
    """
    src_path = os.path.join(INPUT_DIR, file_name)
    
    # 创建该视频对应的输出文件夹
    video_name_no_ext = os.path.splitext(file_name)[0]
    video_output_dir = os.path.join(OUTPUT_DIR, video_name_no_ext)
    
    # 如果文件夹已存在且有内容，可以选择跳过，这里默认覆盖
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"[{file_name}] ❌ 无法打开视频")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{file_name}] 🚀 开始处理 {frame_count} 帧 (动态裁剪)...")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # 处理每一帧
            final_img = process_single_frame_logic(frame)
            
            # 保存路径
            save_path = os.path.join(video_output_dir, f"{idx:04d}.png")
            
            # 保存图片 (压缩等级3，平衡速度和大小)
            cv2.imwrite(save_path, final_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
        except Exception as e:
            print(f"[{file_name}] ⚠️ 第 {idx} 帧处理失败: {e}")

        idx += 1

    cap.release()
    print(f"[{file_name}] ✨ 完成。")

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
    print(f"模式: 逐帧动态裁剪 + 居中补全")
    print(f"目标分辨率: {TARGET_W}x{TARGET_H}")
    print(f"背景颜色(BGR): {BG_COLOR}")
    print("-" * 30)

    start_time = time.time()

    # 3. 启动多进程
    with Pool(processes=PROCESS_NUM) as pool:
        pool.map(process_single_video, video_files)

    end_time = time.time()
    print("-" * 30)
    print(f"🎬 全部完成！总耗时: {end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    main()