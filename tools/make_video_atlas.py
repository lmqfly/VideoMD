import os
import time
import subprocess
from multiprocessing import Process
import argparse

# 路径配置
raw_path = '/root/autodl-tmp/atlas/ATLAS'
tmp_path = '/root/autodl-tmp/tmp_atlas'
out_path = '/root/autodl-tmp/video_atlas'
pdb_out_path = '/root/autodl-tmp/pdb_atlas'  # PDB结构输出路径

# GPU配置
gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]  # GPU编号
max_parallel = len(gpu_list)  # 控制最多同时执行的任务数


def is_video_complete(video_file, min_size_mb=1):
    """
    检查视频文件是否完整存在且有效
    Args:
        video_file: 视频文件路径
        min_size_mb: 最小文件大小（MB），默认1MB
    Returns:
        bool: 如果视频文件存在且大小合理，返回True
    """
    if not os.path.exists(video_file):
        return False
    
    # 检查文件大小（至少1MB，避免损坏的或不完整的文件）
    file_size = os.path.getsize(video_file)
    min_size_bytes = min_size_mb * 1024 * 1024
    
    if file_size < min_size_bytes:
        return False
    
    # 可选：使用ffprobe验证视频文件完整性（如果可用）
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
             '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        # 如果ffprobe成功且返回了时长信息，说明视频文件有效
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            if duration > 0:
                return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        # ffprobe不可用或失败，仅基于文件大小判断
        pass
    
    # 如果没有ffprobe，仅基于文件大小判断
    return True


def make_video_vmd(pdbfile, xtcfile, tmp_dir, out_dir, pdb_dir):
    """
    使用VMD渲染轨迹并生成视频，同时保存每一帧的PDB结构
    """
    # 检查视频是否已经生成
    output_file = os.path.join(out_dir, "output_lossless.mp4")
    if is_video_complete(output_file):
        print(f"⏭️  跳过：视频已存在且完整 - {output_file}")
        return
    
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 定义要写入的内容
    tcl_script = f"""# 加载蛋白质结构和轨迹
mol new {pdbfile}
mol addfile {xtcfile} waitfor all

# Delete the default representation
mol delrep 0 top

# Set the view and focus
mol representation NewCartoon
mol color Structure
mol material AOShiny
mol addrep top

# 删除坐标轴
axes location Off

# 背景为白色
color Display Background white

# 开启 GLSL
display rendermode GLSL

# --- 准备工作 ---
set ref_sel [atomselect top "protein and name CA" frame 0]
set num_frames [molinfo top get numframes]

# --- 对齐设置 ---
animate goto 0
display resetview
# 可以手动调整一下缩放，防止贴边太紧
scale by 0.9 

# 循环遍历每一帧
for {{set i 0}} {{$i < $num_frames}} {{incr i 10}} {{

    animate goto $i
    display update

    # --- Align (对齐) ---
    set current_sel [atomselect top "protein and name CA" frame $i]
    set all [atomselect top all frame $i]
    
    set trans_mat [measure fit $current_sel $ref_sel]
    $all move $trans_mat
    
    $current_sel delete
    $all delete

    # 设置文件名
    set datfile "{tmp_dir}/frame[format "%04d" $i].dat"
    set imgfile "{tmp_dir}/frame[format "%04d" $i].bmp"
    
    # 渲染
    render Tachyon $datfile "/usr/local/lib/vmd/tachyon_LINUXAMD64" \\
        -aasamples 12 $datfile -format BMP -o $imgfile -res 2048 2048
}}

quit
"""

    # 写入到一个 .tcl 文件
    file_name = os.path.join(tmp_dir, "script.tcl")
    with open(file_name, "w") as file:
        file.write(tcl_script)
    
    # 执行VMD渲染
    os.system(f'vmd -dispdev text -e {file_name} > /dev/null 2>&1')
    
    # 使用ffmpeg合成视频
    framerate = 15  # 帧率
    output_file = os.path.join(out_dir, "output_lossless.mp4")
    ffmpeg_path = "ffmpeg"
    cmd = [
        ffmpeg_path,
        "-framerate", str(framerate),
        "-pattern_type", "glob",
        "-i", os.path.join(tmp_dir, "frame*.bmp"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 视频合成完成：{output_file}")
    except subprocess.CalledProcessError as e:
        print("\n❌ FFmpeg 执行失败！")
        print("错误信息:", e)
    
    # 清理临时文件
    os.system(f'rm -rf {tmp_dir}')


def process_pdb_dir(pdbid):
    """
    处理单个蛋白质目录，为所有轨迹文件生成视频和PDB结构
    """
    protein_dir = os.path.join(raw_path, pdbid)
    analysis_dir = os.path.join(protein_dir, "analysis")
    pdb_file = os.path.join(analysis_dir, f"{pdbid}.pdb")
    
    # 检查必要文件是否存在
    if not os.path.exists(pdb_file):
        print(f"⚠️  警告：{pdbid} 目录下未找到 {pdbid}.pdb")
        return
    
    if not os.path.exists(analysis_dir):
        print(f"⚠️  警告：{pdbid} 目录下未找到 analysis 目录")
        return
    
    # 处理R1, R2, R3三个轨迹文件
    for run_num in [1, 2, 3]:
        traj_file = os.path.join(analysis_dir, f"{pdbid}_R{run_num}.xtc")
        
        if not os.path.exists(traj_file):
            print(f"⚠️  警告：{pdbid} 目录下未找到轨迹文件 {pdbid}_R{run_num}.xtc")
            continue
        
        traj_name = f"R{run_num}"
        out_dir = os.path.join(out_path, pdbid, traj_name)
        pdb_dir = os.path.join(pdb_out_path, pdbid, traj_name)
        tmp_dir = os.path.join(tmp_path, pdbid, traj_name)
        
        # 在处理前检查视频是否已存在
        output_file = os.path.join(out_dir, "output_lossless.mp4")
        if is_video_complete(output_file):
            print(f"⏭️  跳过：{pdbid}/{traj_name} - 视频已存在且完整")
            continue
        
        print(f"🎬 开始处理：{pdbid}/{traj_name}")
        try:
            make_video_vmd(pdb_file, traj_file, tmp_dir, out_dir, pdb_dir)
            print(f"✅ 完成：{pdbid}/{traj_name}")
        except Exception as e:
            print(f"❌ 处理失败：{pdbid}/{traj_name}，错误：{e}")


def run_process_with_gpu(pdbid, gpu_id):
    """
    在指定GPU上运行处理任务
    """
    # 设置仅可见的GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"🚀 启动任务 {pdbid}，使用 GPU {gpu_id}")
    process_pdb_dir(pdbid)
    print(f"✅ 任务 {pdbid} 在 GPU {gpu_id} 完成")


def schedule_jobs():
    """
    调度所有任务，使用GPU并行处理
    """
    if not os.path.exists(raw_path):
        print(f"❌ 错误：路径 {raw_path} 不存在")
        return
    
    # 获取所有蛋白质目录
    pdb_dirs = [
        d for d in os.listdir(raw_path)
        if os.path.isdir(os.path.join(raw_path, d))
    ]
    
    if not pdb_dirs:
        print(f"⚠️  警告：在 {raw_path} 下未找到蛋白质目录")
        return
    
    print(f"📋 找到 {len(pdb_dirs)} 个蛋白质目录")
    
    active_processes = []
    gpu_idx = 0

    for pdbid in pdb_dirs:
        # 如果当前GPU均繁忙，则等待
        while len(active_processes) >= max_parallel:
            # 检查是否有子进程结束
            for p in list(active_processes):
                if not p.is_alive():
                    active_processes.remove(p)
            time.sleep(5)

        # 分配GPU并启动进程
        gpu_id = gpu_list[gpu_idx]
        gpu_idx = (gpu_idx + 1) % max_parallel

        p = Process(target=run_process_with_gpu, args=(pdbid, gpu_id))
        p.start()
        active_processes.append(p)

    # 等待所有任务结束
    for p in active_processes:
        p.join()

    print("\n🎉 所有任务已完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为ATLAS数据集生成视频和PDB结构')
    parser.add_argument("pdbid", nargs='?', default=None, 
                       help="单独处理指定的蛋白质ID（例如：1c1k_A）")
    args = parser.parse_args()
    
    if args.pdbid:
        # 处理单个蛋白质目录
        protein_dir = os.path.join(raw_path, args.pdbid)
        if os.path.exists(protein_dir):
            process_pdb_dir(args.pdbid)
        else:
            print(f"❌ 错误：目录 {protein_dir} 不存在")
    else:
        # 处理所有目录
        schedule_jobs()