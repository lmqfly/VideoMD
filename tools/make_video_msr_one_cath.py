import os
import time
import subprocess
from multiprocessing import Process
import argparse

# 路径配置
raw_path = '/root/autodl-tmp/bio_emu_data/ONE_cath1'
tmp_path = '/root/autodl-tmp/tmp_msr_cath1'
out_path = '/root/autodl-tmp/video_msr_cath1'

# GPU配置
gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]  # 三张GPU编号
max_parallel = len(gpu_list)  # 控制最多同时执行3个任务


def make_video_vmd(pdbfile, xtcfile, tmp_dir, out_dir):
    """
    使用VMD渲染轨迹并生成视频
    """
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)
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

# --- 关键修改 1: 先做一次预处理 Align ---
# 为了确定最佳视角，我们先只做 Align，不渲染。
# 这样可以确保第一帧的位置是正确的。
# (其实如果只以第0帧为基准，这一步可以省略，直接在循环里做也行，
# 但为了保险起见，我们先设定好第0帧的视角)

animate goto 0
display resetview
# 可以手动调整一下缩放，防止贴边太紧
scale by 0.9 


# 循环遍历每一帧
for {{set i 0}} {{$i < $num_frames}} {{incr i 5}} {{

    animate goto $i
    display update

    # --- Align (对齐) ---
    set current_sel [atomselect top "protein and name CA" frame $i]
    set all [atomselect top all frame $i]
    
    set trans_mat [measure fit $current_sel $ref_sel]
    $all move $trans_mat
    
    $current_sel delete
    $all delete
    
    # --- 关键修改 2: 删除了 display resetview ---
    # 摄像机现在完全不动了，分子只会在原地扭动，不会忽大忽小。

    # 设置文件名 (请替换 {tmp_dir})
    set datfile "{tmp_dir}/frame[format "%04d" $i].dat"
    set imgfile "{tmp_dir}/frame[format "%04d" $i].bmp"
    
    # 渲染
    render Tachyon $datfile "/usr/local/lib/vmd/tachyon_LINUXAMD64" \
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


def process_protein_dir(protein_dir):
    """
    处理单个蛋白质目录，为所有轨迹文件生成视频
    """
    protein_name = os.path.basename(protein_dir)
    pdb_file = os.path.join(protein_dir, "topology.pdb")
    trajs_dir = os.path.join(protein_dir, "trajs")
    
    # 检查必要文件是否存在
    if not os.path.exists(pdb_file):
        print(f"⚠️  警告：{protein_name} 目录下未找到 topology.pdb")
        return
    
    if not os.path.exists(trajs_dir):
        print(f"⚠️  警告：{protein_name} 目录下未找到 trajs 目录")
        return
    
    # 获取所有轨迹文件
    traj_files = [
        os.path.join(trajs_dir, f)
        for f in os.listdir(trajs_dir)
        if 'run' in f and f.endswith('.xtc') and int(f[3:6]) >= 1
    ]
    
    if not traj_files:
        print(f"⚠️  警告：{protein_name} 目录下未找到轨迹文件")
        return
    
    # 为每个轨迹文件生成视频
    for traj_file in sorted(traj_files):
        traj_name = os.path.splitext(os.path.basename(traj_file))[0]
        out_dir = os.path.join(out_path, protein_name, traj_name)
        tmp_dir = os.path.join(tmp_path, protein_name, traj_name)
        
        print(f"🎬 开始处理：{protein_name}/{traj_name}")
        try:
            make_video_vmd(pdb_file, traj_file, tmp_dir, out_dir)
            print(f"✅ 完成：{protein_name}/{traj_name}")
        except Exception as e:
            print(f"❌ 处理失败：{protein_name}/{traj_name}，错误：{e}")


def run_process_with_gpu(protein_dir, gpu_id):
    """
    在指定GPU上运行处理任务
    """
    protein_name = os.path.basename(protein_dir)
    # 设置仅可见的GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"🚀 启动任务 {protein_name}，使用 GPU {gpu_id}")
    process_protein_dir(protein_dir)
    print(f"✅ 任务 {protein_name} 在 GPU {gpu_id} 完成")


def schedule_jobs():
    """
    调度所有任务，使用GPU并行处理
    """
    if not os.path.exists(raw_path):
        print(f"❌ 错误：路径 {raw_path} 不存在")
        return
    
    # 获取所有蛋白质目录
    protein_dirs = [
        os.path.join(raw_path, d)
        for d in os.listdir(raw_path)
        if os.path.isdir(os.path.join(raw_path, d)) and d.startswith('cath1_')
    ]
    
    if not protein_dirs:
        print(f"⚠️  警告：在 {raw_path} 下未找到蛋白质目录")
        return
    
    print(f"📋 找到 {len(protein_dirs)} 个蛋白质目录")
    
    active_processes = []
    gpu_idx = 0

    for protein_dir in protein_dirs:
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

        p = Process(target=run_process_with_gpu, args=(protein_dir, gpu_id))
        p.start()
        active_processes.append(p)

    # 等待所有任务结束
    for p in active_processes:
        p.join()

    print("\n🎉 所有任务已完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为MSR_cath2数据集生成视频')
    parser.add_argument("protein_name", nargs='?', default=None, 
                       help="单独处理指定的蛋白质目录名（例如：cath2_1a1wA00）")
    args = parser.parse_args()
    
    if args.protein_name:
        # 处理单个蛋白质目录
        protein_dir = os.path.join(raw_path, args.protein_name)
        if os.path.exists(protein_dir):
            process_protein_dir(protein_dir)
        else:
            print(f"❌ 错误：目录 {protein_dir} 不存在")
    else:
        # 处理所有目录
        schedule_jobs()