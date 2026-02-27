import os
from multiprocessing import Pool
from PIL import Image
import subprocess

raw_path = 'autodl-tmp/ATLAS_protein_only/ATLAS'
files = os.listdir(raw_path)
parms = [file.split('.')[0] for file in files]
tmp_path = 'autodl-tmp/tmp'
out_path = 'autodl-tmp/video'

def convert_tga_to_png(args):
    """
    Converts a single TGA file to PNG and saves it in the target folder.
    """
    file_path, output_folder = args
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        png_path = os.path.join(output_folder, base_name + '.png')

        with Image.open(file_path) as img:
            img.save(png_path, 'PNG')
        
        print(f"Converted: {file_path} -> {png_path}")
    except Exception as e:
        print(f"Error converting {file_path}: {e}")

def process_folder(input_folder, output_folder):
    """
    Processes all TGA files in the input folder and converts them to PNG in the output folder.
    """
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all .tga files in the input folder
    tga_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith('.tga')
    ]

    if not tga_files:
        print("No TGA files found in the input folder.")
        return

    # Use Pool.map for parallel processing
    with Pool() as pool:
        pool.map(convert_tga_to_png, [(file, output_folder) for file in tga_files])
        
        
def render_frames(start, end, tmp_dir, pdbfile, xtcfile):
    # 临时脚本文件
    tcl_script = f"""# Adjusted script
    set start_frame {start}
    set end_frame {end}
    set tmp_dir "{tmp_dir}"
    mol new {pdbfile}
    mol addfile {xtcfile}
    # Delete the default representation
        mol delrep 0 top

        # Set the view and focus
        mol representation NewCartoon
        mol color Structure
        mol material AOShiny
        mol addrep top

        # 居中并缩放蛋白质
        set sel [atomselect top "protein"]
        set center [measure center $sel]
        set size [measure minmax $sel]
        set span [vecsub [lindex $size 1] [lindex $size 0]]
        translate by [vecscale -1 $center]
        set max_span [lindex [lsort -real $span] end]
        scale by [expr 1.0 / $max_span]

        # 删除坐标轴
        axes location Off

        # 背景为白色
        color Display Background white

        # Adjust view and reset
        display resetview

        display rendermode GLSL

        # Render the scene using Tachyon
        for {{set i 0}} {{$i < [molinfo top get numframes]}} {{incr i}} {{
            animate goto $i
            render TachyonInternal "{tmp_dir}/frame[format "%04d" $i].tga width 1024 height 1024 dpi 600"
        }}
"""
    file_name = f"{tmp_dir}/render_{start}_{end}.tcl"
    with open(file_name, "w") as f:
        f.write(tcl_script)
    os.system(f'vmd -dispdev text -e {file_name}')

def parallel_rendering(total_frames, num_processes,pdbfile,xtcfile,tmp_dir,out_dir):
    chunk_size = total_frames // num_processes
    tasks = [
        (i * chunk_size, (i + 1) * chunk_size, tmp_dir, pdbfile, xtcfile)
        for i in range(num_processes)
    ]
    # 最后一块包含所有剩余帧
    tasks[-1] = (tasks[-1][0], total_frames, tmp_dir, pdbfile, xtcfile)

    with Pool(num_processes) as pool:
        pool.starmap(render_frames, tasks)
        

def make_video_vmd(pdbfile,xtcfile,tmp_dir,out_dir):
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

    # 居中并缩放蛋白质
    set sel [atomselect top "protein"]
    set center [measure center $sel]
    set size [measure minmax $sel]
    set span [vecsub [lindex $size 1] [lindex $size 0]]
    translate by [vecscale -1 $center]
    set max_span [lindex [lsort -real $span] end]
    scale by [expr 1.0 / $max_span]

    # 删除坐标轴
    axes location Off

    # 背景为白色
    color Display Background white

    # Adjust view and reset
    display resetview

    display rendermode GLSL

    # Render the scene using Tachyon
    for {{set i 0}} {{$i < [molinfo top get numframes]}} {{incr i 100}} {{
        animate goto $i
        set datfile "{tmp_dir}/frame[format "%04d" $i].dat"
        set imgfile "{tmp_dir}/frame[format "%04d" $i].bmp"
        # 使用外部 Tachyon 渲染器输出 BMP 格式图片
        render Tachyon $datfile /usr/local/lib/vmd/tachyon_LINUXAMD64 $datfile\
            -o $imgfile -format BMP -res 2048 2048 -aasamples 24
    }}
    quit
    """

    # 写入到一个 .tcl 文件
    file_name = os.path.join(tmp_dir, "script.tcl")  # 文件名
    with open(file_name, "w") as file:
        file.write(tcl_script)
    # os.system(f'vmd -dispdev text -e {file_name} > /dev/null 2>&1')
    os.system(f'vmd -dispdev text -e {file_name} > /dev/null 2>&1')
    framerate = 15                      # 帧率（可根据你 VMD 输出频率调整）
    output_file = os.path.join(out_dir, "output_lossless.mp4")
    ffmpeg_path = "ffmpeg"              # 如果 ffmpeg 已加入 PATH，保持默认即可
    cmd = [
        ffmpeg_path,
        "-framerate", str(framerate),
        "-pattern_type", "glob",
        "-i", os.path.join(tmp_dir, "frame*.bmp"),
        "-c:v", "libx264",
        "-preset", "slow",       # 可选 ultrafast/veryfast/faster/slow/slower
        "-crf", "18",            # 0=无损, 18≈视觉无损
        "-pix_fmt", "yuv420p",   # 兼容性最好
        "-movflags", "+faststart",  # 确保播放器可读取
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 视频合成完成：{output_file}")
    except subprocess.CalledProcessError as e:
        print("\n❌ FFmpeg 执行失败！")
        print("错误信息:", e)
    
    
def process_pdb(pdbid):
    out_dir = os.path.join(out_path,pdbid)
    tmp_dir = os.path.join(tmp_path,pdbid)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(3):
        os.makedirs(tmp_dir, exist_ok=True)
        out_sub = os.path.join(out_dir,f'R{i+1}')
        os.makedirs(out_sub, exist_ok=True)
        make_video_vmd(os.path.join(raw_path, pdbid, 'protein', pdbid+'.pdb'), os.path.join(raw_path,pdbid, 'protein', pdbid+f'_prod_R{i+1}_fit.xtc'), tmp_dir, out_sub)
        # parallel_rendering(10001,48,os.path.join(tmp_dir,pdbid+'.pdb'),os.path.join(tmp_dir,pdbid+'_R1.xtc'),tmp_dir,out_dir)
        # process_folder(tmp_dir, out_dir)
        os.system(f'rm -rf {tmp_dir}')

        
import os
import time
import subprocess
from multiprocessing import Process

raw_path = 'autodl-tmp/ATLAS_protein_only/ATLAS'
gpu_list = [0, 1, 2]  # 三张GPU编号
max_parallel = len(gpu_list)  # 控制最多同时执行3个任务

def run_process_with_gpu(pdbid, gpu_id):
    """
    调用 process_pdb，在指定 GPU 上运行
    """
    # 设置仅可见的 GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"🚀 启动任务 {pdbid}，使用 GPU {gpu_id}")
    cmd = ["python", "make_video.py", pdbid]  # 调用自身脚本（或替换为主脚本路径）
    subprocess.run(cmd, env=env)
    print(f"✅ 任务 {pdbid} 在 GPU {gpu_id} 完成")

def schedule_jobs():
    files = os.listdir(raw_path)
    pdbids = [f for f in files if os.path.isdir(os.path.join(raw_path, f))]

    active_processes = []
    gpu_idx = 0

    for pdbid in pdbids:
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
    
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pdbid", nargs='?', default=None, help="单独处理指定的蛋白编号")
    args = parser.parse_args()
    if args.pdbid:
        process_pdb(args.pdbid)
    else:
        schedule_jobs()