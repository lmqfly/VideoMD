import os
import csv

def generate_csv():
    # ================= 配置区域 =================
    # 视频文件夹路径
    video_folder = '/home/xujunzhang/mingquan/DiffSynth-Studio-main/protein_1280x720'
    
    # 划分文件的路径 (请确保此路径正确)
    split_file_path = '/home/xujunzhang/mingquan/DiffSynth-Studio-main/split/splits_cdhit/mini_filtered.tsv'
    
    # 输出的文件名 (将保存在 video_folder 下)
    output_csv_name = '/home/xujunzhang/mingquan/DiffSynth-Studio-main/split/splits_cdhit/metadata.csv'
    
    # 统一的 Prompt 文本
    fixed_prompt = "Simulations were performed on proteins only, using the all-atom CHARMM36m force field (July 2020) in explicit TIP3P water within a periodic triclinic box, with Na⁺/Cl⁻ ions added to neutralise the system at 150 mM, and a production run length of 50 ns."
    
    # 支持的视频格式后缀 (不区分大小写)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
    # ===========================================

    output_csv_path = os.path.join(video_folder, output_csv_name)

    # 1. 读取 split_data.tsv，获取所有属于 train 的 PDB ID
    train_pdbs = set()
    if not os.path.exists(split_file_path):
        print(f"❌ 错误：找不到划分文件 {split_file_path}")
        return

    print(f"正在读取划分文件: {split_file_path} ...")
    with open(split_file_path, 'r', encoding='utf-8') as f:
        # 假设是制表符分隔的 TSV
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # 确保列名匹配，且去除可能存在的空格
            if row['split'].strip() == 'train':
                train_pdbs.add(row['PDB'].strip())
    
    print(f"✅ 已加载 {len(train_pdbs)} 个训练集 PDB ID。")

    # 2. 遍历视频文件夹并筛选
    if not os.path.exists(video_folder):
        print(f"❌ 错误：找不到视频文件夹 {video_folder}")
        return

    files = sorted(os.listdir(video_folder))
    video_count = 0
    skipped_count = 0

    print(f"正在生成 {output_csv_path} ...")
    
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        writer.writerow(['video', 'prompt'])
        
        for filename in files:
            # 检查后缀
            if filename.lower().endswith(video_extensions):
                # 解析文件名 PDB_R{i}.mp4
                # 1. 去掉后缀 -> PDB_R{i}
                name_no_ext = os.path.splitext(filename)[0]
                
                # 2. 根据 '_R' 分割获取 PDB ID
                # 如果文件名中包含 _R，取前半部分
                if "_R" in name_no_ext:
                    pdb_id = name_no_ext.split('_R')[0]
                    
                    # 3. 检查是否在训练集中
                    if pdb_id in train_pdbs:
                        writer.writerow([filename, fixed_prompt])
                        video_count += 1
                        # print(f"已添加 (Train): {filename}")
                    else:
                        skipped_count += 1
                        # print(f"跳过 (非Train): {filename}")
                else:
                    print(f"⚠️ 警告：文件名格式不符合 PDB_R{{i}} 规范，已跳过: {filename}")

    print("-" * 30)
    print(f"✅ 处理完成！")
    print(f"📥 训练集 PDB 总数: {len(train_pdbs)}")
    print(f"📂 写入视频数量: {video_count}")
    print(f"🚫 跳过视频数量: {skipped_count} (非训练集或格式错误)")
    print(f"📄 结果已保存为: {output_csv_path}")

if __name__ == '__main__':
    generate_csv()