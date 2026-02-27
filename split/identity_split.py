#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import random
from typing import Dict, List, Tuple
import pandas as pd

# 常量定义
PDB_COL = "PDB"
SEQ_COL = "sequence"
TEMP_ID_COL = "temp_cdhit_id"  # 内部使用的临时唯一ID列

def write_fasta_with_unique_ids(df: pd.DataFrame, fasta_path: str):
    """
    使用 DataFrame 的索引作为唯一 ID 生成 FASTA，避免原始 PDB ID 重复或含空格的问题。
    """
    with open(fasta_path, "w") as f:
        for idx, row in df.iterrows():
            seq = str(row[SEQ_COL]).strip().replace(" ", "")
            # 使用 temp_id (即行号) 作为 FASTA header
            f.write(f">{row[TEMP_ID_COL]}\n")
            f.write(seq + "\n")

def run_cdhit(fasta_in: str, fasta_out: str, identity: float, threads: int):
    """
    运行 CD-HIT。自动推导 word_size。
    """
    # 简单的 word_size 自动选择逻辑
    if identity < 0.5:
        word_size = 2
    elif identity < 0.6:
        word_size = 3
    elif identity < 0.7:
        word_size = 4
    else:
        word_size = 5

    cmd = [
        "cd-hit",
        "-i", fasta_in,
        "-o", fasta_out,
        "-c", str(identity),
        "-n", str(word_size),
        "-T", str(threads),
        "-M", "0",  # 内存不限制
        "-d", "0"   # header 完整输出
    ]
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def parse_cdhit_clstr(clstr_path: str) -> Dict[str, int]:
    """
    解析 .clstr 文件。
    返回: { 'temp_id_str': cluster_id_int }
    """
    if not os.path.exists(clstr_path):
        raise FileNotFoundError(f"未找到聚类文件: {clstr_path}")

    seq_to_cluster = {}
    current_cluster = -1

    with open(clstr_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
            elif line:
                # 解析行: "0   200aa, >12345... *"
                # 我们只需要 '>' 后面的 ID，直到 '...' 或空格
                # 这里的 ID 是我们在 write_fasta 时生成的行号
                content = line.split(">")[1]
                seq_id = content.split("...")[0].strip()
                seq_to_cluster[seq_id] = current_cluster
    return seq_to_cluster

def split_clusters_greedy(cluster_to_samples: Dict[int, List[str]], 
                          ratios: List[float], 
                          seed: int) -> Dict[str, str]:
    """
    贪心算法划分簇，使样本数量比例尽可能接近 ratios。
    
    Args:
        cluster_to_samples: {cluster_id: [sample_id1, sample_id2, ...]}
        ratios: [train_ratio, valid_ratio, test_ratio] e.g., [0.8, 0.1, 0.1]
    
    Returns:
        sample_to_split: {sample_id: 'train'/'valid'/'test'}
    """
    # 1. 准备数据
    cluster_ids = list(cluster_to_samples.keys())
    # 按簇的大小降序排列，优先处理大簇，这样更容易平衡
    # (也可以随机打乱，但在极度不平衡时，降序通常效果更好，或者 shuffle 后贪心)
    random.seed(seed)
    random.shuffle(cluster_ids) 
    # 也可以尝试: cluster_ids.sort(key=lambda k: len(cluster_to_samples[k]), reverse=True)
    
    total_samples = sum(len(v) for v in cluster_to_samples.values())
    target_counts = [total_samples * r for r in ratios] # [Target_Train, Target_Valid, Target_Test]
    current_counts = [0, 0, 0]
    split_names = ["train", "valid", "test"]
    
    sample_to_split = {}

    # 2. 贪心分配
    for cid in cluster_ids:
        samples = cluster_to_samples[cid]
        n_s = len(samples)
        
        # 计算如果放入某集合，该集合离目标的差距（剩余需要多少）
        # 我们选择 "当前完成度最低" 或者 "放入后最不超标" 的集合
        # 这里使用简单的逻辑：选择 (current / target) 比例最小的集合
        
        best_split_idx = -1
        min_ratio = float('inf')
        
        for i in range(3):
            # 避免除以零
            t = target_counts[i] if target_counts[i] > 0 else 1e-9
            ratio = current_counts[i] / t
            if ratio < min_ratio:
                min_ratio = ratio
                best_split_idx = i
        
        # 分配
        split_label = split_names[best_split_idx]
        current_counts[best_split_idx] += n_s
        
        for sid in samples:
            sample_to_split[sid] = split_label

    print("\n划分结果统计 (样本数):")
    for i, name in enumerate(split_names):
        print(f"  {name}: {current_counts[i]} ({current_counts[i]/total_samples:.2%})")
        
    return sample_to_split

def main(args):
    # ... (路径检查逻辑同原代码) ...
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 读取数据
    df = pd.read_csv(args.input, sep="\t", dtype=str)
    df = df.dropna(subset=[SEQ_COL])
    df = df[df[SEQ_COL].str.strip() != ""].reset_index(drop=True)
    
    # 创建临时唯一ID（字符串格式的行号），用于 CD-HIT 交互
    df[TEMP_ID_COL] = df.index.astype(str)
    
    print(f"总样本数: {len(df)}")

    # 2. 生成 FASTA
    fasta_in = os.path.join(args.output_dir, "input.fasta")
    write_fasta_with_unique_ids(df, fasta_in)
    
    # 3. 运行 CD-HIT
    fasta_out = os.path.join(args.output_dir, "cdhit_out") # 注意这里不要加后缀，cdhit会自动加
    clstr_file = fasta_out + ".clstr"
    
    # 如果已经跑过且不强制重跑，可以跳过（这里为了演示每次都跑）
    run_cdhit(fasta_in, fasta_out, args.identity, args.threads)
    
    # 4. 解析聚类结果
    # 得到 { 'row_index_str': cluster_id }
    seq_to_cluster = parse_cdhit_clstr(clstr_file)
    
    # 5. 构建 Cluster -> [Sample IDs] 的映射
    cluster_to_samples = {}
    
    # 注意：CD-HIT 可能丢弃极短序列或异常序列，
    # 或者如果序列完全一致，CD-HIT 可能会把它们合并。
    # 我们需要遍历 DataFrame 确保每个样本都有归属
    # 如果 seq_to_cluster 里没有（极少情况），就当作独立簇处理
    
    max_cluster_id = max(seq_to_cluster.values()) if seq_to_cluster else 0
    
    for idx_str in df[TEMP_ID_COL]:
        if idx_str in seq_to_cluster:
            c_id = seq_to_cluster[idx_str]
        else:
            # 未被聚类的（可能是因为某些原因漏掉，或者作为代表序列被重命名等复杂情况）
            # 简单起见，分配新簇
            max_cluster_id += 1
            c_id = max_cluster_id
            
        if c_id not in cluster_to_samples:
            cluster_to_samples[c_id] = []
        cluster_to_samples[c_id].append(idx_str)
        
    print(f"共生成 {len(cluster_to_samples)} 个簇。")

    # 6. 执行贪心划分
    ratios = [args.train_ratio, args.valid_ratio, args.test_ratio]
    sample_split_map = split_clusters_greedy(cluster_to_samples, ratios, args.seed)
    
    # 7. 将划分结果映射回 DataFrame
    df["split"] = df[TEMP_ID_COL].map(sample_split_map)
    
    # 清理临时列
    df.drop(columns=[TEMP_ID_COL], inplace=True)
    
    # 8. 保存
    out_path = os.path.join(args.output_dir, args.output_name)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"完成。结果已保存至: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="splits_cdhit")
    parser.add_argument("--output_name", default="split_data.tsv")
    parser.add_argument("--identity", type=float, default=0.4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    main(args)