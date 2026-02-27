#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据 PDB 列，检查 target_cropped/<PDB>_R1.mp4 是否存在，存在则保留该行。
使用方法示例：
    python filter_by_video.py \
        --input mini.tsv \
        --video_dir target_cropped \
        --output_dir filtered \
        --output_name mini_filtered.tsv
"""

import os
import argparse
import pandas as pd


def main(args):
    input_path = args.input
    video_dir = args.video_dir
    output_dir = args.output_dir
    output_name = args.output_name

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入 TSV 文件不存在：{input_path}")

    if not os.path.isdir(video_dir):
        raise NotADirectoryError(f"视频目录不存在或不是目录：{video_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    # 读取 TSV
    df = pd.read_csv(input_path, sep="\t", dtype=str)
    if "PDB" not in df.columns:
        raise ValueError("TSV 文件中未找到 PDB 列，请检查表头。")

    # 去除首尾空白
    df["PDB"] = df["PDB"].astype(str).str.strip()

    # 为每一行构造对应视频路径，并检查是否存在
    def has_video(pdb_id: str) -> bool:
        if not pdb_id:
            return False
        filename = f"{pdb_id}_R1.mp4"
        full_path = os.path.join(video_dir, filename)
        return os.path.exists(full_path)

    mask = df["PDB"].apply(has_video)

    kept_df = df[mask].copy()
    removed_count = len(df) - len(kept_df)

    print(f"总行数：{len(df)}")
    print(f"保留行数（存在对应 mp4）：{len(kept_df)}")
    print(f"被过滤掉的行数：{removed_count}")

    # 保存过滤后的 TSV
    kept_df.to_csv(output_path, sep="\t", index=False)
    print(f"过滤后的结果已保存到：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据 PDB 列匹配 target_cropped/<PDB>_R1.mp4 是否存在来过滤 TSV 行"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="输入原始 TSV 文件路径（如 mini.tsv）")
    parser.add_argument("--video_dir", type=str, default="target_cropped",
                        help="存放 mp4 文件的目录（默认：target_cropped）")
    parser.add_argument("--output_dir", type=str, default="filtered",
                        help="过滤后的 TSV 输出目录（默认：filtered）")
    parser.add_argument("--output_name", type=str, default="mini_filtered.tsv",
                        help="过滤后 TSV 文件名（默认：mini_filtered.tsv）")

    args = parser.parse_args()
    main(args)

# python filter.py --input /home/xujunzhang/mingquan/DiffSynth-Studio-main/split/2023_03_09_ATLAS_info.tsv --video_dir /home/xujunzhang/mingquan/DiffSynth-Studio-main/protein_1280x720 --output_dir ./ --output_name mini_filtered.tsv