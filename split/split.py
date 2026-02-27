#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对 mini.tsv 做全面分析，并生成多种数据集划分方案。

使用方法：
    python analyze_mini_tsv.py --input mini.tsv --outdir splits

依赖：
    pip install pandas numpy scikit-learn
"""

import os
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 一些字段名常量，避免手写错误
PDB_COL = "PDB"
SEQ_COL = "sequence"
LENGTH_COL = "length"
NR_PROT_COL = "non_redundant_protein"
NR_DOM_COL = "non_redundant_domain"
CATH_SUPFAM_COL = "CATH_supfamily"
ECOD_DOMAIN_COL = "ECOD_domain_ID"
SCOP_CLASS_COL = "SCOP_class"
ALPHA_COL = "alpha%"
BETA_COL = "beta%"
COIL_COL = "coil%"

BOOL_COLS_CANDIDATES = [
    "contact_chain", "contact_ligand", "contact_ion",
    "contact_nucleotide", "no_contact",
    "non_redundant_protein", "non_redundant_domain"
]


def read_tsv(path: str) -> pd.DataFrame:
    """
    读取 TSV 文件，做基础清洗：
    - 删除每个单元格首尾的引号、空白
    - 尝试自动转换布尔/数值
    """
    df = pd.read_csv(path, sep="\t", dtype=str)  # 先全部以字符串读入

    # 去首尾空格和多余引号
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r'^"+', '', regex=True)
            .str.replace(r'"+$', '', regex=True)
        )

    # 转换布尔字段：True/False -> bool
    for col in BOOL_COLS_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].map(
                {"True": True, "False": False, "true": True, "false": False}
            )

    # 尝试将部分列转换为数值（如果可能）
    numeric_candidates = [
        "PDB_resolution", "refinement_TMscore", "length",
        "alpha%", "beta%", "coil%",
        "div_SE", "div_MM", "avg_RMSF", "avg_gyration"
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def basic_info(df: pd.DataFrame):
    print("=== 基本信息 ===")
    print(f"总行数（条目数）：{len(df)}")
    print(f"总列数：{df.shape[1]}")
    print("\n列名：")
    print(", ".join(df.columns))
    print("\n前 3 行：")
    print(df.head(3))
    print("\n数据信息：")
    print(df.info())


def missing_values_report(df: pd.DataFrame):
    print("\n=== 缺失值统计 ===")
    na_counts = df.isna().sum()
    na_ratio = na_counts / len(df)
    report = pd.DataFrame({
        "missing_count": na_counts,
        "missing_ratio": na_ratio
    }).sort_values("missing_ratio", ascending=False)
    print(report)


def numeric_summary(df: pd.DataFrame, max_cols: int = 30):
    print("\n=== 数值列统计概览 ===")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("无数值列")
        return
    # 仅打印前 max_cols 列，避免太长
    summary = numeric_df.describe().T
    if summary.shape[0] > max_cols:
        summary = summary.head(max_cols)
    print(summary)


def categorical_summary(df: pd.DataFrame,
                        max_unique: int = 50,
                        top_k: int = 10):
    print("\n=== 分类列（非数值）统计（前若干列）===")
    non_num_df = df.select_dtypes(exclude=[np.number, "bool"])
    for col in non_num_df.columns:
        nunique = non_num_df[col].nunique(dropna=True)
        if nunique == 0:
            continue
        print(f"\n列：{col}（不同值数量 = {nunique}）")
        # 对特别高基数的列，只看 top_k
        value_counts = non_num_df[col].value_counts(dropna=False)
        if nunique > max_unique:
            print(f"仅显示前 {top_k} 个：")
            print(value_counts.head(top_k))
        else:
            print(value_counts)


def estimate_domain_count(col: pd.Series) -> pd.Series:
    """
    简单估计每条记录 domain 个数：
    - 按逗号分割字符串元素
    - 非空部分个数作为 domain 个数
    """
    def count_domains(val):
        if pd.isna(val) or val == "-":
            return 0
        parts = [x.strip() for x in str(val).split(",") if x.strip()]
        return len(parts)

    return col.apply(count_domains)


def structural_hierarchy_analysis(df: pd.DataFrame):
    print("\n=== 结构层级标签分析 ===")

    if ECOD_DOMAIN_COL in df.columns:
        dom_counts = estimate_domain_count(df[ECOD_DOMAIN_COL])
        print(f"ECOD_domain_ID：平均 domain 数 = {dom_counts.mean():.2f}, "
              f"最大 domain 数 = {dom_counts.max()}, 最小 = {dom_counts.min()}")

    if CATH_SUPFAM_COL in df.columns:
        print(f"\nCATH_supfamily（结构超家族）统计：")
        cath_sup = df[CATH_SUPFAM_COL].fillna("-")
        vc = cath_sup.value_counts()
        print(f"不同 CATH 超家族数量：{len(vc)}")
        print("前 10 个超家族及其计数：")
        print(vc.head(10))

    if SCOP_CLASS_COL in df.columns:
        print(f"\nSCOP_class（折叠类别）统计：")
        scop_cls = df[SCOP_CLASS_COL].fillna("-")
        print(scop_cls.value_counts())


def sequence_and_length_analysis(df: pd.DataFrame):
    print("\n=== 序列与长度分析 ===")
    if SEQ_COL not in df.columns:
        print("未找到 sequence 列，跳过。")
        return

    seq_len = df[SEQ_COL].astype(str).str.len()
    df["_seq_len"] = seq_len

    if LENGTH_COL in df.columns and df[LENGTH_COL].notna().any():
        length = df[LENGTH_COL]
        mism = (length != seq_len)
        mismatch_count = mism.sum()
        print(f"length 与 实际 sequence 长度不一致的条目数：{mismatch_count}")
        if mismatch_count > 0:
            print("示例（前 5 条不一致记录）：")
            print(df.loc[mism, [PDB_COL, LENGTH_COL, "_seq_len"]].head(5))

    print("\n序列长度统计：")
    print(seq_len.describe())

    # 简单桶化长度
    bins = [0, 100, 200, 300, 1000]
    labels = ["<100", "100-199", "200-299", ">=300"]
    len_bin = pd.cut(seq_len, bins=bins, labels=labels, right=False)
    print("\n长度区间分布：")
    print(len_bin.value_counts().sort_index())


def secondary_structure_analysis(df: pd.DataFrame):
    print("\n=== 二级结构比例分析（alpha%, beta%, coil%） ===")
    for col in [ALPHA_COL, BETA_COL, COIL_COL]:
        if col not in df.columns:
            print(f"缺少列 {col}，跳过。")
            return

    # 三者和的分布
    ss_sum = df[ALPHA_COL] + df[BETA_COL] + df[COIL_COL]
    print("alpha+beta+coil 的统计：")
    print(ss_sum.describe())
    inconsistent = (np.abs(ss_sum - 100) > 5) & ss_sum.notna()
    print(f"alpha+beta+coil 与 100 差超过 5 的条目数：{inconsistent.sum()}")

    print("\nalpha%, beta%, coil% 分布（describe）：")
    print(df[[ALPHA_COL, BETA_COL, COIL_COL]].describe())


def contact_features_analysis(df: pd.DataFrame):
    print("\n=== 接触特征（配体/离子/核苷酸/无接触）统计 ===")
    for col in ["contact_ligand", "contact_ion", "contact_nucleotide", "no_contact"]:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            print(f"\n{col}：")
            print(vc)


# ========= 数据集划分部分 =========

def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)


def save_split(df: pd.DataFrame,
               split_series: pd.Series,
               out_path: str,
               extra_cols: List[str] = None):
    """
    将划分结果保存为 CSV:
    - 至少包含 PDB, split 两列
    - 可附加一些额外列供检查
    """
    cols = [PDB_COL, "split"]
    tmp = pd.DataFrame({
        PDB_COL: df[PDB_COL],
        "split": split_series
    })
    if extra_cols:
        for c in extra_cols:
            if c in df.columns:
                tmp[c] = df[c]
    tmp.to_csv(out_path, index=False)
    print(f"已保存划分文件：{out_path}")


def random_split(df: pd.DataFrame,
                 train_ratio=0.8,
                 valid_ratio=0.1,
                 test_ratio=0.1,
                 seed=42) -> pd.Series:
    """
    简单随机划分，输出与 df 等长的 split Series。
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

    n = len(df)
    indices = np.arange(n)
    train_idx, temp_idx = train_test_split(indices, test_size=(1 - train_ratio),
                                           random_state=seed, shuffle=True)
    valid_rel = valid_ratio / (valid_ratio + test_ratio)
    valid_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - valid_rel), random_state=seed, shuffle=True
    )

    split = pd.Series(index=df.index, dtype="object")
    split.iloc[train_idx] = "train"
    split.iloc[valid_idx] = "valid"
    split.iloc[test_idx] = "test"
    return split


def length_stratified_split(df: pd.DataFrame,
                            train_ratio=0.8,
                            valid_ratio=0.1,
                            test_ratio=0.1,
                            seed=42) -> pd.Series:
    """
    按长度桶 stratify 再在每桶内随机划分。
    """
    if LENGTH_COL not in df.columns and SEQ_COL not in df.columns:
        print("无法按长度分层（length/sequence 均不存在），退化为随机划分。")
        return random_split(df, train_ratio, valid_ratio, test_ratio, seed)

    if LENGTH_COL in df.columns and df[LENGTH_COL].notna().any():
        length = df[LENGTH_COL].fillna(df[LENGTH_COL].median())
    else:
        length = df[SEQ_COL].astype(str).str.len()

    # 定义三个桶：短、中、长
    quantiles = np.quantile(length, [0.33, 0.66])
    # 避免相同边界引发错误
    q1, q2 = quantiles
    if q1 == q2:
        q1 = length.median()
        q2 = q1 + 1

    def bucket(l):
        if l <= q1:
            return "short"
        elif l <= q2:
            return "medium"
        return "long"

    buckets = length.apply(bucket)
    split = pd.Series(index=df.index, dtype="object")

    for b in ["short", "medium", "long"]:
        idx = df.index[buckets == b].to_numpy()
        if len(idx) == 0:
            continue
        sub_df = df.loc[idx]
        sub_split = random_split(sub_df, train_ratio, valid_ratio, test_ratio, seed)
        split.loc[idx] = sub_split

    return split


def group_based_split(df: pd.DataFrame,
                      group_col: str,
                      train_ratio=0.8,
                      valid_ratio=0.1,
                      test_ratio=0.1,
                      seed=42,
                      min_group_size: int = 1) -> pd.Series:
    """
    以 group_col 为单元进行划分，确保同组不跨 train/valid/test。
    - group_col 中的不同值视为不同 group（无论是否只出现一次）
    - 可设置 min_group_size，太小的 group 可合并逻辑（当前简单保留）
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6
    if group_col not in df.columns:
        raise ValueError(f"列 {group_col} 不存在，无法按 group 划分")

    groups = df[group_col].fillna(f"Unknown_{group_col}")
    # 唯一 group 列表
    unique_groups = groups.unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = int(n_groups * train_ratio)
    n_valid = int(n_groups * valid_ratio)
    # 剩余给 test
    n_test = n_groups - n_train - n_valid

    train_groups = set(unique_groups[:n_train])
    valid_groups = set(unique_groups[n_train:n_train + n_valid])
    test_groups = set(unique_groups[n_train + n_valid:])

    split = pd.Series(index=df.index, dtype="object")
    for idx, g in groups.items():
        if g in train_groups:
            split.at[idx] = "train"
        elif g in valid_groups:
            split.at[idx] = "valid"
        elif g in test_groups:
            split.at[idx] = "test"
        else:
            split.at[idx] = "train"  # 极小概率逻辑兜底

    return split


def example_leave_one_group_out(df: pd.DataFrame,
                                group_col: str,
                                seed: int = 42) -> Dict[str, pd.Series]:
    """
    演示如何制作 "leave-one-group-out" 划分：
    返回一个 dict，key 是 group 名，value 是对应的一次划分 Series。
    适合小数据集 / 用于结构泛化评估。
    """
    if group_col not in df.columns:
        raise ValueError(f"列 {group_col} 不存在，无法按 group leave-one-out")

    groups = df[group_col].fillna(f"Unknown_{group_col}")
    unique_groups = groups.unique()

    splits = {}
    for g in unique_groups:
        split = pd.Series(index=df.index, dtype="object")
        # 将该 group 作为 test，其余都作为 train（也可以再划分 valid）
        is_test = (groups == g)
        split[is_test] = "test"
        split[~is_test] = "train"
        splits[g] = split

    return splits


def main(args):
    df = read_tsv(args.input)

    # ========= 分析部分 =========
    print("#" * 80)
    basic_info(df)
    print("#" * 80)
    missing_values_report(df)
    print("#" * 80)
    numeric_summary(df)
    print("#" * 80)
    categorical_summary(df)
    print("#" * 80)
    structural_hierarchy_analysis(df)
    print("#" * 80)
    sequence_and_length_analysis(df)
    print("#" * 80)
    secondary_structure_analysis(df)
    print("#" * 80)
    contact_features_analysis(df)
    print("#" * 80)

    # ========= 划分部分 =========
    ensure_outdir(args.outdir)

    # 1. 简单随机划分
    split_random = random_split(df, seed=args.seed)
    save_split(df, split_random,
               os.path.join(args.outdir, "split_random.csv"),
               extra_cols=[LENGTH_COL, CATH_SUPFAM_COL, NR_PROT_COL])

    # 2. 按长度分层的随机划分
    split_len = length_stratified_split(df, seed=args.seed)
    save_split(df, split_len,
               os.path.join(args.outdir, "split_length_stratified.csv"),
               extra_cols=[LENGTH_COL])

    # 3. 以 non_redundant_protein 为单位划分（适合评估蛋白级泛化）
    if NR_PROT_COL in df.columns:
        split_nr_prot = group_based_split(df, NR_PROT_COL, seed=args.seed)
        save_split(df, split_nr_prot,
                   os.path.join(args.outdir, "split_non_redundant_protein.csv"),
                   extra_cols=[NR_PROT_COL])

    # 4. 以 CATH_supfamily 为单位划分（适合评估跨结构家族泛化）
    if CATH_SUPFAM_COL in df.columns:
        split_cath = group_based_split(df, CATH_SUPFAM_COL, seed=args.seed)
        save_split(df, split_cath,
                   os.path.join(args.outdir, "split_CATH_supfamily.csv"),
                   extra_cols=[CATH_SUPFAM_COL])

    # 5. Leave-one-structure-family-out 示例（不全部写文件，只展示一个例子）
    if CATH_SUPFAM_COL in df.columns:
        print("\n示例：针对某个 CATH_supfamily 做 leave-one-group-out 划分：")
        loo_splits = example_leave_one_group_out(df, CATH_SUPFAM_COL)
        # 仅示例打印其中一个 key
        example_key = list(loo_splits.keys())[0]
        print(f"示例超家族：{example_key}")
        print(loo_splits[example_key].value_counts())

    print("\n分析与划分完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="对 mini.tsv 进行全面分析并生成多种数据集划分方案"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="输入的 TSV 文件路径，如 mini.tsv")
    parser.add_argument("--outdir", type=str, default="splits",
                        help="划分结果输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()
    main(args)

# python split.py --input /home/xujunzhang/mingquan/DiffSynth-Studio-main/split/mini_filtered.tsv --outdir ./splits