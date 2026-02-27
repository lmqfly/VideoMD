# DiffSynth-Studio

<a href="https://github.com/modelscope/DiffSynth-Studio"><img src=".github/workflows/logo.gif" title="Logo" style="max-width:100%;" width="55" /></a>

[![PyPI](https://img.shields.io/pypi/v/DiffSynth)](https://pypi.org/project/DiffSynth/)
[![license](https://img.shields.io/github/license/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/blob/master/LICENSE)

[English](#introduction) | [中文](#简介)

## Introduction

DiffSynth-Studio is an open-source Diffusion model engine by [ModelScope](https://www.modelscope.cn/). This repository uses **Wan2.1-I2V-14B-720P** for image-to-video LoRA fine-tuning (e.g. protein dynamics video). The following sections describe **environment setup**, **training**, and **inference/testing** based on the project's `run.txt`.

---

## 简介

本仓库基于 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)，使用 **Wan2.1-I2V-14B-720P** 进行图生视频（I2V）的 LoRA 微调，适用于蛋白质分子动力学等视频数据。下文说明**环境配置**、**训练**与**测试/推理**流程，主要参考项目中的 `run.txt`。

---

## 环境配置

### 1. 克隆与安装

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

或在本仓库目录下直接执行：

```bash
cd /home/xujunzhang/mingquan/DiffSynth-Studio-main
pip install -e .
```

### 2. 依赖说明

主要依赖见 `requirements.txt`，包括但不限于：

- `torch` >= 2.0.0、`torchvision`
- `transformers`、`safetensors`、`accelerate`、`peft`
- `imageio`、`imageio[ffmpeg]`、`modelscope`、`pandas`

若安装出现问题，可参考 [PyTorch](https://pytorch.org/get-started/locally/)、[sentencepiece](https://github.com/google/sentencepiece)、[cmake](https://cmake.org) 等上游文档。

### 3. 多卡训练：Accelerate

多卡训练使用 `accelerate launch`。首次使用建议配置一次：

```bash
accelerate config
```

按提示选择 GPU 数量、是否使用混合精度等。训练时通过 `CUDA_VISIBLE_DEVICES` 指定使用的 GPU。

---

## 数据准备

- **视频目录**：训练用的视频文件所在路径，如 `video_data/video_cropped_renamed`，每个样本为一条视频（如 `.mp4`）。
- **元数据表**：CSV 文件，需包含训练脚本所要求的列（如 `video` 等指向视频文件名或相对路径）。可使用项目内的 `video_data/gen_meta_data.py` 根据视频目录和 `prompts.csv` 生成 `metadata.csv`。

训练时通过以下参数指定：

- `--dataset_base_path`：视频所在根目录。
- `--dataset_metadata_path`：元数据 CSV 的路径（如 `video_data/metadata.csv`）。

分辨率与帧数需与数据一致，并在训练/推理时统一（见下方参数说明）。

---

## 训练

### 单卡训练示例

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /home/xujunzhang/mingquan/DiffSynth-Studio-main/video_data/video_cropped_renamed \
  --dataset_metadata_path /home/xujunzhang/mingquan/DiffSynth-Studio-main/video_data/metadata.csv \
  --height 720 \
  --width 1280 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-720P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-720P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-720P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-720P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-I2V-14B-720P_lora_protein_all_data_mgpu" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing_offload
```

### 多卡训练示例（参考 run.txt）

使用 GPU 2 与 6，后台运行并写日志：

```bash
CUDA_VISIBLE_DEVICES=2,6 nohup setsid accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /home/xujunzhang/mingquan/DiffSynth-Studio-main/video_data/video_cropped_renamed \
  --dataset_metadata_path /home/xujunzhang/mingquan/DiffSynth-Studio-main/video_data/metadata.csv \
  --height 720 \
  --width 1280 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-720P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-720P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-720P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-720P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-I2V-14B-720P_lora_protein_all_data_mgpu" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing_offload \
  > train_protein_v2_lora_multiGPU.log 2>&1 &
```

可根据实际 GPU 数量修改 `CUDA_VISIBLE_DEVICES`（如 `2,5` 或 `0,1,2,3`）。

### 主要训练参数说明

| 参数 | 含义 | 示例/默认 |
|------|------|-----------|
| `--dataset_base_path` | 视频数据根目录 | 必填 |
| `--dataset_metadata_path` | 元数据 CSV 路径 | 必填 |
| `--height` / `--width` | 视频高/宽（需与数据一致） | 720 / 1280 |
| `--num_frames` | 每条视频采样帧数（需满足时间维约束，如 4n+1） | 49 |
| `--dataset_repeat` | 每 epoch 重复数据集次数 | 1 |
| `--model_id_with_origin_paths` | 基座模型 ID 与权重文件模式，逗号分隔 | 见上 |
| `--learning_rate` | 学习率 | 1e-4 |
| `--num_epochs` | 训练轮数 | 5 |
| `--output_path` | 检查点保存目录 | `./models/train/...` |
| `--remove_prefix_in_ckpt` | 保存 LoRA 时去掉的 key 前缀 | `pipe.dit.` |
| `--lora_base_model` | 挂载 LoRA 的模块名 | `dit` |
| `--lora_target_modules` | LoRA 作用子模块 | `q,k,v,o,ffn.0,ffn.2` |
| `--lora_rank` | LoRA 秩 | 32 |
| `--extra_inputs` | 额外输入（I2V 需首帧图） | `input_image` |
| `--use_gradient_checkpointing_offload` | 梯度检查点卸载到 CPU，省显存 | 建议多卡/大模型时开启 |

---

## 测试 / 推理

训练得到的 LoRA 权重会保存在 `--output_path` 下（如按 epoch 保存为 `epoch-3.safetensors`）。使用 **Wan2.1-I2V-14B-720P** 推理脚本加载基座模型并挂载 LoRA 进行图生视频。

### 单卡推理示例（参考 run.txt）

```bash
CUDA_VISIBLE_DEVICES=2 nohup python ./examples/wanvideo/model_inference/Wan2.1-I2V-14B-720P.py &
```

脚本内会：

1. 从 TSV（如 `split/splits_cdhit/mini_filtered.tsv`）读取 train/test 的 PDB 列表并采样；
2. 使用 `WanVideoPipeline.from_pretrained` 加载 **Wan-AI/Wan2.1-I2V-14B-720P** 基座；
3. 使用 `pipe.load_lora(pipe.dit, "<LoRA 路径>", alpha=1)` 加载训练好的 LoRA（如 `./models/train/Wan2.1-I2V-14B-720P_lora_protein_all_data/epoch-3.safetensors`）；
4. 对每个样本取首帧或指定帧作为 `input_image`，调用 `pipe(...)` 生成视频并保存。

### 推理脚本内可修改项

- **LoRA 路径**：在 `Wan2.1-I2V-14B-720P.py` 中修改 `load_lora` 的路径与 `alpha`。
- **输入列表**：修改 TSV 路径或采样逻辑以更换 test/train 样本。
- **分辨率与帧数**：`height=720, width=1280`、`num_frames`（如 49）需与训练一致；帧数需满足 4 的倍数 +1（如 49、81、101）。
- **保存路径**：脚本内 `base_save_dir` 等可改为你的输出目录。

---

## 注意事项

1. **显存**：14B 720P 模型与 49 帧训练显存占用较大，多卡时建议开启 `--use_gradient_checkpointing_offload`；单卡推理可配合 `pipe.enable_vram_management()` 与 `offload_device="cpu"`。
2. **数据格式**：metadata CSV 的列名和路径需与 `UnifiedDataset` 的 `data_file_keys` 等约定一致（默认包含 `video` 等）。
3. **帧数约束**：`num_frames` 需满足管线要求（如 4 的倍数 +1），否则可能报错或自动截断。
4. **基座权重**：首次运行会从 ModelScope/HuggingFace 拉取 Wan2.1-I2V-14B-720P 权重，请保证网络可访问或已配置镜像。

更多模型与用法请参考官方 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 与 [examples/wanvideo/README_zh.md](./examples/wanvideo/README_zh.md)。
