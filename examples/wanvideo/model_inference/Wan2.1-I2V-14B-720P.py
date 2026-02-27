import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import os
import imageio
import pandas as pd
import random

# 读取 TSV 文件
tsv_path = "/home/xujunzhang/mingquan/DiffSynth-Studio-main/split/splits_cdhit/mini_filtered.tsv"
df = pd.read_csv(tsv_path, sep='\t')

# 按 split 分组
train_samples = df[df['split'] == 'train']['PDB'].tolist()
test_samples = df[df['split'] == 'test']['PDB'].tolist()

# 从 train 和 test 中分别随机采样 5 个样本
random.seed(42)  # 设置随机种子以确保可复现性
sampled_train = random.sample(train_samples, min(20, len(train_samples)))
sampled_test = random.sample(test_samples, min(20, len(test_samples)))

print(f"从 train 中采样了 {len(sampled_train)} 个样本: {sampled_train}")
print(f"从 test 中采样了 {len(sampled_test)} 个样本: {sampled_test}")

# 创建 pdbid 到 split 的映射
pdbid_to_split = {}
for pdbid in sampled_train:
    pdbid_to_split[pdbid] = 'train'
for pdbid in sampled_test:
    pdbid_to_split[pdbid] = 'test'

# 合并所有要处理的 pdbids
pdbids = sampled_train + sampled_test

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
)
pipe.load_lora(pipe.dit, "./models/train/Wan2.1-I2V-14B-720P_lora_protein_all_data/epoch-3.safetensors", alpha=1)
pipe.enable_vram_management()

# 设置生成帧数（需要满足 4的倍数+1 的约束，如 49, 81, 101 等）
num_frames = 49  # 可以根据需要修改为 49, 81, 101 等

for pdbid in pdbids:
    image_path = f"/home/xujunzhang/mingquan/DiffSynth-Studio-main/protein_1280x720/{pdbid}_R1.mp4"

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}, skipping...")
        continue

    # 获取GT视频的fps和帧数
    gt_reader = imageio.get_reader(image_path)
    gt_fps = gt_reader.get_meta_data().get('fps', 15)  # 默认15fps
    gt_total_frames = gt_reader.count_frames()
    gt_reader.close()
    
    # 确保num_frames不超过GT视频的帧数
    actual_num_frames = min(num_frames, gt_total_frames)
    # 确保满足 4的倍数+1 的约束
    if actual_num_frames % 4 != 1:
        actual_num_frames = (actual_num_frames + 2) // 4 * 4 + 1
        if actual_num_frames > gt_total_frames:
            actual_num_frames = ((gt_total_frames - 1) // 4) * 4 + 1
    
    print(f"Processing {pdbid}: GT视频帧数={gt_total_frames}, fps={gt_fps:.2f}, 生成帧数={actual_num_frames}")

    image = VideoData(image_path, height=720, width=1280)[0]

    # Image-to-video
    video = pipe(
        prompt="Simulations were performed on proteins only, using the all-atom CHARMM36m force field (July 2020) in explicit TIP3P water within a periodic triclinic box, with Na⁺/Cl⁻ ions added to neutralise the system at 150 mM, and a production run length of 50 ns. Keep the same chain connectivity and backbone topology as the input frame.",
        negative_prompt="",
        input_image=image,
        seed=0, tiled=True,  
        height=720, width=1280,
        num_frames=actual_num_frames,  # 明确指定生成帧数
    )
    
    # 根据 pdbid 的 split 确定保存路径
    split = pdbid_to_split[pdbid]
    base_save_dir = f"/home/xujunzhang/mingquan/DiffSynth-Studio-main/protein_all_data_generated_epcho3"
    save_dir = f"{base_save_dir}/{split}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存生成的视频
    gen_save_path = f"{save_dir}/{pdbid}_gen.mp4"
    save_video(video, gen_save_path, fps=gt_fps, quality=5)
    print(f"Saved generated video: {gen_save_path} (帧数={len(video)}, fps={gt_fps:.2f})")
    
    # 保存GT视频（只取前actual_num_frames帧，确保与生成视频帧数一致）
    gt_video_data = VideoData(image_path, height=720, width=1280)
    gt_frames = [gt_video_data[i] for i in range(actual_num_frames)]
    gt_save_path = f"{save_dir}/{pdbid}_gt.mp4"
    save_video(gt_frames, gt_save_path, fps=gt_fps, quality=5)
    print(f"Saved GT video: {gt_save_path} (帧数={len(gt_frames)}, fps={gt_fps:.2f})")
