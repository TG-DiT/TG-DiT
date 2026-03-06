import os
import random
import numpy as np
from PIL import Image
import torch
from turb_sim.simulator import Simulator


# 配置
val_txt_path = "./val.txt"  # val.txt 路径 (6K 行，每行 "path class_id")
params_npy_path = "./turb_sim/data/turb_params.npy"  # params.npy 路径 (150000, 3)
val_output_dir = "/home/nvidia/MachineLearning/Data/val"  # 输出目录
img_size = 256  # 目标尺寸
num_samples = 6000  # 采样数
random.seed(42)  # 固定种子，确保复现
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 创建输出目录
os.makedirs(val_output_dir, exist_ok=True)

# 步骤1: 读取 val.txt，解析 paths 和 class_ids，并 shuffle
with open(val_txt_path, "r") as f:
    lines = f.read().splitlines()

paths = [line.split()[0] for line in lines if line.strip()]  # 提取路径
class_ids = [int(line.split()[-1]) for line in lines if line.strip()]  # 提取 class_id (1/2/3)
print(f"Loaded {len(paths)} validation paths and class_ids.")

# 同时 shuffle paths 和 class_ids（保持配对）
combined = list(zip(paths, class_ids))
random.shuffle(combined)
paths, class_ids = zip(*combined)
paths = list(paths)
class_ids = list(class_ids)

# 步骤2: 加载图像，resize 非 256x256 的到 256x256，存为列表
imgs = []
valid_paths = []  # 跟踪有效路径（用于保存）
valid_class_ids = []  # 跟踪有效 class_id
for path, class_id in zip(paths, class_ids):
    try:
        img = Image.open(path).convert("RGB")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), Image.BICUBIC)
        img_np = np.array(img).astype(np.uint8)
        imgs.append(img_np)
        valid_paths.append(path)
        valid_class_ids.append(class_id)
        print(f"Processed {os.path.basename(path)} (class {class_id}): shape {img_np.shape}")
    except Exception as e:
        print(f"Error processing {path}: {e}")
        continue

print(f"Processed {len(imgs)} images.")

# 步骤3: 加载 params.npy，等间隔采样 6000 个参数，然后 shuffle
params = np.load(params_npy_path)
print(f"Loaded params shape: {params.shape}")

M = len(params)
if M < num_samples:
    print(f"Warning: Params {M} < {num_samples}, using all with repetition.")
    indices = np.random.choice(M, num_samples, replace=True)  # 随机重复补齐
else:
    # 等间隔采样 6000 个索引
    indices = np.round(np.linspace(0, M - 1, num_samples)).astype(int)
    indices = np.unique(indices)  # 去重（linspace 可能有重复）
    if len(indices) < num_samples:
        extra = np.random.choice(M, num_samples - len(indices), replace=False)
        indices = np.concatenate([indices, extra])

selected_params = params[indices]
random.shuffle(selected_params)  # 打乱参数，与图像顺序匹配
print(f"Selected {len(selected_params)} params with uniform spacing and shuffled.")

# 匹配图像数
min_len = min(len(imgs), len(selected_params))
imgs = imgs[:min_len]
selected_params = selected_params[:min_len]
valid_paths = valid_paths[:min_len]
valid_class_ids = valid_class_ids[:min_len]
print(f"Matched to {min_len} samples.")

# 步骤4: 初始化 Simulator 并生成湍流图像
simulator = Simulator(img_size=img_size).to(device, dtype=torch.float32)
print(f"Simulator initialized on {device}.")

for idx, (img_np, param, class_id, path) in enumerate(zip(imgs, selected_params, valid_class_ids, valid_paths)):
    try:
        D, r0, L = param
        S = D / r0 if r0 != 0 else 0.0
        S_str = f"{S:.4f}"  # 格式化 S 到 4 位小数
        
        # 准备 tensor [1, C, H, W] [0,1]
        clean_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
        
        with torch.no_grad():
            simulator.update_dr0(S)
            high_turb_tensor = simulator(clean_tensor)
        
        # 转换回 [H, W, C] uint8 [0,255]
        high_turb_img = high_turb_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255.0
        high_turb_img = np.clip(high_turb_img, 0, 255).astype(np.uint8)
        
        # 修改：构建新文件名 original_类别_S.png
        base_name = os.path.basename(path).split('.')[0]  # 去扩展名 (e.g., "img" from "img.png")
        new_filename = f"{base_name}_{class_id}_{S_str}.png"
        save_path = os.path.join(val_output_dir, new_filename)
        Image.fromarray(high_turb_img).save(save_path)
        print(f"Saved high_turb as {new_filename} (class {class_id}, S={S:.4f})")
    except Exception as e:
        print(f"Error generating for {os.path.basename(path)}: {e}")
        continue

print("Generation completed. All high_turb images saved to val directory with naming _class_S.png.")