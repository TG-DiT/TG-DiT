import os
import random

# 配置
data_root = "/home/nvidia/MachineLearning/Data/"
categories = {
    "FFHQ": 1,      # 人脸 = 1
    "Lsun_airplane": 2,  # 飞行器 = 2
    "Lsun_church": 3     # 建筑物 = 3
}
train_size_per_cat = 50000  # 修改：每个类别 5 万训练
val_size_per_cat = 2000    # 修改：每个类别 2K 验证
img_extensions = ('.png', '.jpg', '.jpeg', '.bmp')  # 支持扩展名
random.seed(42)  # 固定种子，确保复现

# 函数：扫描文件夹获取所有图像路径
def get_image_paths(folder_path):
    img_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(img_extensions):
                img_paths.append(os.path.join(root, file))
    return img_paths

# 生成 txt
train_lines = []
val_lines = []

for folder_name, class_id in categories.items():
    folder_path = os.path.join(data_root, folder_name)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found!")
        continue
    
    all_imgs = get_image_paths(folder_path)
    print(f"{folder_name}: Found {len(all_imgs)} images (assuming 50K total).")
    
    if len(all_imgs) < train_size_per_cat + val_size_per_cat:
        print(f"Warning: {folder_name} has fewer images than required!")
        continue
    
    # 随机 shuffle
    random.shuffle(all_imgs)
    
    # 抽取 train
    train_imgs = all_imgs[:train_size_per_cat]
    for img_path in train_imgs:
        train_lines.append(f"{img_path} {class_id}")
    
    # 剩余用于 val
    remaining = all_imgs[train_size_per_cat:]
    val_imgs = remaining[:val_size_per_cat]
    for img_path in val_imgs:
        val_lines.append(f"{img_path} {class_id}")

# 保存 txt（只生成 train 和 val）
with open("train.txt", "w") as f:
    f.write("\n".join(train_lines))
with open("val.txt", "w") as f:
    f.write("\n".join(val_lines))

print(f"Generated: train.txt ({len(train_lines)} lines), val.txt ({len(val_lines)} lines)")