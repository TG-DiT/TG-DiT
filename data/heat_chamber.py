import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class HeatChamberDataset(Dataset):
    """
    适用于 Heat Chamber 数据集的 PyTorch Dataset 类。
    V3: 支持均匀采样文件夹子集。
    """
    def __init__(self, root_dir, image_size=256, num_folders_to_use=None):
        """
        初始化数据集。

        Args:
            root_dir (str): 数据集的根目录路径。
            image_size (int): 图像预处理后的目标尺寸。
            num_folders_to_use (int, optional): 要加载的文件夹数量。
                                                 如果为 None，则加载所有文件夹。默认为 None。
        """
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.num_folders_to_use = num_folders_to_use
        self.image_pairs = self._create_image_pairs()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _create_image_pairs(self):
        """
        扫描数据集目录，创建图像路径对，并提取类别信息。
        支持从所有文件夹中均匀采样一个子集。
        """
        pairs = []
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"指定的根目录不存在: {self.root_dir}")
            
        all_scene_folders = sorted(os.listdir(self.root_dir))
        
        # [修改点] 修改文件夹筛选逻辑，实现均匀采样
        if self.num_folders_to_use is not None and 0 < self.num_folders_to_use < len(all_scene_folders):
            total_folders = len(all_scene_folders)
            # 计算采样间隔
            step = total_folders / self.num_folders_to_use
            # 根据间隔生成要选取的文件夹的索引
            indices = [int(i * step) for i in range(self.num_folders_to_use)]
            # 根据索引选取文件夹
            folders_to_process = [all_scene_folders[i] for i in indices]
            
            print(f"在 {total_folders} 个总文件夹中，已均匀采样 {len(folders_to_process)} 个进行加载。")
        else:
            # 如果不指定数量，或数量超出范围，则加载全部
            folders_to_process = all_scene_folders
            print(f"将加载全部 {len(folders_to_process)} 个文件夹。")


        for scene_folder in folders_to_process:
            scene_path = os.path.join(self.root_dir, scene_folder)
            if not os.path.isdir(scene_path):
                continue

            class_id = None
            if scene_folder.endswith('_1'):
                class_id = 1.0
            elif scene_folder.endswith('_2'):
                class_id = 2.0
            else:
                print(f"警告: 文件夹 '{scene_folder}' 名称不符合 '_1' 或 '_2' 的后缀规范，已跳过。")
                continue
            
            y_info = [class_id, 0.5]

            gt_path = os.path.join(scene_path, 'gt.png')
            turb_folder_path = os.path.join(scene_path, 'turb')

            if os.path.exists(gt_path) and os.path.isdir(turb_folder_path):
                turb_images = sorted(glob.glob(os.path.join(turb_folder_path, '*.png')))
                for turb_path in turb_images:
                    pairs.append({'turb': turb_path, 'gt': gt_path, 'y': y_info})
        
        if not pairs:
            print(f"警告: 在目录 {self.root_dir} 中没有找到符合结构的图像数据。")
            
        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        pair_info = self.image_pairs[index]
        gt_path = pair_info['gt']
        turb_path = pair_info['turb']
        y_info = pair_info['y']

        try:
            gt_image = Image.open(gt_path).convert('RGB')
            turb_image = Image.open(turb_path).convert('RGB')
        except Exception as e:
            print(f"错误: 加载图像失败 at index {index}, path: {gt_path} or {turb_path}")
            print(f"Error message: {e}")
            raise

        gt_tensor = self.transform(gt_image)
        turb_tensor = self.transform(turb_image)
        y_tensor = torch.tensor(y_info, dtype=torch.float32)

        return {
            "clean": gt_tensor,
            "high_turb": turb_tensor,
            "y": y_tensor,
            "path":turb_path
        }