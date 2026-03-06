import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2
import random
from turb_sim.simulator import Simulator
import torch
import glob  # 新增：用于模糊查找


class LSUNBase(Dataset):
    def __init__(
        self,
        txt_file,
        data_root,  # 保留但未使用（兼容原有）
        size=None,
        interpolation="bicubic",
        flip_p=0.5,
        turbulence=False,  # 新增参数：是否启用湍流模式
        turb_root=None,    # 新增：预生成 high_turb 根目录（可选）
        weak_turb_root =None,
        device=None,       # 新增：设备指定，必选 for turbulence=True
    ):
        self.data_root = data_root  # 保留但未使用

        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            lines = f.read().splitlines()
        # 修改：解析路径和 class_id
        self.image_paths = [line.split()[0] for line in lines if line.strip()]  # 完整路径（split()[0]）
        self.class_ids = [int(line.split()[-1]) for line in lines if line.strip()]  # 类别 ID（整数 1/2/3）
        self._length = len(self.image_paths)
        self.labels = {
            "file_path_": self.image_paths,  # 完整路径列表
            "class_id": self.class_ids,      # 新增：类别 ID 列表
        }

        self.size = size
        self.interpolation = {
            "linear": Image.Resampling.BILINEAR,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }[interpolation]
        self.flip_p = flip_p

        self.turbulence = turbulence
        if self.turbulence:
            # assert turb_root is not None, "In turbulence mode, turb_root must be provided."
            # self.turb_root = turb_root
            if device is None:
                raise ValueError("In turbulence mode, device must be provided (e.g., torch.device('cuda:0')).")
            self.device = device
            
            # --- [核心改动 1]: 初始化时预处理湍流参数 ---

            # Dataset也直接加载 s_values.npy
            s_values_path = "/home/nvidia/MachineLearning/code/TurbulenceSim_P2S-master/data/s_values.npy" # 修改为您的路径
            if not os.path.exists(s_values_path):
                raise ValueError(f"S_values file not found at {s_values_path}.")
            self.s_values = np.load(s_values_path)
            
            # turb_params.npy 仍然需要加载，因为它包含了 r0 等信息，但我们不再用它来反算S
            turb_params_path = "/home/nvidia/MachineLearning/code/TurbulenceSim_P2S-master/data/turb_params.npy"
            if not os.path.exists(turb_params_path):
                raise ValueError(f"Turb params file not found at {turb_params_path}.")
            self.turb_params = np.load(turb_params_path)

            # 确保二者长度一致
            assert len(self.s_values) == len(self.turb_params), "s_values and turb_params must have the same length."

            # 我们不再需要对s_values或turb_params进行排序，因为s_values来自linspace，
            # 并且我们假设turb_params是按照s_values的顺序生成的。
            # --- [修复结束] ---

            # 对S值进行排序，并同步排序params数组，这是高效查找的基础
            sort_indices = np.argsort(self.s_values)
            self.s_values = self.s_values[sort_indices]
            self.turb_params = self.turb_params[sort_indices]


            # # 新增：过滤只保留 [0.5, 3.0] 区间内的湍流强度
            # mask = (self.s_values >= 0.5) & (self.s_values <= 5.0)
            # self.s_values = self.s_values[mask]
            # self.turb_params = self.turb_params[mask]

            # 新增：统计并打印过滤后的参数数量
            num_filtered = len(self.s_values)
            print(f"Filtered parameters in [0.5, 5.0]: {num_filtered}")
            assert num_filtered > 0, "No parameters left after filtering [2.5, 3.0]. Check s_values range."

            # 定义弱湍流采样策略的参数
            self.s_min = 0.5
            self.alpha_min = 0.5
            self.alpha_max = 0.8
            self.threshold_T = self.s_min / self.alpha_min # 决策边界 T = 1.0
            # --- [核心改动 1 结束] ---
            self.param_order = np.arange(len(self.turb_params)) # 使用排序后参数的长度
            self.turb_root = turb_root
            self.weak_turb_root = weak_turb_root 
            self.use_simulator = (turb_root is None)

            if self.use_simulator:
                # 初始化 simulator（默认参数，之后动态更新）
                self.simulator = Simulator(img_size=self.size,device=self.device).to(self.device, dtype=torch.float32)
                print(f"Simulator initialized on {self.device} for turbulence mode.")  # 调试日志
            else:
                if not os.path.exists(self.turb_root):
                    raise ValueError(f"Turb root path {self.turb_root} not found.")
                print(f"Using precomputed high_turb from {self.turb_root}.")


    # --- [核心改动 2]: 新增一个辅助方法用于高效查找最接近的S值索引 ---
    def _find_closest_s_index(self, s_target):
        """
        使用二分查找在 self.s_values 中找到最接近 s_target 的值的索引。
        """
        # np.searchsorted 找到插入点以保持数组有序
        idx = np.searchsorted(self.s_values, s_target, side="left")

        # 处理边界情况
        if idx == 0:
            return 0
        if idx == len(self.s_values):
            return len(self.s_values) - 1

        # 比较左边和右边哪个更近
        left_val = self.s_values[idx - 1]
        right_val = self.s_values[idx]
        
        if (s_target - left_val) < (right_val - s_target):
            return idx - 1
        else:
            return idx
    # --- [核心改动 2 结束] ---

    # 新增：方法用于每个 epoch shuffle 参数顺序（基于种子确保确定性和跨 rank 一致）
    def shuffle_params(self, seed):
        if not self.turbulence or not self.use_simulator:  # 修改：只在用 simulator 时 shuffle
            return  # 无需 shuffle
        np.random.seed(seed)  # 确定性 shuffle
        np.random.shuffle(self.param_order)
        #if hasattr(self, 'rank') and self.rank == 0:  # 假设 rank 在外部设置；仅 rank 0 打印调试
        print(f"Shuffled param_order for seed {seed}, first 5: {self.param_order[:5]}")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        full_path = self.labels["file_path_"][i]  # 修改：完整路径
        class_id = self.labels["class_id"][i]     # 新增：提取类别 ID
        try:
            if self.turbulence:
                # 湍流模式：加载clean、high_turb，生成low_turb
                # 修改：直接使用 full_path 作为 clean_path（假设 txt 中的路径是 clean）
                clean_path = full_path  # 无需 rsplit，因为路径是完整 clean 路径
                
                # 加载clean图像
                clean_image = Image.open(clean_path)
                if not clean_image.mode == "RGB":
                    clean_image = clean_image.convert("RGB")
                
                # 新增：检查尺寸，只有非 256x256 时 resize
                if self.size is not None and clean_image.size != (self.size, self.size):
                    clean_image = clean_image.resize((self.size, self.size), self.interpolation)
                
                clean_img = np.array(clean_image).astype(np.uint8)
                
                # 检查图像尺寸必须是256x256（resize 后应已满足）
                if clean_img.shape[:2] != (self.size, self.size):
                    raise ValueError(f"Clean image at {clean_path} is not 256x256 after resize, got {clean_img.shape[:2]}")
                

                #废弃，使用模拟器生成low turb
                # # 生成low_turb：下采样到64x64，再上采样到256x256（使用cv2 INTER_LINEAR）
                # low_turb_img = cv2.resize(clean_img, (64, 64), interpolation=cv2.INTER_LINEAR)
                # weak_turb_img = cv2.resize(low_turb_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                
                S = 0.0
                # 修改：根据 shuffle 后的索引 i 获取参数，计算 S = D / r0（只在 use_simulator 时用）
                # --- [核心改动 3]: 实现弱湍流采样和生成 ---
                if self.use_simulator:
                    # 1. 获取强湍流参数
                    strong_param_idx = self.param_order[i % len(self.param_order)] # 使用取模防止索引越界
                    strong_params = self.turb_params[strong_param_idx]
                    s_strong = self.s_values[strong_param_idx]
                    S = s_strong
                    # 2. 应用鲁棒混合采样策略计算 s_weak_target
                    if s_strong >= self.threshold_T: # 安全区域
                        alpha = random.uniform(self.alpha_min, self.alpha_max)
                        s_weak_target = s_strong * alpha
                    else: # 约束区域
                        # 确保 s_weak_target < s_strong
                        s_weak_target = random.uniform(self.s_min, s_strong)

                    # 3. 查找与 s_weak_target 最匹配的参数索引
                    weak_param_idx = self._find_closest_s_index(s_weak_target)
                    weak_params = self.turb_params[weak_param_idx]
                    s_weak = self.s_values[weak_param_idx] # 这是实际使用的弱湍流强度

                    # 4. 使用模拟器生成强/弱湍流图像
                    clean_tensor = torch.from_numpy(clean_img.transpose(2, 0, 1)).to(self.device, dtype=torch.float32) / 255.0
                    with torch.no_grad():
                        # 生成强湍流
                        self.simulator.update_dr0(s_strong)
                        strong_turb_tensor = self.simulator(clean_tensor)
                        strong_turb_img = strong_turb_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255.0
                        strong_turb_img = np.clip(strong_turb_img, 0, 255).astype(np.uint8)

                        # 生成弱湍流
                        self.simulator.update_dr0(s_weak)
                        weak_turb_tensor = self.simulator(clean_tensor)
                        weak_turb_img = weak_turb_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255.0
                        weak_turb_img = np.clip(weak_turb_img, 0, 255).astype(np.uint8)
                else:
                # 新增：从 turb_root 加载预生成 high_turb（使用模糊查找匹配前缀）
                    base_name = os.path.basename(full_path).split('.')[0]  # 去扩展名 (e.g., "church_0042432")
                    # 模糊模式：搜索 base_name_类别_*.png
                    pattern = os.path.join(self.turb_root, f"{base_name}_{class_id}_*.png")
                    matches = glob.glob(pattern)
                    if len(matches) == 0:
                        raise ValueError(f"No matching high_turb file found for {base_name} (class {class_id}) in {self.turb_root}. Pattern: {pattern}")
                    elif len(matches) > 1:
                        print(f"Warning: Multiple matches for {base_name} (class {class_id}): {matches}. Using first: {matches[0]}")
                    high_turb_path = matches[0]  # 取第一个匹配

                    # 新增：从文件名提取 S（假设文件名如 base_name_class_id_S.png，S 在最后一个 _ 后，去扩展名）
                    strong_turb_base_name = os.path.basename(high_turb_path).rsplit('.', 1)[0]  # 用 rsplit('.', 1)[0] 处理多 '.' 文件名
                    parts = strong_turb_base_name.split('_')
                    S = float(parts[-1])
      
                    try:
                        strong_turb_img = Image.open(high_turb_path)
                        if not strong_turb_img.mode == "RGB":
                            strong_turb_img = strong_turb_img.convert("RGB")
                        
                        # 检查尺寸，只有非 256x256 时 resize（兼容）
                        if self.size is not None and strong_turb_img.size != (self.size, self.size):
                            strong_turb_img = strong_turb_img.resize((self.size, self.size), self.interpolation)
                        
                        strong_turb_img = np.array(strong_turb_img).astype(np.uint8)

                        
                        # 检查图像尺寸必须是256x256（resize 后应已满足）
                        if strong_turb_img.shape[:2] != (self.size, self.size):
                            raise ValueError(f"High turb image at {high_turb_path} is not 256x256 after resize, got {strong_turb_img.shape[:2]}")
                    except FileNotFoundError:
                        raise ValueError(f"High turb image not found at {high_turb_path}.")
                    except Exception as e:
                        raise ValueError(f"Error loading high turb image at {high_turb_path}: {e}")
                    
                    if self.weak_turb_root:
                        # 使用同样的基础名和类别ID，在弱湍流文件夹中进行查找
                        weak_pattern = os.path.join(self.weak_turb_root, f"{base_name}_{class_id}_*.png")
                        weak_matches = glob.glob(weak_pattern)

                        if len(weak_matches) == 0:
                            # 如果找不到配对的弱湍流文件，这是一个严重的数据问题，直接报错
                            raise ValueError(f"CRITICAL: Found strong turb file but NO matching WEAK turb file for {base_name} (class {class_id}) in {self.weak_turb_root}. Pattern: {weak_pattern}")
                        elif len(weak_matches) > 1:
                            print(f"Warning: Multiple matches for WEAK turb file {base_name}: {weak_matches}. Using first.")
                        
                        weak_turb_path = weak_matches[0]

                        try:
                            weak_turb_pil = Image.open(weak_turb_path).convert("RGB")
                            if self.size is not None and weak_turb_pil.size != (self.size, self.size):
                                weak_turb_pil = weak_turb_pil.resize((self.size, self.size), self.interpolation)
                            weak_turb_img = np.array(weak_turb_pil).astype(np.uint8)
                        except Exception as e:
                            raise ValueError(f"Error loading WEAK turb image at {weak_turb_path}: {e}")
                
                # 对所有三张图像应用相同的预处理（调整大小（如果需要）、翻转、归一化） - 无中心裁剪
                # if self.size is not None and self.size != 256:
                #     raise ValueError("Image size must be 256 for this dataset configuration.")
                
                clean_img = Image.fromarray(clean_img)
                strong_turb_img = Image.fromarray(strong_turb_img)
                weak_turb_img = Image.fromarray(weak_turb_img)
                
                # 决定是否翻转，并对所有三张应用相同翻转
                do_flip = random.random() < self.flip_p
                if do_flip:
                    clean_img = clean_img.transpose(Image.FLIP_LEFT_RIGHT)
                    strong_turb_img = strong_turb_img.transpose(Image.FLIP_LEFT_RIGHT)
                    weak_turb_img = weak_turb_img.transpose(Image.FLIP_LEFT_RIGHT)
                
                # 转换为np，归一化到[-1,1]
                clean_img = np.array(clean_img).astype(np.uint8)
                clean_img = (clean_img / 127.5 - 1.0).astype(np.float32)
                
                strong_turb_img = np.array(strong_turb_img).astype(np.uint8)
                strong_turb_img = (strong_turb_img / 127.5 - 1.0).astype(np.float32)
                
                weak_turb_img = np.array(weak_turb_img).astype(np.uint8)
                weak_turb_img = (weak_turb_img / 127.5 - 1.0).astype(np.float32)


                # # # 先转换回 uint8 [0,255] 以保存
                # def normalize_to_uint8(img):
                #     return ((img + 1.0) / 2.0 * 255.0).astype(np.uint8)

                # clean_uint8 = normalize_to_uint8(clean_img)
                # low_uint8 = normalize_to_uint8(low_turb_img)
                # high_uint8 = normalize_to_uint8(high_turb_img)

                # # 水平拼接：clean | low | high
                # combined_img = np.hstack((clean_uint8, low_uint8, high_uint8))

                # # 保存（使用 PIL；路径可自定义，如基于 i）
                # save_path = f"debug/concatenated_sample_{i}.png"  # 假设在 __getitem__ 中有 i
                # Image.fromarray(combined_img).save(save_path)

                # 返回字典（新增 class_id）
                return {
                    "clean": clean_img,
                    "high_turb": strong_turb_img,
                    "low_turb": weak_turb_img,
                    "y": [class_id, S]  # 新增：返回类别 ID
                }
            else:
                # 非湍流模式：原逻辑，仅加载单张图像作为"image"
                image = Image.open(full_path)  # 修改：使用完整路径
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                
                # 新增：检查尺寸，只有非 256x256 时 resize
                if self.size is not None and image.size != (self.size, self.size):
                    image = image.resize((self.size, self.size), self.interpolation)
                
                img = np.array(image).astype(np.uint8)
                
                # 检查图像尺寸必须是256x256（resize 后应已满足）
                if img.shape[:2] != (256, 256):
                    raise ValueError(f"Image at {full_path} is not 256x256 after resize, got {img.shape[:2]}")
                
                if self.size is not None and self.size != 256:
                    raise ValueError("Image size must be 256 for this dataset configuration.")
                
                img = Image.fromarray(img)
                
                # 决定是否翻转
                do_flip = random.random() < self.flip_p
                if do_flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                img = np.array(img).astype(np.uint8)
                img = (img / 127.5 - 1.0).astype(np.float32)
                # 修改：example 基于 full_path 和 class_id
                example = {
                    "file_path_": full_path,
                    "image": img,
                    "y": class_id  # 新增：返回类别 ID
                }
                return example
        except EOFError as e:
            # 当捕获到 EOFError 时
            print(f"\n[WARNING!!!!!!!!!!!] Caught EOFError for item {i} ({self.labels['file_path_'][i]}). Skipping and loading next item. Error: {e}")
            # 递归地加载下一个样本，使用取模运算防止索引越界
            return self.__getitem__((i + 1) % self.__len__())
        except Exception as e:
            # 也可以捕获其他可能的加载错误
            print(f"\n[WARNING!!!!!!!!!!!] Caught an unexpected error for item {i} ({self.labels['file_path_'][i]}). Skipping and loading next item. Error: {e}")
            return self.__getitem__((i + 1) % self.__len__())   


class LSUNChurchesTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="data/lsun/churches", **kwargs)


class LSUNChurchesValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/churches",
                         flip_p=flip_p, **kwargs)


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
                         flip_p=flip_p, **kwargs)


class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
                         flip_p=flip_p, **kwargs)
        
# Modified class for training dataset with correct paths and turbulence mode
class LSUNAirplaneTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="./train.txt",  # 修改：使用 train.txt
            data_root=None,          # 修改：无需 data_root
            turbulence=True,         # 启用湍流模式
            **kwargs
        )


# Modified class for validation dataset with correct paths and turbulence mode
class LSUNAirplaneValidation(LSUNBase):
    def __init__(self, turb_root=None, flip_p=0., **kwargs):  # 新增：turb_root 参数
        super().__init__(
            txt_file="./val.txt",    # 修改：使用 val.txt
            data_root=None,          # 修改：无需 data_root
            turbulence=True,         # 启用湍流模式
            turb_root="/home/nvidia/MachineLearning/Data/val_new",     # 新增：传入 turb_root
            weak_turb_root="/home/nvidia/MachineLearning/Data/val_weak_pair_new", # <--- 新增：传入弱湍流路径
            flip_p=flip_p,
            **kwargs
        )