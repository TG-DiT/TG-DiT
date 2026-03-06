import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

# Project Imports
from models.model import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

#################################################################################
#                        Inference Dataset (Demo Version)                 #
#################################################################################

class InferenceDataset(Dataset):
    """
    Reads images and parses turbulence params from filename for conditional inference.
    Format: {Name}_{ClassID}_{Strength}.png
    Example: 00648_0_4.799.png -> Class: 0, Strength: 4.799
    """
    def __init__(self, root_dir, image_size=256, default_class=0, default_s=0.5):
        self.root_dir = root_dir
        self.image_size = image_size
        self.default_class = default_class
        self.default_s = default_s
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
        self.image_paths = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.lower().endswith(valid_exts)
        ])
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def parse_filename(self, filename):
        """
        Extracts ClassID and Strength from filename.
        """
        try:
            name_no_ext = os.path.splitext(filename)[0]
            parts = name_no_ext.split('_')
            
            # Expecting at least: Name_Class_Strength
            if len(parts) >= 3:
                s_val = float(parts[-1])
                c_val = int(parts[-2])
                return c_val, s_val
        except Exception:
            pass
        
        # Fallback defaults
        return self.default_class, self.default_s

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.basename(path)
        
        # 1. Load Input Image
        try:
            img = Image.open(path).convert("RGB")
            input_tensor = self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            input_tensor = torch.zeros(3, self.image_size, self.image_size)

        # 2. Parse Conditions
        class_id, s_val = self.parse_filename(filename)
        
        return input_tensor, class_id, s_val, filename

#################################################################################
#                             Core Inference Logic                              #
#################################################################################

def load_checkpoint(model, ckpt_path, device):
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"Loading checkpoint from: {ckpt_path}")
    elif not dist.is_initialized():
        print(f"Loading checkpoint from: {ckpt_path}")

    # Load with map_location to CPU to save GPU memory during load
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
    
    # Auto-detect key structure (EMA vs Model vs Raw)
    state_dict = checkpoint.get("ema", checkpoint.get("model", checkpoint))

    # Remove DDP 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    return model

@torch.no_grad()
def sample_batch(model, diffusion, vae, high_turb_lat, y_cond, args, device, fixed_noise=None):
    """
    Performs reverse diffusion sampling.
    """
    n = high_turb_lat.shape[0]
    
    # 1. Prepare Conditions for CFG
    if args.class_free_guide:
        # Null condition for Unconditional Branch
        y_null = torch.tensor([[args.num_classes, 0.5]], device=device).expand(n, 2)
        y_combined = torch.cat([y_cond, y_null], dim=0)
        high_turb_lat_combined = high_turb_lat.repeat(2, 1, 1, 1)
        sample_fn = model.module.forward_with_cfg if hasattr(model, 'module') else model.forward_with_cfg
    else:
        y_combined = y_cond
        high_turb_lat_combined = high_turb_lat
        sample_fn = model.module.forward if hasattr(model, 'module') else model.forward

    # 2. Noise Injection (Fixed noise ensures fair comparison between Cond/Uncond)
    if fixed_noise is not None:
        z = fixed_noise
    else:
        z = torch.randn(n, 4, high_turb_lat.shape[2], high_turb_lat.shape[2], device=device)

    # Duplicate noise for CFG (2N batch)
    z_in = torch.cat([z, z], dim=0) if args.class_free_guide else z
    
    # 3. Diffusion Sampling Loop
    model_kwargs = dict(
        y=y_combined, 
        high_turb=high_turb_lat_combined, 
        cfg_scale=args.cfg_scale
    )
    
    samples = diffusion.p_sample_loop(
        sample_fn,
        z_in.shape, 
        z_in, 
        clip_denoised=False, 
        model_kwargs=model_kwargs,
        progress=False, 
        device=device
    )
    
    # Split results (Keep Conditional output part)
    if args.class_free_guide:
        samples, _ = samples.chunk(2, dim=0)
        
    # 4. VAE Decode
    decoded = vae.decode(samples / 0.18215).sample
    return decoded

#################################################################################
#                                   Main Loop                                   #
#################################################################################

def main(args):
    # Setup Distributed or Single GPU
    if args.distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {device}")

    # Set Seed
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"TG-DiT Inference | Model: {args.model} | Input: {args.input_dir}")

    # Load VAE & DiT
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    model = DiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes
    ).to(device)
    model = load_checkpoint(model, args.ckpt, device)
    model.eval()
    
    if args.distributed:
        model = DDP(model, device_ids=[device_id])

    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Data Loader
    dataset = InferenceDataset(
        root_dir=args.input_dir, 
        image_size=args.image_size,
        default_class=args.default_class,
        default_s=args.default_s
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if args.distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)

    if rank == 0:
        print(f"Found {len(dataset)} images. Starting inference...")

    with torch.no_grad():
        for batch_idx, (turb_imgs, class_ids, s_vals, filenames) in enumerate(tqdm(loader, disable=(rank != 0))):
            turb_imgs = turb_imgs.to(device)
            class_ids = class_ids.to(device)
            s_vals = s_vals.to(device)
            
            # Encode Input to Latents
            high_turb_lat = vae.encode(turb_imgs).latent_dist.sample().mul_(0.18215)
            
            # Generate Shared Noise (Crucial for fair Cond vs Uncond comparison)
            current_bs = turb_imgs.shape[0]
            latent_dim = high_turb_lat.shape[2]
            shared_noise = torch.randn(current_bs, 4, latent_dim, latent_dim, device=device)

            # --- Pass 1: Conditional Restoration (Ours) ---
            # Uses parsed ClassID and Strength from filename
            y_cond = torch.stack([class_ids.float(), s_vals.float()], dim=1)
            
            restored_cond = sample_batch(
                model, diffusion, vae, 
                high_turb_lat, y_cond, 
                args, device, fixed_noise=shared_noise
            )

            # --- Pass 2: Unconditional Restoration (Baseline) ---
            # Uses Null Class and Null Strength (0.5 is placeholder, mapped to null embedding internally)
            null_cls = torch.full_like(class_ids, args.num_classes).float()
            null_s = torch.full_like(s_vals, 0.5).float()
            y_uncond = torch.stack([null_cls, null_s], dim=1)
            
            restored_uncond = sample_batch(
                model, diffusion, vae, 
                high_turb_lat, y_uncond, 
                args, device, fixed_noise=shared_noise
            )

            # --- Visualization ---
            # Normalize [-1, 1] -> [0, 1]
            input_vis = torch.clamp((turb_imgs + 1.0) / 2.0, 0.0, 1.0)
            cond_vis = torch.clamp((restored_cond + 1.0) / 2.0, 0.0, 1.0)
            uncond_vis = torch.clamp((restored_uncond + 1.0) / 2.0, 0.0, 1.0)

            for i in range(current_bs):
                # Concatenate: [Input] | [Conditional] | [Unconditional]
                combined = torch.cat([input_vis[i], cond_vis[i], uncond_vis[i]], dim=2)
                
                filename = filenames[i]
                save_path = os.path.join(args.output_dir, filename)
                save_image(combined, save_path)

    if args.distributed:
        cleanup()
    
    if rank == 0:
        print(f"Done. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Required Paths
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder containing turbulence images")
    parser.add_argument("--output-dir", type=str, default="./results_demo", help="Output folder")
    
    # Model Config
    parser.add_argument("--model", type=str, default="DiT-XL/2", choices=DiT_models.keys())
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=4, help="Total classes (including null token index)")
    
    # Fallback Params (if parsing fails)
    parser.add_argument("--default-class", type=int, default=0)
    parser.add_argument("--default-s", type=float, default=0.5)

    # Inference Settings
    parser.add_argument("--cfg-scale", type=float, default=1.0) # 1.0 = standard conditional, >1.0 = stronger guidance
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-free-guide", action="store_true", default=True)
    
    # System
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--distributed", action="store_true", help="Enable Multi-GPU DDP")

    args = parser.parse_args()
    main(args)