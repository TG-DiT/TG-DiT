import torch
import pyiqa

class ImageQualityEvaluator:
    """
    A standalone utility class for image quality assessment (IQA), supporting batch evaluation on GPU.
    Implements standard metrics including PSNR, SSIM, LPIPS, NIQE, GMSD, FSIM, and VSI using the pyiqa library.

    Example usage:
        evaluator = ImageQualityEvaluator(device='cuda')
        results = evaluator.compute(preds, targets)
        print(results)  # Output dict, e.g., {'psnr': tensor(30.5), ...}
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the evaluator and create metric instances.
        
        Args:
            device (str): Device to run metrics on ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        self.metrics = {
            'psnr': pyiqa.create_metric('psnr', device=self.device),
            'ssim': pyiqa.create_metric('ssim', device=self.device),
            'lpips': pyiqa.create_metric('lpips', device=self.device),
            'niqe': pyiqa.create_metric('niqe', device=self.device),
            'gmsd': pyiqa.create_metric('gmsd', device=self.device),
            'fsim': pyiqa.create_metric('fsim', device=self.device),
            'vsi': pyiqa.create_metric('vsi', device=self.device),
        }
    
    def compute(self, preds: torch.Tensor, targets: torch.Tensor = None) -> dict:
        """
        Compute metrics for a batch of images and return aggregated results (mean).

        Args:
            preds (torch.Tensor): Predicted images [N, 3, H, W], RGB, range [0, 1].
            targets (torch.Tensor): Reference images [N, 3, H, W], RGB, range [0, 1]. 
                                    Optional for no-reference metrics like NIQE.

        Returns:
            dict: A dictionary mapping metric names to their scalar scores (averaged over batch).
        """
        if not isinstance(preds, torch.Tensor):
            raise ValueError("preds must be a torch.Tensor with shape [N, 3, H, W]")
        if targets is not None and not isinstance(targets, torch.Tensor):
            raise ValueError("targets must be a torch.Tensor with shape [N, 3, H, W]")
        
        preds = preds.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)
        
        results = {}
        for name, metric in self.metrics.items():
            if name == 'niqe':
                # NIQE is a no-reference metric
                score = metric(preds)
            else:
                if targets is None:
                    raise ValueError(f"{name} requires targets")
                score = metric(preds, targets)
            
            # Aggregation: if score is a batch tensor, calculate the mean
            if score.ndim > 0:
                score = torch.mean(score)
            results[name] = score
            
        return results