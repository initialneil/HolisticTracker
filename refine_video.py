import os
import os.path as osp
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tyro
from omegaconf import OmegaConf
from src.utils.rprint import rlog as log
from src.configs.argument_config import ArgumentConfig
from src.data_prepare_pipeline import DataPreparePipeline
from src.configs.data_prepare_config import DataPreparationConfig
import accelerate
import torch


class Dataset:
    def __init__(self, data_dirs):
        self.data_dirs = data_dirs
    def __len__(self):
        return len(self.data_dirs)
    def __getitem__(self, idx):
        return self.data_dirs[idx]


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    acc = accelerate.Accelerator()
    os.makedirs(args.output_dir, exist_ok=True)
    
    log(f"Processing single video: {args.in_root}")
    data_dirs = [osp.join(args.in_root, d) for d in os.listdir(args.in_root) if osp.exists(
        osp.join(args.in_root, d, "videos_info.json"))]

    data_dirs = [d for d in data_dirs if not osp.exists(
        osp.join(args.output_dir, osp.basename(d), "optim_tracking_ehm.pkl"))]
    
    data_dirs.sort()

    if args.reversed_order:
        data_dirs = data_dirs[::-1]
    
    # Load optimization config
    if not os.path.exists(args.optim_cfg):
        log(f"Error: Optimization config not found: {args.optim_cfg}")
        return
    
    optim_cfg = OmegaConf.load(args.optim_cfg)
    log(f"Loaded optimization config from: {args.optim_cfg}")
    
    dataset = Dataset(data_dirs)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=1, persistent_workers=True,
        collate_fn=lambda x: x,
    )
    dataloader = acc.prepare(dataloader)

    # Initialize pipeline
    pdata_cfg = partial_fields(DataPreparationConfig, args.__dict__)
    pdata_cfg.device = str(acc.device)
    print(f"Thread {acc.process_index} using device: {pdata_cfg.device}")
    data_prepare_pipeline = DataPreparePipeline(data_prepare_cfg=pdata_cfg)
    data_prepare_pipeline = acc.prepare(data_prepare_pipeline)
    
    # Iterate over videos
    for dirs in dataloader:
        # Set source to single video
        args.source_dir = dirs
        
        # Run refinement with optimization config
        data_prepare_pipeline.refine(args, optim_cfg)


if __name__ == '__main__':
    main()
