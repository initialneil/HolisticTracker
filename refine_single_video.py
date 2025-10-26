import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tyro
from omegaconf import OmegaConf
from src.utils.rprint import rlog as log
from src.configs.argument_config import ArgumentConfig
from src.data_prepare_pipeline import DataPreparePipeline
from src.configs.data_prepare_config import DataPreparationConfig

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    
    log(f"Processing single video: {args.in_root}")
    
    # Load optimization config
    if not os.path.exists(args.optim_cfg):
        log(f"Error: Optimization config not found: {args.optim_cfg}")
        return
    
    optim_cfg = OmegaConf.load(args.optim_cfg)
    log(f"Loaded optimization config from: {args.optim_cfg}")
    
    # Initialize pipeline
    pdata_cfg = partial_fields(DataPreparationConfig, args.__dict__)
    data_prepare_pipeline = DataPreparePipeline(data_prepare_cfg=pdata_cfg)
    
    # Set source to single video
    args.source_dir = [args.in_root]
    
    # Run refinement with optimization config
    data_prepare_pipeline.refine(args, optim_cfg)


if __name__ == '__main__':
    main()
