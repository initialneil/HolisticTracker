import os
import os.path as osp
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tyro
import time
import shlex
import subprocess
from omegaconf import OmegaConf
from src.utils.rprint import rlog as log
from src.configs.argument_config import ArgumentConfig
from src.data_prepare_pipeline import DataPreparePipeline
from src.configs.data_prepare_config import DataPreparationConfig


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def build_command(args, part_lst, gpu_id):
    cmd_parts = [
        f'CUDA_VISIBLE_DEVICES={gpu_id}',
        'python', os.path.basename(os.path.abspath(__file__))
    ]
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name not in ["part_lst", "visible_gpus", "output_dir", "n_divide", "optim_cfg"]:
            if isinstance(arg_value, bool):
                if arg_value:
                    cmd_parts.append(f'--{arg_name}')
            elif isinstance(arg_value, list):
                cmd_parts.append(f'--{arg_name} {" ".join(map(str, arg_value))}')
            else:
                cmd_parts.append(f'--{arg_name} {shlex.quote(str(arg_value))}')
    
    cmd_parts.extend([
        '-p', part_lst,
        '-n', '1',
        '-v', '0',
        '--output_dir', args.output_dir,
        '--optim_cfg', args.optim_cfg
    ])
    
    return ' '.join(cmd_parts)

def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    os.makedirs(args.output_dir, exist_ok=True)
    
    log(f"Processing videos from: {args.in_root}")
    all_dirs = [osp.join(args.in_root, d) for d in os.listdir(args.in_root) if osp.exists(
        osp.join(args.in_root, d, "videos_info.json"))]

    all_dirs = [d for d in all_dirs if not osp.exists(
        osp.join(args.output_dir, osp.basename(d), "optim_tracking_ehm.pkl"))]
    
    all_dirs.sort()

    if args.reversed_order:
        all_dirs = all_dirs[::-1]
    
    # Load optimization config
    if not os.path.exists(args.optim_cfg):
        log(f"Error: Optimization config not found: {args.optim_cfg}")
        return
    
    optim_cfg = OmegaConf.load(args.optim_cfg)
    log(f"Loaded optimization config from: {args.optim_cfg}")
    
    visible_gpus = [int(x) for x in args.visible_gpus.split(',') if x.strip() != '']
    
    if args.part_lst is None or args.part_lst.lower() == 'nan':
        # Parallel mode: spawn subprocesses for each GPU
        per_num = len(all_dirs) // int(args.n_divide) + 1
        all_procs = []
        counter = 0
        for iii, i in enumerate(range(0, len(all_dirs), per_num)):
            part_lst = f'{i},{i + per_num}'
            gpu_id = iii % len(visible_gpus)
            
            cmd = build_command(args, part_lst, visible_gpus[gpu_id])
            log(cmd)
            all_procs.append(subprocess.Popen(cmd, shell=True))
            counter += 1
            if counter % len(visible_gpus) == 0:
                # Sleep 30 seconds after every visible_gpus processes to avoid memory overload during warm-up
                print("start sleeping for 30 seconds.......")
                time.sleep(30)
                print("finish sleeping.......")
        for p in all_procs:
            p.wait()
    else:
        # Single process mode: process assigned range of videos
        pdata_cfg = partial_fields(DataPreparationConfig, args.__dict__)
        
        data_prepare_pipeline = DataPreparePipeline(data_prepare_cfg=pdata_cfg)
        
        part_lst = [int(x) for x in args.part_lst.split(',')]
        for i in range(len(part_lst)):
            if part_lst[i] == 0:
                part_lst[i] = None
        all_dirs = all_dirs[part_lst[0]: part_lst[1]]
        
        # Set source_dir as list of directories to process
        args.source_dir = all_dirs
        
        # Run refinement with optimization config
        data_prepare_pipeline.refine(args, optim_cfg)


if __name__ == '__main__':
    main()
