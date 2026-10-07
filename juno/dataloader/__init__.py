import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from juno.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from juno.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=int(cfg.datasets.vla_data.get("num_workers", 2)),
            # shuffle=True
        )        
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "ttt_simplerenv":
        from juno.dataloader.lerobot_datasets import collate_fn
        from juno.dataloader.ttt_simplerenv_dataset import TTTSimplerEnvDataset

        vla_dataset_cfg = cfg.datasets.vla_data
        ttt_dataset = TTTSimplerEnvDataset(
            eval_dir=vla_dataset_cfg.eval_dir,
            num_future_frames=int(vla_dataset_cfg.get("future_video_horizon", 8)),
            frame_stride=int(vla_dataset_cfg.get("frame_stride", 2)),
            history_frames=int(vla_dataset_cfg.get("jepa_history_frames", 3)),
            history_stride=int(vla_dataset_cfg.get("jepa_history_stride", 2)),
            action_horizon=int(vla_dataset_cfg.get("action_horizon", 16)),
            timestep_stride=int(vla_dataset_cfg.get("timestep_stride", 1)),
            success_only=bool(vla_dataset_cfg.get("success_only", False)),
            action_only=bool(vla_dataset_cfg.get("ttt_action_only", False)),
            require_full_horizon=bool(vla_dataset_cfg.get("require_full_horizon", False)),
            action_stats_path=vla_dataset_cfg.get("action_stats_path", None),
            action_stats_key=str(vla_dataset_cfg.get("action_stats_key", "oxe_bridge")),
            recorded_action_scale=float(vla_dataset_cfg.get("recorded_action_scale", 1.0)),
        )
        if dist.get_rank() == 0:
            save_dataset_statistics(
                {ttt_dataset.action_stats_key: {"action": dict(ttt_dataset.action_stats)}},
                Path(cfg.output_dir),
            )
        return DataLoader(
            ttt_dataset,
            batch_size=vla_dataset_cfg.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=int(vla_dataset_cfg.get("num_workers", 2)),
            shuffle=True,
        )
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
