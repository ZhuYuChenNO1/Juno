# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

import copy
from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from juno.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from juno.dataloader.gr00t_lerobot.registry import (
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    DATASET_NAMED_MIXTURES,
    EmbodimentTag,
)

def collate_fn(batch):
    return batch


def _cfg_get(data_cfg, key, default=None):
    if data_cfg is None:
        return default
    if hasattr(data_cfg, "get"):
        return data_cfg.get(key, default)
    return getattr(data_cfg, key, default)


def _cfg_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() not in ("false", "0", "no", "off")
    return bool(value)


def _apply_jepa_modality_overrides(modality_config: dict, data_cfg: dict | None) -> None:
    """Patch JEPA temporal deltas after dynamic registry configs are built."""
    if data_cfg is None:
        return

    future_horizon = _cfg_get(data_cfg, "future_video_horizon", None)
    history_frames = int(_cfg_get(data_cfg, "jepa_history_frames", 1) or 1)
    return_history_actions = _cfg_bool(_cfg_get(data_cfg, "jepa_return_history_actions", False))
    if future_horizon is None and history_frames <= 1 and not return_history_actions:
        return

    future_horizon = int(future_horizon or 0)
    frame_stride = int(_cfg_get(data_cfg, "frame_stride", 1) or 1)
    history_stride = int(_cfg_get(data_cfg, "jepa_history_stride", frame_stride) or frame_stride)
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if history_stride < 1:
        raise ValueError(f"jepa_history_stride must be >= 1, got {history_stride}")

    if "video" in modality_config and (future_horizon > 0 or history_frames > 1):
        history = [-(history_stride * i) for i in range(history_frames - 1, 0, -1)]
        future = [frame_stride * (i + 1) for i in range(future_horizon)]
        modality_config["video"].delta_indices = history + [0] + future

    if "action" in modality_config and return_history_actions:
        base = [int(x) for x in modality_config["action"].delta_indices if int(x) >= 0]
        history = list(range(-history_stride * (history_frames - 1), 0)) if history_frames > 1 else []
        modality_config["action"].delta_indices = history + base


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = copy.deepcopy(ROBOT_TYPE_CONFIG_MAP[robot_type])
    if data_cfg is not None:
        # Dynamic registry configs may not declare these attributes on the class.
        # Set them unconditionally before modality_config() builds delta indices.
        for key in ("future_video_horizon", "frame_stride", "jepa_history_frames", "jepa_history_stride"):
            value = data_cfg.get(key, None)
            if value is not None:
                setattr(data_config, key, int(value))
        value = data_cfg.get("jepa_return_history_actions", None)
        if value is not None:
            if isinstance(value, str):
                value = value.lower() not in ("false", "0", "no", "off")
            setattr(data_config, "jepa_return_history_actions", bool(value))
    modality_config = data_config.modality_config()
    _apply_jepa_modality_overrides(modality_config, data_cfg)
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="examples/LIBERO/train_files/juno_cotrain_libero.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        if count > 100:
            break
        count += 1
        pass
