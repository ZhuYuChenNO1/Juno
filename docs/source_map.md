# Juno source map

| Paper concept | Implementation |
| --- | --- |
| JEPA predictive target and alignment | `juno/model/modules/world_model/JEPA.py` |
| Juno policy and JEPA visual interface | `juno/model/framework/VLM4A/juno_policy.py` |
| Mixture-of-Transformers action branch | `juno/model/framework/VLM4A/juno_policy.py` |
| Direct base class (`Qwen_GR00T`) | `juno/model/framework/VLM4A/QwenGR00T.py` |
| Bridge/Fractal data mixture | `juno/dataloader/gr00t_lerobot/mixtures.py` |
| Training loop | `juno/training/train_juno.py` |
| Released recipe | `configs/bridge_fractal_jepa7_mot_1005.yaml` |
| Launch contract | `scripts/run_bridge_fractal_jepa7_mot_train.sh` |
