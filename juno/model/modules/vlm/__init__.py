def get_vlm_model(config):
    """Build the Qwen3-VL interface used by the Juno policy."""
    vlm_name = config.framework.qwenvl.base_vlm
    if "Qwen3-VL" not in vlm_name:
        raise NotImplementedError(
            f"Juno currently supports Qwen3-VL checkpoints only; got {vlm_name!r}"
        )
    from .QWen3 import _QWen3_VL_Interface
    return _QWen3_VL_Interface(config)
