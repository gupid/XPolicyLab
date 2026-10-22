"""Single-arm Franka recipe; statistics are supplied by the training cache."""
import copy
import os

from .va_robotwin30_train_cfg import va_robotwin30_lora_train_cfg
try:
    from ..real_world_dataset import ACTION_CHANNELS, INVERSE_CHANNELS, CAMERAS
except ImportError:  # upstream server imports configs as a top-level package
    from real_world_dataset import ACTION_CHANNELS, INVERSE_CHANNELS, CAMERAS

va_real_world_lora_cfg = copy.deepcopy(va_robotwin30_lora_train_cfg)
va_real_world_lora_cfg.update(
    __name__='Config: VA real-world LoRA', dataset_type='real_world_cached',
    used_action_channel_ids=ACTION_CHANNELS, inverse_used_action_channel_ids=INVERSE_CHANNELS,
    obs_cam_keys=CAMERAS, norm_stat=None,
    batch_size=8, gradient_accumulation_steps=1,
    enable_wandb=False, load_worker=4, warmup_steps=100, save_interval=500, validation_interval=500,
    action_metric_interval=100, action_metric_steps=10, action_metric_val_windows=8,
)

# For the upstream inference server, point this at the checkpoint's saved metadata.
metadata_path = os.getenv('LINGBOT_VA_REAL_METADATA')
if metadata_path:
    import json
    from pathlib import Path
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata['used_action_channel_ids'] != ACTION_CHANNELS:
        raise ValueError('Inference metadata has an incompatible action mapping')
    va_real_world_lora_cfg.norm_stat = metadata['norm_stat']
    va_real_world_lora_cfg.height = va_real_world_lora_cfg.width = metadata['image_size']
    va_real_world_lora_cfg.action_per_frame = metadata['stride'] * 4
