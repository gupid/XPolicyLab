"""Map the two recorded Franka cameras and preserve recorded EE deltas."""
import dataclasses

import cv2
import numpy as np
from openpi import transforms


def center_crop_resize(image, crop_size=640, output_size=224):
    """Center-crop RGB input to a square, then resize to the model resolution."""
    image = np.asarray(image)
    if image.shape == (output_size, output_size, 3):
        return image
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    height, width = image.shape[:2]
    if height < crop_size or width < crop_size:
        raise ValueError(f"Image {width}x{height} is smaller than required {crop_size}x{crop_size} crop")
    top, left = (height - crop_size) // 2, (width - crop_size) // 2
    crop = image[top:top + crop_size, left:left + crop_size]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    def __call__(self, data):
        images = {}
        for name, source in (("base_0_rgb", "camera_1"), ("left_wrist_0_rgb", "camera_2")):
            image = np.asarray(data[source])
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            if np.issubdtype(image.dtype, np.floating):
                image = (image * 255).round().clip(0, 255).astype(np.uint8)
            images[name] = center_crop_resize(image)
        images["right_wrist_0_rgb"] = np.zeros_like(images["base_0_rgb"])
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != 8:
            raise ValueError(f"Expected EE xyz+xyzw+measured gripper width, got {state.shape}")
        result = {"state": state, "image": images,
                  "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                                 "right_wrist_0_rgb": np.False_}}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 8:
                raise ValueError(f"Expected recorded delta xyz+xyzw+gripper command, got {actions.shape}")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        # Controller expects relative xyz+xyzw, negative=open / positive=close.
        # No absolute-pose conversion or quaternion repair is hidden here.
        return {"actions": np.asarray(data["actions"])[..., :8]}
