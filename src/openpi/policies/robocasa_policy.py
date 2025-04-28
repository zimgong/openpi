import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robocasa_example() -> dict:
    """Creates a random input example for the RoboCasa policy."""
    return {
        "observation/state": np.random.rand(8),  
        "observation/agentview_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/agentview_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/eye_in_hand": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboCasaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We pad the proprioceptive input to the action dimension of the model.
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). 
        agentview_left = _parse_image(data["observation/agentview_left"])
        agentview_right = _parse_image(data["observation/agentview_right"])
        eye_in_hand = _parse_image(data["observation/eye_in_hand"])

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": eye_in_hand,
                "left_wrist_0_rgb": agentview_left,
                "right_wrist_0_rgb": agentview_right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCasaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. It is 
    used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        return {"actions": np.asarray(data["actions"][:, :8])}
