import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robocasa_example() -> dict:
    """Creates a random input example for the RoboCasa policy."""
    return {
        "observation/state": np.random.rand(26),  
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
    
    RoboCasa dataset contains 26-dimensional state observations and 13-dimensional actions.
    It has three camera views: agentview_left, agentview_right, and eye_in_hand.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST.
        mask_padding = self.model_type == _model.ModelType.PI0

        # Pad the proprioceptive input (state) to the action dimension of the model
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Parse the three camera views provided in RoboCasa dataset
        agentview_left = _parse_image(data["observation/agentview_left"])
        agentview_right = _parse_image(data["observation/agentview_right"])
        eye_in_hand = _parse_image(data["observation/eye_in_hand"])

        # Create inputs dict with appropriate camera view mapping
        inputs = {
            "state": state,
            "image": {
                # Use the agent view (third-person perspectives) as the base view
                "base_0_rgb": eye_in_hand,
                # Use eye-in-hand as the left wrist image
                "left_wrist_0_rgb": agentview_left,
                # Use the right agent view for the right wrist image
                "right_wrist_0_rgb": agentview_right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension (for training)
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt if available
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCasaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format.
    It is used for inference only.
    
    RoboCasa uses 13-dimensional actions.
    """

    def __call__(self, data: dict) -> dict:
        # Return the first 13 actions (RoboCasa's action dimension)
        return {"actions": np.asarray(data["actions"][:, :13])}