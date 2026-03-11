import os
import torch

from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file as load_safetensor


def load_checkpoint(model_path):
    """
    Load model checkpoint supporting both:
    - safetensors
    - pytorch_model.bin
    """

    safetensor_path = os.path.join(model_path, "model.safetensors")
    bin_path = os.path.join(model_path, "pytorch_model.bin")

    if os.path.exists(safetensor_path):
        print("Loading safetensors checkpoint")
        return load_safetensor(safetensor_path)

    elif os.path.exists(bin_path):
        print("Loading PyTorch checkpoint")
        return torch.load(bin_path, map_location="cpu")

    else:
        raise RuntimeError("No checkpoint found in model path")


def load_partition(model_path, start_layer, end_layer):
    """
    Load only the transformer layers assigned to this node.
    """

    print("Loading model config...")

    config = AutoConfig.from_pretrained(model_path)

    print("Building model skeleton...")

    model = AutoModelForCausalLM.from_config(config)

    print("Loading checkpoint weights...")

    checkpoint = load_checkpoint(model_path)

    new_state = {}

    for key, value in checkpoint.items():

        # embeddings needed for Node1
        if key.startswith("model.embed_tokens"):
            new_state[key] = value

        # transformer layers
        elif key.startswith("model.layers"):

            layer_id = int(key.split(".")[2])

            if start_layer <= layer_id <= end_layer:
                new_state[key] = value

        # final normalization
        elif key.startswith("model.norm"):
            new_state[key] = value

        # lm_head needed for last node
        elif key.startswith("lm_head"):
            new_state[key] = value

    print("Loading filtered weights into model...")

    model.load_state_dict(new_state, strict=False)

    model.eval()

    torch.set_grad_enabled(False)

    print(f"Loaded layers {start_layer} → {end_layer}")

    return model