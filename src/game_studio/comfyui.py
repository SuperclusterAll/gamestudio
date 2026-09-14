"""Compile the supplied ComfyUI UI workflow into the /prompt API format."""

from __future__ import annotations

import json
from pathlib import Path


def load_z_image_turbo_prompt(
    workflow_path: Path,
    *,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    filename_prefix: str,
) -> dict[str, dict]:
    """Build ComfyUI's API prompt object from text_to_image_z_image_turbo_nodes.json.

    The supplied file is a UI graph, so widgets and links are translated to the
    class_type/inputs representation expected by POST /prompt.
    """
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    nodes = {str(node["id"]): node for node in workflow["nodes"]}
    values = {node_id: node.get("widgets_values", []) for node_id, node in nodes.items()}
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": values["1"][0], "weight_dtype": values["1"][1]}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": values["2"][0], "type": values["2"][1], "device": values["2"][2]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": positive_prompt}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative_prompt}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": values["5"][0]}},
        "6": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": values["6"][0]}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["6", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["7", 0], "seed": seed, "steps": values["8"][2], "cfg": values["8"][3], "sampler_name": values["8"][4], "scheduler": values["8"][5], "denoise": values["8"][6]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["5", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": filename_prefix}},
    }
