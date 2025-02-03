import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Union

import torch

import transformer_lens

import transformer_lens.utils as utils
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

from transformer_lens.loading_from_pretrained import *

def get_pretrained_model_config_fixed(
    model_name: str,
    hf_cfg: Optional[dict] = None,
    checkpoint_index: Optional[int] = None,
    checkpoint_value: Optional[int] = None,
    fold_ln: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    n_devices: int = 1,
    default_prepend_bos: Optional[bool] = None,
    dtype: torch.dtype = torch.float32,
    first_n_layers: Optional[int] = None,
    **kwargs,
):

    if Path(model_name).exists():
        # If the model_name is a path, it's a local model
        cfg_dict = convert_hf_model_config(model_name, **kwargs)
        official_model_name = model_name
    else:
        official_model_name = get_official_model_name(model_name)
    if (
        official_model_name.startswith("NeelNanda")
        or official_model_name.startswith("ArthurConmy")
        or official_model_name.startswith("Baidicoot")
    ):
        cfg_dict = convert_neel_model_config(official_model_name, **kwargs)
    else:
        if official_model_name.startswith(NEED_REMOTE_CODE_MODELS) and not kwargs.get(
            "trust_remote_code", False
        ):
            logging.warning(
                f"Loading model {official_model_name} requires setting trust_remote_code=True"
            )
            kwargs["trust_remote_code"] = True
        cfg_dict = convert_hf_model_config(official_model_name, **kwargs)
    # Processing common to both model types
    # Remove any prefix, saying the organization who made a model.
    cfg_dict["model_name"] = official_model_name.split("/")[-1]
    # Don't need to initialize weights, we're loading from pretrained
    cfg_dict["init_weights"] = False

    if (
        "positional_embedding_type" in cfg_dict
        and cfg_dict["positional_embedding_type"] == "shortformer"
        and fold_ln
    ):
        logging.warning(
            "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_ln=False instead."
        )
        fold_ln = False

    if device is not None:
        cfg_dict["device"] = device

    cfg_dict["dtype"] = dtype

    if fold_ln:
        if cfg_dict["normalization_type"] in ["LN", "LNPre"]:
            cfg_dict["normalization_type"] = "LNPre"
        elif cfg_dict["normalization_type"] in ["RMS", "RMSPre"]:
            cfg_dict["normalization_type"] = "RMSPre"
        else:
            logging.warning("Cannot fold in layer norm, normalization_type is not LN.")

    if checkpoint_index is not None or checkpoint_value is not None:
        checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(
            official_model_name,
            **kwargs,
        )
        cfg_dict["from_checkpoint"] = True
        cfg_dict["checkpoint_label_type"] = checkpoint_label_type
        if checkpoint_index is not None:
            cfg_dict["checkpoint_index"] = checkpoint_index
            cfg_dict["checkpoint_value"] = checkpoint_labels[checkpoint_index]
        elif checkpoint_value is not None:
            assert (
                checkpoint_value in checkpoint_labels
            ), f"Checkpoint value {checkpoint_value} is not in list of available checkpoints"
            cfg_dict["checkpoint_value"] = checkpoint_value
            cfg_dict["checkpoint_index"] = checkpoint_labels.index(checkpoint_value)
    else:
        cfg_dict["from_checkpoint"] = False

    cfg_dict["device"] = device
    cfg_dict["n_devices"] = n_devices

    if default_prepend_bos is not None:
        # User explicitly set prepend_bos behavior, override config/default value
        cfg_dict["default_prepend_bos"] = default_prepend_bos
    elif "default_prepend_bos" not in cfg_dict:
        # No config value or user override, set default value (True)
        cfg_dict["default_prepend_bos"] = True

    if hf_cfg is not None:
        cfg_dict["load_in_4bit"] = hf_cfg.get("quantization_config", {}).get("load_in_4bit", False)
    if first_n_layers is not None:
        cfg_dict["n_layers"] = first_n_layers
    cfg_dict["d_vocab"] = hf_cfg["vocab_size"]
    cfg = HookedTransformerConfig.from_dict(cfg_dict)
    return cfg
transformer_lens.loading_from_pretrained.get_pretrained_model_config = get_pretrained_model_config_fixed
