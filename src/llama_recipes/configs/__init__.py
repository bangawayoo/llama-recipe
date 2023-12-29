# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, asdict

from llama_recipes.configs.peft import lora_config, llama_adapter_config, prefix_config
from llama_recipes.configs.fsdp import fsdp_config
from llama_recipes.configs.training import train_config

def convert_to_dict(config: dataclass):
    dict = {}
    for key, _ in train_config.__annotations__.items():
        dict[key] = getattr(config, key)

    return dict