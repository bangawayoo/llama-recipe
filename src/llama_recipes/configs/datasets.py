# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/chat_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class chat_dataset:
    dataset: str = "chat_dataset"
    file: str = "examples/chat_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class instruction_dataset:
    dataset: str = "instruction_dataset"
    file: str = "examples/chat_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class info_dataset:
    dataset: str = "info_dataset"
    file: str = "examples/chat_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"