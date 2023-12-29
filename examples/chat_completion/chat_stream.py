# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
##
import fire
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
import sys

import torch
from transformers import LlamaTokenizer, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.inference.safety_utils import get_safety_checker


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


# kfkas/Llama-2-ko-7b-Chat, # meta-llama/Llama-2-7b-hf
model_name = "hyunseoki/ko-en-llama2-13b"
peft_model: str = "./ckpt/ko-en-llama"
quantization: bool = True
max_new_tokens = 100  # The maximum numbers of tokens to generate
min_new_tokens: int = 0  # The minimum numbers of tokens to generate
prompt_file: str = None
seed: int = 42  # seed value for reproducibility
safety_score_threshold: float = 0.5
do_sample: bool = False # Whether or not to use sampling ; use greedy decoding otherwise.
use_cache: bool = True  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
top_p: float = 1.0 # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
temperature: float = 1.0  # [optional] The value used to modulate the next token probabilities.
top_k: int = 50 # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
repetition_penalty: float = 1.0  # The parameter for repetition penalty. 1.0 means no penalty.
length_penalty: int = 1  # [optional] Exponential penalty to the length that is used with beam-based generation.
use_fast_kernels: bool = False  # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
# Set the seeds for reproducibility
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
model = load_model(model_name, quantization)
if peft_model:
    model = load_peft_model(model, peft_model)
if use_fast_kernels:
    """
    Setting 'use_fast_kernels' will enable
    using of Flash Attention or Xformer memory-efficient kernels 
    based on the hardware being used. This would speed up inference when used for batched inputs.
    """
    try:
        from optimum.bettertransformer import BetterTransformer

        model = BetterTransformer.transform(model)
    except ImportError:
        print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

# tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(
    {
        "pad_token": "<PAD>",
    }
)
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

stop_words = ["[/INST]", "[INST]"]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for
                  stop_word in stop_words]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
##

def format_user_input(user_input, start_of_diag=False):
    if start_of_diag:
        dialog_tokens = tokenizer.encode(f"{B_INST} {user_input.strip()} {E_INST}")
    else:
        dialog_tokens = tokenizer.encode(f"{user_input.strip()} {E_INST}")

    return dialog_tokens


print("Starting Dialogue")
history = f""
print("유저: ", end="")
user_input = input()
history = history + user_input.strip()
chat = format_user_input(history, start_of_diag=True)
while True:
    with torch.no_grad():
        tokens = torch.tensor(chat).long()
        tokens = tokens.unsqueeze(0)
        tokens = tokens.to("cuda:0")
        outputs = model.generate(
            input_ids=tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            stopping_criteria=stopping_criteria,
        )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output_text.replace(history, "").replace(B_INST, "").replace(E_INST, "")
        print(f"마루:{response}")
        print("유저: ", end="")
        user_input = input()
        if user_input == 'q':
            print("Exiting Dialogue...")
            break
        history = output_text
        history += f"{user_input.strip()} {E_INST}"
        chat = tokenizer.encode(history)
