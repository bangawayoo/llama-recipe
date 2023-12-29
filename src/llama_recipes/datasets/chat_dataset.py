# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import copy
# For dataset details visit: https://huggingface.co/datasets/samsum

from copy import deepcopy
import datasets
import itertools
import random

import torch

from llama_recipes.datasets.prompter import Prompter
from llama_recipes.datasets.utils import Concatenator
from llama_recipes.datasets.prompt import VOLCANO_PROMPT, B_SYS, E_SYS, B_INST, E_INST, USER, CHARACTER


from transformers import LlamaForCausalLM
def get_chat_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("csv", data_files="./data/1102-exp-2_1_1.csv", split=None)['train']
    dataset = dataset.shuffle()

    def tokenize_dialog(example, tokenizer, max_length=4096):
        dialog_tokens = tokenizer(
                f"{(example['Text']).strip()} {(example['Completion']).strip()} ",
                add_special_tokens=True)
        prompt_tokens = tokenizer(example['Text'], add_special_tokens=False)
        prompt_length = len(prompt_tokens['input_ids'])
        dialog_tokens['labels'] = deepcopy(dialog_tokens['input_ids'])
        dialog_tokens['labels'][:prompt_length] = [-100] * prompt_length
        return dialog_tokens

    dataset = dataset.map(lambda x: tokenize_dialog(x, tokenizer), remove_columns=dataset.features)

    return dataset

# def get_chat_dataset(dataset_config, tokenizer, split):
#     dataset = datasets.load_dataset("csv", data_files="./data/volcano.csv", split=None)['train']
#     # backward fill
#     topics = dataset['topic']
#     dataset = dataset.remove_columns(['topic'])
#     for idx in range(len(topics)):
#         topic = topics[idx]
#         if topic is None:
#             topics[idx] = topics[idx-1]
#     dataset = dataset.add_column('topic', topics)
#
#     def to_dialog(example):
#         thread = example['dialogue'].split("\n")
#         thread = [t for t in thread if len(t) > 0]
#         dialog = []
#         for i, line in enumerate(thread):
#             try:
#                 name_index = line.index(":")
#             except:
#                 name_index = -1
#             dialog.append({
#                 "role": "user" if i % 2 == 0 else "assistant",
#                 "content": line[name_index+1:].strip(),
#             })
#
#         if example['episode'] is None:
#             example['episode'] = "NA"
#         prompt = VOLCANO_PROMPT.format(topic=example['topic'], episode=example['episode'].strip())
#         return {"dialog": dialog, 'prompt': prompt}
#
#     def tokenize_dialog(example, tokenizer, max_length=4096):
#         dialog = example['dialog']
#
#         dialog_tokens = [
#             tokenizer(
#                 f"{USER} {(prompt['content']).strip()} {CHARACTER} {(answer['content']).strip()} ",
#                 add_special_tokens=False,
#             )
#             for prompt, answer in zip(dialog[::2], dialog[1::2])
#         ]
#
#         if len(dialog) % 2:
#             dialog_tokens += [tokenizer(
#                 f"{CHARACTER} {(dialog[-1]['content']).strip()}",
#             )]
#
#         prompt_tokens = tokenizer(example['prompt'], add_special_tokens=False)
#         total_tokens = len(prompt_tokens['input_ids']) + sum(len(d['input_ids']) for d in dialog_tokens)
#         num_truncated = total_tokens - max_length
#         if num_truncated > 0:
#             prompt = tokenizer.decode(prompt_tokens['input_ids'][:-num_truncated])
#             prompt_tokens = tokenizer(prompt, add_special_tokens=False)
#
#         combined_tokens = deepcopy(prompt_tokens)
#         for k in dialog_tokens[0].keys():
#             combined_tokens[k] += list(itertools.chain(*(t[k] for t in dialog_tokens)))
#
#
#         combined_tokens['labels'] = deepcopy(combined_tokens['input_ids'])
#         prompt_length = len(prompt_tokens['input_ids'])
#         combined_tokens['labels'][:prompt_length] = [-100] * prompt_length
#         return combined_tokens
#
#     dataset = dataset.map(to_dialog)
#     dataset = dataset.map(lambda x: tokenize_dialog(x, tokenizer), remove_columns=dataset.features)
#
#     return dataset


def get_info_dataset(dataset_config, tokenizer, split):
    qa_dataset = datasets.load_dataset("csv", data_files="./data/volcano-knowledge.csv", split=None)['train']
    sum_dataset = datasets.load_dataset("csv", data_files="./data/volcano-summary.csv", split=None)['train']

    dataset = datasets.concatenate_datasets([qa_dataset, sum_dataset])
    augment_factor = 5

    for _ in range(augment_factor):
        tmp = copy.deepcopy(dataset)
        tmp = tmp.shuffle()
        dataset = datasets.concatenate_datasets([dataset, tmp])

    TASK_PROMPTS = {
        'age': ["{person}의 나이는?", "{person}은 몇 살인가?"],
        'faction': ["{person}의 종파 또는 소속은?", "{person} 은 어디 소속인가?"],
        'gender': ["{person}의 성별은?"],
        'introduction': ["청명이의 시점에서 {person}을 설명하시오.", "청명이가 {person}을 소개한다면?"],
        'relationship': ["청명이와 {person}의 관계는?", "청명이와 {person} 은 어떤 사이인가?"],
        'title': ["청명이는 {person}을 뭐라 부르는가?", "{person} 를 청명이는 뭐라 부르는가?"],
        'summary': ['에피소드{num_episode}를 청명 시점에서 요약하라.']
    }
    def add_prompt(example):
        task = example.get('task')
        prompts = TASK_PROMPTS[task]
        if task == "summary":
            prompt = random.sample(prompts, 1)[0].format(num_episode=example['input'])
        else:
            prompt = random.sample(prompts, 1)[0].format(person=example['input'])
        text = B_SYS + prompt + E_SYS + example['output']
        example['text'] = text
        return example


    dataset = dataset.map(add_prompt)
    dataset = dataset.map(lambda x: tokenizer(x['text']), remove_columns=dataset.features)
    concatenated_ds = dataset.map(Concatenator(chunk_size=4096), batched=True)

    return concatenated_ds

def get_instruction_dataset(dataset_config, tokenizer, split):
    data = datasets.load_dataset('kiyoonyoo/platypus-ko-en-v2')['train'].shuffle()
    prompter = Prompter("alpaca")
    cutoff_len = 4096
    train_on_inputs = False
    add_eos_token = False

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"])

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"])

            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token)

            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + \
                                              tokenized_full_prompt["labels"][user_prompt_len:]  # TODO: Speed up?
        return tokenized_full_prompt

    data = data.map(generate_and_tokenize_prompt, remove_columns=data.features)

    return data