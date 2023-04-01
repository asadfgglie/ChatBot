import datetime

print('Importing...')
import numpy as np

from transformers import (
    BertTokenizerFast,
    AutoModelForCausalLM,
    PreTrainedModel,
    TrainingArguments,
    BatchEncoding,
)

import torch


import random
import os

import glob
import re

torch.cuda.empty_cache()

def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sorted_checkpoints(args: TrainingArguments, use_mtime=False) -> list[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*checkpoint-([0-9]+)", path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

args = TrainingArguments(
    output_dir='model/checkpoint-5005254-save',
    no_cuda=True,
    seed=random.randint(0, 999),
)
set_seed(args.seed)

print('loading')
model_name = args.output_dir
try:
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(model_name, padding_side='left')
except:
    try:
        model_name = sorted_checkpoints(args)[-1]
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(model_name, padding_side='left')
    except:
        raise ValueError(f'There\'s no model in {args.output_dir}')

print(f'load model: {model_name}')

device = torch.device('cuda') if torch.cuda.is_available() and not args.no_cuda else torch.device('cpu')

print(model.generation_config)
config = model.generation_config

model.to(device)
model.eval()

all_history = []
chat_history = []


def save_dialog():
    with open(os.path.join(model_name, 'dialog_log.txt'), 'a', encoding='UTF-8') as writer:
        now_time = str(datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y/%m/%d %H:%M:%S'))
        writer.write('Save at ' + now_time + '\n')
        for txt in all_history:
            writer.write(txt + '\n')
        writer.write('\n')


while True:
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    user_in = input("User: ")

    if user_in.startswith('/'):
        command_list = ['stop', 'clear', 'save', 'help']
        if user_in[1:] == command_list[0]:
            break
        if user_in[1:] == command_list[1]:
            chat_history = []
            all_history.append('\nNew dialog.\n')
            continue
        if user_in[1:] == command_list[2]:
            save_dialog()
            if not model_name.endswith('-save'):
                os.rename(model_name, model_name + '-save')
                model_name += '-save'
            continue
        if user_in[1:] == command_list[3]:
            print('/stop: stop this bot and exit the program.')
            print('/clear: clear all chat history in bot history.')
            print('/save: save all chat history and checkpoint. It will and "-save" at the last checkpoint-folder name.')
            continue
        else:
            print('Available command:', command_list)
            continue

    chat_history.append(user_in)
    all_history.append('User: ' + user_in)

    user_inputs: BatchEncoding = tokenizer(tokenizer.sep_token.join(chat_history), return_tensors='pt', truncation=True, max_length=1024-config.max_new_tokens)
    user_inputs.to(device)

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(generation_config=config, **user_inputs)
    
    chat_history.append(''.join(tokenizer.decode(chat_history_ids[:, user_inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True).split()))
    all_history.append('GPT:  ' + chat_history[-1])

    # pretty print last ouput tokens from bot
    print("GPT:  {}".format(chat_history[-1]), flush=True)