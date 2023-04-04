import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s -%(levelname)s- %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger.info('Importing...')

import time
from envparse import env
env.read_envfile('dev.env')

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import numpy as np

def log_filter(record: logging.LogRecord) -> bool:
    if record.msg.find('Using custom data configuration') != -1:
        return False
    if record.msg.find('Found cached dataset parquet') != -1:
        return False
    if record.msg.find('Loading cached processed dataset at') != -1:
        return False
    return True
import datasets
logging.getLogger(datasets.builder.__name__).addFilter(log_filter)
logging.getLogger(datasets.arrow_dataset.__name__).addFilter(log_filter)

from datasets import load_dataset

from transformers import (
    BertTokenizerFast, 
    AutoModelForCausalLM, 
    PreTrainedModel,
    DataCollatorForLanguageModeling,
    BatchEncoding,
    get_scheduler,
    TrainingArguments,
    GenerationConfig
)
import huggingface_hub._login as hub_login

import torch
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm, trange

import random

import glob
import re
import shutil

logger.info('Import done.')



hub_login.login(os.environ.get('HF_TOKEN'))

args = TrainingArguments(
    eval_steps=5000,
    num_train_epochs=2,
    logging_steps=1000,
    output_dir='model',
    logging_dir='model/log',
    max_steps=0,
    logging_first_step=False,
    no_cuda=False,
    evaluation_strategy='steps',
    do_train=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    seed=random.randint(0, 999),
    save_total_limit=50,
    overwrite_output_dir=False,
)
args.max_eval_steps = 0
args.should_continue = True
args.tokenizer_path_or_name = 'bert-base-chinese'
args.model_path_or_name = 'ckiplab/gpt2-tiny-chinese'
args.show_args = False
args.datasets = ['asadfgglie/lccc_base_zh', {'test': 'validation', 'train': 'train'}]
args.model_train_step_from = 0
if args.show_args:
    logger.info("Training/evaluation parameters \n%s", str(vars(args)))
if args.should_continue and args.overwrite_output_dir:
    raise ValueError('Can\'t use --overwrite_output_dir and --should_continue at same time')



def sorted_checkpoints(args: TrainingArguments, use_mtime=False) -> list[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*checkpoint-([0-9]+)$", path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def rotate_checkpoints(args: TrainingArguments, use_mtime=False) -> None:
    """Check if we should delete older checkpoint(s)"""
    if args.save_total_limit is None:
        return
    if args.save_total_limit <= 0:
        return

    checkpoints_sorted = sorted_checkpoints(args, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.debug("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        try:
            shutil.rmtree(checkpoint)
        except PermissionError:
            time_out = 600

            i = 0
            for i in range(time_out):
                time.sleep(1)
                try:
                    shutil.rmtree(checkpoint)
                except PermissionError:
                    pass
            
            if i == time_out -1:
                raise PermissionError(f'Can\'t delete checkpoint [{checkpoint}]')
                


def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f'Set seed: {seed}')



if (os.path.exists(args.output_dir) and os.listdir(args.output_dir)
    and args.do_train 
    and not args.overwrite_output_dir
    and not args.should_continue):
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

if args.should_continue:
    _sorted_checkpoints = sorted_checkpoints(args)
    if len(_sorted_checkpoints) == 0:
        raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
    else:
        args.model_path_or_name = _sorted_checkpoints[-1]
        args.model_train_step_from = int(_sorted_checkpoints[-1].split('-')[-1])
        args.seed = torch.load(os.path.join(args.model_path_or_name, 'training_args.bin')).seed

set_seed(args.seed)

if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) 
    and args.do_train 
    and args.overwrite_output_dir
    and not args.should_continue):
    for checkpoint in sorted_checkpoints(args):
        shutil.rmtree(checkpoint)
    if os.path.exists(os.path.join(args.output_dir, 'eval_results.txt')):
        os.remove(os.path.join(args.output_dir, 'eval_results.txt'))
    shutil.rmtree(args.logging_dir)



logger.info('Model Loading...')
logger.info(f'Model path or name: {args.model_path_or_name}')
try:
    logger.info(f'Tokenizer path or name: {args.model_path_or_name}')
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(args.model_path_or_name)
except OSError:
    logger.info(f'Can\'t find tokenizer in {args.model_path_or_name}.')
    logger.info(f'Tokenizer path or name {args.tokenizer_path_or_name}.')
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(args.tokenizer_path_or_name)
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(args.model_path_or_name)
config = GenerationConfig.from_model_config(model.generation_config)
config.max_new_tokens=50
config.do_sample=True
config.top_k=50
config.pad_token_id=tokenizer.pad_token_id
model.generation_config = config
logger.info('Model load done.')



logger.info(f'Datasets name: {args.datasets[0]}')
logger.info(f'Datasets config: {args.datasets[1]}')
logger.info('Datasets loading...')
train_dataset = load_dataset(args.datasets[0], use_auth_token=True, split=args.datasets[1]['train'])
test_dataset = load_dataset(args.datasets[0], use_auth_token=True, split=args.datasets[1]['test'])

def tokenize_function(example) -> BatchEncoding:
    return tokenizer(''.join('[SEP]'.join(example["dialog"]).split()), truncation=True)

logger.info('Datasets tokenize...')
tokenized_train_dataset = train_dataset.map(tokenize_function)
tokenized_test_dataset = test_dataset.map(tokenize_function)
logger.info('Datasets tokenize done.')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



tokenized_train_dataset = tokenized_train_dataset.remove_columns('dialog')
tokenized_train_dataset.set_format("torch")
train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=data_collator)

tokenized_test_dataset = tokenized_test_dataset.remove_columns('dialog')
tokenized_test_dataset.set_format("torch")
test_dataloader = DataLoader(tokenized_test_dataset, shuffle=True, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator)



optimizer = AdamW(
    model.parameters(),
    lr=args.learning_rate, 
    betas=(args.adam_beta1, args.adam_beta2), 
    eps=args.adam_epsilon, 
    weight_decay=args.weight_decay
)

num_training_steps = min(args.num_train_epochs * len(train_dataloader), args.max_steps) if args.max_steps > 0 else args.num_train_epochs * len(train_dataloader)
lr_scheduler: LambdaLR = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

try:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not args.no_cuda:
        model.to(device)
    else:
        device = torch.device("cpu")
except:
    device = torch.device("cpu")
logger.info(f'Device: {device}')
# Check if saved optimizer or scheduler states exist
if (
    args.model_path_or_name
    and os.path.isfile(os.path.join(args.model_path_or_name, "optimizer.pt"))
    and os.path.isfile(os.path.join(args.model_path_or_name, "scheduler.pt"))
):
    # Load in optimizer and scheduler states
    optimizer.load_state_dict(torch.load(os.path.join(args.model_path_or_name, "optimizer.pt")))
    lr_scheduler.load_state_dict(torch.load(os.path.join(args.model_path_or_name, "scheduler.pt")))



def evaluate(max_eval_steps=0) -> dict[str, Tensor]:
    global global_step, args
    eval_output_dir = args.output_dir

    nb_eval_steps = 0
    eval_loss = 0.0
    model.eval()
    
    batch_bar = tqdm(leave=False, iterable=range(min(max_eval_steps, len(test_dataloader)) if max_eval_steps > 0 else len(test_dataloader)), desc="Batch evaluating step")
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if batch['input_ids'].shape[1] > 1024: continue

        with torch.no_grad():
            outputs = model(**batch)
            loss: Tensor = outputs.loss
            eval_loss += loss.mean().item()
        
        nb_eval_steps += 1

        batch_bar.update()
        
        if max_eval_steps > 0 and nb_eval_steps > max_eval_steps:
            break
    
    batch_bar.close()

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, 'loss': eval_loss}

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        for key in sorted(result.keys()):
            writer.write(f"step {global_step}, {key} = {result[key].item()}\n")
    
    return result


logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_dataset))
logger.info("  Num Epochs = %d", args.num_train_epochs)
logger.info("  Total optimization steps = %d", num_training_steps)

tb_writer = SummaryWriter(log_dir=args.logging_dir)

epoch_bar = trange(args.num_train_epochs, desc="Epoch")
progress_bar = trange(num_training_steps, desc="Total training step")

global_step = 0
train_loss = 0
logging_loss = 0

model.zero_grad()

def eval() -> None:
    global global_step, tb_writer, logger
    results = evaluate(args.max_eval_steps)
    for key, value in results.items():
        tb_writer.add_scalar("eval_{}".format(key), value, global_step)

for epoch in epoch_bar:

    batch_bar = tqdm(train_dataloader, desc="Batch training step")
    for batch in batch_bar:
        if args.model_train_step_from > global_step:
            progress_bar.update()
            global_step += 1
            continue

        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            if model.device.type != device.type:
                model.to(device)
        except torch.cuda.OutOfMemoryError:
            batch = {k: v.to('cpu') for k, v in batch.items()}
            model.to('cpu')
            torch.cuda.empty_cache()

        if batch['input_ids'].shape[1] > 1024:
            continue

        model.train()
        outputs = model(**batch)
        loss: Tensor = outputs.loss
        try:
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            loss.to('cpu')
            model.to('cpu')
            loss.backward()
        train_loss += loss.to('cpu').item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        def log() -> None:
            global train_loss, logging_loss, logger, tb_writer, lr_scheduler, global_step, args, model, tokenizer

            tb_writer.add_scalar("lr", np.array(lr_scheduler.get_last_lr()), global_step)
            tb_writer.add_scalar("loss", (train_loss - logging_loss) / args.logging_steps, global_step)
            logging_loss = train_loss

            
            checkpoint_prefix = "checkpoint"
            output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            rotate_checkpoints(args)

        global_step += 1
        progress_bar.update()

        if args.logging_steps > 0 and global_step % args.logging_steps == 0:
            log()
        
        if args.logging_first_step and global_step == 1:
            log()
        
        if (args.evaluation_strategy == 'steps' 
            and args.eval_steps is not None 
            and global_step % args.eval_steps == 0 
            and global_step > args.eval_delay):
            eval()

        if args.max_steps > 0 and global_step > args.max_steps:
            break
    
    batch_bar.close()
    torch.cuda.empty_cache()

    if args.max_steps > 0 and global_step > args.max_steps:
        break
    
    if args.evaluation_strategy == 'epoch' and epoch > args.eval_delay:
        eval()
    
    

tb_writer.close()
epoch_bar.close()
progress_bar.close()



if args.do_train:
    logger.info("Saving model checkpoint to %s", args.output_dir)

    checkpoint_prefix = "checkpoint"
    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step + args.model_train_step_from))
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    eval()