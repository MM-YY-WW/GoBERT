#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import obonet
import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    BertForPreTraining,
    BertConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import send_example_telemetry
import torch.nn as nn
from datetime import datetime
import pytz
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR
from sklearn.metrics import roc_auc_score
import time

logger = get_logger(__name__)
# require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
    os.environ['MASTER_PORT'] = '12355'  # ensure this port is available
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def compute_topk_accuracy(logits, labels, k=5):
    mask = labels.squeeze(-1) != -100
    masked_logits = logits[mask].view(-1, logits.size(-1))
    masked_labels = labels[mask].view(-1)
    _, top_k_indices = torch.topk(masked_logits, k, dim=-1)
    correct = torch.tensor([masked_labels[i] in top_k_indices[i] for i in range(masked_labels.size(0))]).sum()
    total = mask.sum()
    return correct.float(), total.float()

def compute_accuracy(logits, labels):
    # Get the most probable tokens predicted by the model
    predictions = torch.argmax(logits, dim=-1)
    # We need to count only the masked tokens, which are labeled as not -100
    mask = labels != -100
    correct = (predictions[mask] == labels[mask]).sum()
    total = mask.sum()
    return correct.float(), total.float()


def compute_accuracy_depthlimit(logits, labels, depth_tensor, k=5):
    mask = labels.squeeze(-1) != -100
    masked_logits = logits[mask].view(-1, logits.size(-1))
    masked_labels = labels[mask].view(-1)
    label_depths = depth_tensor[masked_labels]
    predicted_depths = depth_tensor.unsqueeze(0).expand_as(masked_logits)
    depth_mask = (predicted_depths == label_depths.unsqueeze(1)).float()
    filtered_logits = masked_logits * depth_mask + (1.0 - depth_mask) * (-1e9)
    _, top_k_indices = torch.topk(filtered_logits, k, dim=-1)
    correct = torch.tensor([masked_labels[i] in top_k_indices[i] for i in range(masked_labels.size(0))]).sum()
    total = mask.sum()
    return correct.float(), total.float()


def seed_all(seed_value, cuda_deterministic=False):
    """
    set all random seeds
    """
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
def compute_probability_of_unseen_predictions(logits, input_ids, labels):
    """
    Compute the probability that the predicted token for each masked position
    is not already present in the input sequence.
    """
    predictions = torch.argmax(logits, dim=-1)  # (batch_size, sequence_length)
    unseen_predictions = 0

    # Loop through each sequence in the batch
    for i, input_ids_row in enumerate(input_ids):
        # Mask indices are positions where the labels are not -100 (masked places)
        mask_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
        # Create a set of tokens already present in the input sequence
        filtered_tokens = [token for idx, token in enumerate(input_ids_row.tolist()) if idx not in mask_indices]
        present_tokens = set(filtered_tokens)

        # Check predictions at each mask index
        for mask_idx in mask_indices:
            predicted_token = predictions[i, mask_idx].item()
            if predicted_token not in present_tokens:
                unseen_predictions += 1

    return unseen_predictions

class GoBERTModel_Ex_Im(BertForPreTraining):
    def __init__(self, config, lam=0.5, semantic_path =""):
        super(GoBERTModel_Ex_Im, self).__init__(config)

        if semantic_path != "":
            # goterm_emb = torch.load(semantic_path).squeeze(1).cpu()
            goterm_emb = torch.tensor(np.load(semantic_path), dtype=torch.float)
        else:
            goterm_emb = torch.randn(47729, 1024)
        special_tokens = torch.randn(5, goterm_emb.shape[1], dtype=torch.float)
        full_word_emb = torch.cat([goterm_emb, special_tokens], dim=0)

#------------------------------------------------------weight---------------------------------------------------------------
        if full_word_emb.shape[1] != config.hidden_size:
            projected_goterm_emb = torch.nn.functional.linear(full_word_emb, torch.nn.Linear(full_word_emb.shape[1], config.hidden_size).weight)
        else:
            projected_goterm_emb = full_word_emb

        original_word_embeddings = self.bert.embeddings.word_embeddings.weight.data

        if original_word_embeddings.size(0) != projected_goterm_emb.size(0):
            raise ValueError("The number of tokens in the original and goterm embeddings do not match.")
        alpha = 0.1
        blended_embeddings = alpha * original_word_embeddings + (1 - alpha) * projected_goterm_emb
        self.bert.embeddings.word_embeddings = torch.nn.Embedding(blended_embeddings.size(0), blended_embeddings.size(1))
        self.bert.embeddings.word_embeddings.weight.data.copy_(blended_embeddings)

        ex_labels = torch.load('data/sparse_adj_matrix.pt', weights_only=True).to_dense()
        self.register_buffer('ex_label_buffer', ex_labels)
        self.lam = lam

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, next_sentence_label=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None, sepcial_tokens_mask=None):

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            next_sentence_label=next_sentence_label,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        lam=self.lam

#----------------------------------------------------------------------------------------------------------------
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(outputs.prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))

#----------------------------------------------------------------------------------------------------------------
        neighbor_labels = (self.ex_label_buffer[input_ids]).view(-1, self.config.vocab_size)
        downsample_rate = 0.001  # Keep 10% of negatives
        random_mask = (torch.rand(neighbor_labels.shape, device=neighbor_labels.device) < downsample_rate).float()
        filtered_labels = (neighbor_labels + (1 - neighbor_labels) * random_mask).clamp_(0, 1).view(-1, self.config.vocab_size)

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1337.7).to(neighbor_labels.device))
        neighbor_loss = loss_fct(outputs.prediction_logits.view(-1, self.config.vocab_size), filtered_labels)

#----------------------------------------------------------------------------------------------------------------

        total_loss = lam * masked_lm_loss.mean() + (1 - lam) * neighbor_loss.mean()
        neighbor_probs = torch.sigmoid(outputs.prediction_logits.view(-1, self.config.vocab_size))
        neighbor_preds = (neighbor_probs > 0.5).float()
        neighbor_correct_predictions = (neighbor_preds == filtered_labels).sum().item()
        neighbor_total_predictions = filtered_labels.numel()

        true_positives = (neighbor_preds * filtered_labels).sum(dim=0)
        predicted_positives = neighbor_preds.sum(dim=0)
        actual_positives = filtered_labels.sum(dim=0)
        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)
        neighbior_f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        neighbior_f1 = neighbior_f1.mean().item() 
#----------------------------------------------------------------------------------------------------------------

        return {'loss': total_loss, 'prediction_logits': outputs.prediction_logits, 'seq_relationship_logits': outputs.seq_relationship_logits,
                 'neighbor_correct_predictions': neighbor_correct_predictions, "neighbor_total_predictions": neighbor_total_predictions, "neighbor_f1": neighbior_f1, "lam": lam}


def custom_mask_tokens(inputs, tokenizer, mlm_probability=0.15, root_token_ids=None, go_graph=None, seed=None, token_id_go_id_dict=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling.
    Masking some random tokens for the MLM task with probabilities as defined in mlm_probability.
    """
    # Set random seed for reproducibility
    # print(len(go_graph.nodes())) 
    if seed:
        # print(f"set random state seed for masking seed {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    # print("apply custome mask")
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability mlm_probability)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    # Ensure root tokens are not masked
    if root_token_ids is not None:
        for root_id in root_token_ids:
            root_token_mask = inputs == root_id
            probability_matrix.masked_fill_(root_token_mask, value=0.0)

    # Ensure selected masked nodes are not parents of any other nodes in the sequence
    if go_graph is not None:
        special_token_ids = set(tokenizer.all_special_ids)
        # print(special_token_ids)
        # time.sleep(100)

        for i in range(inputs.size(0)):  # iterate over each sequence in the batch
            for j in range(inputs.size(1)):  # iterate over each token in the sequence
                token_id = inputs[i, j].item()
                if token_id not in special_token_ids and token_id_go_id_dict[str(token_id)] in go_graph.nodes() and token_id not in root_token_ids:
                    children = set(list(go_graph.predecessors(token_id_go_id_dict[str(token_id)])))
                    mask_token = True
                    if children & {token_id_go_id_dict[str(token_id)] for token_id in (set(inputs[i].tolist()) - special_token_ids)} != set():
                        # Do not mask the parent node
                        mask_token = False
                    if mask_token and random.random() < mlm_probability:
                        probability_matrix[i, j] = 1.0

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15, root_token_ids=None, go_graph=None, max_length=100, seed=None, token_id_go_id_dict=None):
        super().__init__(tokenizer=tokenizer, mlm_probability=mlm_probability)
        self.root_token_ids = root_token_ids
        self.go_graph = go_graph
        self.max_length = max_length
        self.seed = seed
        self.token_id_go_id_dict=token_id_go_id_dict

    def __call__(self, examples):
        batch = self.tokenizer.pad(
            examples,
            padding="max_length",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        inputs, labels = custom_mask_tokens(batch["input_ids"], self.tokenizer, mlm_probability=self.mlm_probability, root_token_ids=self.root_token_ids, go_graph=self.go_graph, seed=self.seed, token_id_go_id_dict=self.token_id_go_id_dict)
        batch["input_ids"] = inputs
        batch["labels"] = labels
        return batch

def save_checkpoint(accelerator, model, optimizer, scheduler, epoch, step, output_dir, tokenizer=None, safe_serialization=False):
    unwrapped_model = accelerator.unwrap_model(model)
    output_dir = f"{output_dir}/epoch_{epoch}"
    # output_dir = f"{output_dir}/best_vaild"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    state_dict = unwrapped_model.state_dict()
    torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
    unwrapped_model.config.save_pretrained(output_dir)
    if tokenizer:
        tokenizer.save_pretrained(output_dir)
    state = {
        'epoch': epoch,
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(state, os.path.join(output_dir, 'training_state.pth'))
    accelerator.save_state(output_dir, safe_serialization=safe_serialization)


def load_checkpoint(accelerator, optimizer, scheduler, checkpoint_path):
    state = torch.load(os.path.join(checkpoint_path, 'training_state.pth'), map_location='cpu')
    accelerator.load_state(checkpoint_path, map_location="cpu")
    # accelerator.load_state(checkpoint_path)    
    # state_dict = torch.load("go_bert/extrago/Ex_Im_mask_unique_depth_topk_0.95_111/best_vaild/pytorch_model_1.bin", map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])
    scheduler.last_epoch = state.get('step', -1)
    epoch = state['epoch']
    step = state['step']
    del state
    torch.cuda.empty_cache()
    return optimizer, scheduler, epoch, step


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument("--validation_split_percentage",default=5,help="The percentage of the train set used as validation set in case there's no validation split",)
    parser.add_argument("--pad_to_max_length",action="store_true",help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",)
    parser.add_argument("--model_name_or_path",type=str,help="Path to pretrained model or model identifier from huggingface.co/models.",required=False,)
    parser.add_argument("--config_name",type=str,default=None,help="Pretrained config name or path if not the same as model_name",)
    parser.add_argument("--tokenizer_name",type=str,default=None,help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--use_slow_tokenizer",action="store_true",help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",)
    parser.add_argument("--per_device_train_batch_size",type=int,default=8,help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--per_device_eval_batch_size",type=int,default=8,help="Batch size (per device) for the evaluation dataloader.",)
    parser.add_argument("--learning_rate",type=float,default=5e-5,help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps",type=int,default=None,help="Total number of training steps to perform. If provided, overrides num_train_epochs.",)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--lr_scheduler_type",type=SchedulerType,default="linear",help="The scheduler type to use.",choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],)
    parser.add_argument("--custom_scheduler", type=str, default=None, choices=["onecycle","triangular"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--model_type",type=str,default=None,help="Model type to use if training from scratch.",choices=MODEL_TYPES,)
    parser.add_argument("--max_seq_length",type=int,default=None,help=("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."),)
    parser.add_argument("--line_by_line",type=bool,default=False,help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",)
    parser.add_argument("--preprocessing_num_workers",type=int,default=None,help="The number of processes to use for the preprocessing.",)
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--trust_remote_code",type=bool,default=False,help=("Whether or not to allow for custom models defined on the Hub in their own modeling files. This option ""should only be set to `True` for repositories you trust and in which you have read the code, as it will ""execute code present on the Hub on your local machine."),)
    parser.add_argument("--checkpointing_steps",type=str,default=None,help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,help="If the training should continue from a checkpoint folder.",)
    parser.add_argument("--with_tracking",action="store_true",help="Whether to enable experiment trackers for logging.",)
    parser.add_argument("--report_to",type=str,default="all",help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '"Only applicable when `--with_tracking` is passed."),)
    parser.add_argument("--low_cpu_mem_usage",action="store_true",help=("It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. ""If passed, LLM loading time and RAM consumption will be benefited."),)
    parser.add_argument("--local_rank", type=int, default=-1, help="local process rank")
    parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
    parser.add_argument("--eval_only", action="store_true", help="if pass this argument omit the training loop only do evaluation")
    parser.add_argument("--mask", type=int, default=1, help="if use special mask strategy then 1, else 0, for test should only use 1 to mask sure the fair comparision")
    parser.add_argument("--semantic_path", type=str, default="", help="the use of llm to capture semantic similarity as input, if not use just leave the path blank, default=Node_embedding_mxbai/47727.npy")
    parser.add_argument("--ratiomlmloss", type=float, default=0.5, help="artiomlmloss=lam, lam*mlmloss + (1-lam)*neighborhood loss, lam*Imloss + (1-lam)*Exloss")
    parser.add_argument("--mask_seed", type=int, default=None, help="the seed for designed masking strategy")
    args = parser.parse_args()


    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = args.local_rank
    is_master = local_rank == 0
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    print(local_rank)
    # device = torch.device("cuda", local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()
    seed_all(args.seed + torch.distributed.get_rank())

    send_example_telemetry("run_mlm_no_trainer", args)

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    if is_master:
        today_date = datetime.utcnow().astimezone(pytz.timezone('America/Chicago')).strftime("%Y_%m_%d")
        log_directory = f"{args.output_dir}/{today_date}"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        filetime = datetime.utcnow().astimezone(pytz.timezone('America/Chicago')).strftime("%H_%M_%S")
        if args.eval_only:
            log_file_path = f'{args.output_dir}/test_results.log'
        else:
            log_file_path = os.path.join(log_directory, f"{filetime}.log")
        logging.basicConfig(
            filename = log_file_path,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        logger.info(args)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_name = args.hub_model_idtied
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
        extension = args.train_file.split(".")[-1]
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files)
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{args.validation_split_percentage}%]",
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{args.validation_split_percentage}%:]",
        )

    config = CONFIG_MAPPING[args.model_type]()
    logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError("You are instantiating a new tokenizer from scratch. This is not supported by this script. ""You can do it from another script, save it, and load it from here, using --tokenizer_name.")


    #Initialize the bert configuration
    config = BertConfig(
        vocab_size=len(tokenizer),  # Size of the tokenizer's vocabulary
        hidden_size=1024, #768 #1024 #4096
        num_hidden_layers=10,
        num_attention_heads=16, #12
        intermediate_size=3072,
        hidden_act='gelu',
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=len(tokenizer),
        type_vocab_size=2,
        initializer_range=0.02
    )
    print(len(tokenizer))

    model = GoBERTModel_Ex_Im(config=config, lam=args.ratiomlmloss, semantic_path=args.semantic_path)
    if is_master:
        logger.info(config)
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 768:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 768. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 768
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=False,
            )

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )
        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result
        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    if len(train_dataset) > 3:
        if is_master:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    go_graph = obonet.read_obo('go-basic.obo', ignore_obsolete=True)
    if args.mask == 1:
        token_id_go_id = pd.read_csv(f"{args.tokenizer_name}/vocab.txt", header=None)
        token_id_go_id_dict = {str(idx): goid for idx, goid in zip(token_id_go_id.index, token_id_go_id[0])}
        roots = [term for term in go_graph if not list(go_graph.successors(term))]
        roots_token_ids =tokenizer("".join(roots))['input_ids'][1:-1]
        data_collator = CustomDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability, root_token_ids=roots_token_ids, go_graph=go_graph, seed=args.mask_seed, token_id_go_id_dict=token_id_go_id_dict)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)


    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(dataset=eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    if args.custom_scheduler == "onecycle":
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            total_steps=len(train_dataloader) * args.num_train_epochs,
            pct_start=0.5,  # Fraction of the cycle for the increasing phase
            anneal_strategy='linear',  # 'linear' or 'cos'
            last_epoch=-1,
            # cycle_momentum=args.cycle_momentum,  # Whether to cycle momentum
        )
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes,
        )

    # model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    # )
    model, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, lr_scheduler
    )
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("mlm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if is_master:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            optimizer, lr_scheduler, start_epoch, start_step = load_checkpoint(
                accelerator, optimizer, lr_scheduler, args.resume_from_checkpoint
            )
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint} (epoch {start_epoch}, step {start_step})")
            checkpoint_path = args.resume_from_checkpoint
            path=os.path.basename(checkpoint_path)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        training_difference = path
        print(training_difference)


        if "epoch" in training_difference:
            starting_epoch= int(float(training_difference.replace("epoch_", "")) //1)
            resume_step = int(float(training_difference.replace("epoch_", "")) * len(train_dataloader))
            completed_steps = resume_step
        else:
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    dist.barrier()
    
    depth_tensor = torch.tensor(np.load("data/depth_array.npy"), dtype=torch.float, device=device)
    best_valid_acc = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        if is_master:
            logging.info(f'-------epoch {epoch}')    
        if args.eval_only:
            if is_master:
                print("--eval_only is passed, omit training loop")
            pass
        else:
            model.train()
            dist.barrier()
            if args.with_tracking:
                total_loss = 0
            active_dataloader = train_dataloader
            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step % len(train_dataloader))                
            total_correct = 0
            train_topk_correct = 0
            train_topk_depth_correct = 0
            total_masked_tokens = 0
            neighbor_total_correct = 0
            neighbor_total_predict = 0
            neighbor_auc = []
            lam = []
            for step, batch in enumerate(active_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch, output_hidden_states = False)
                    loss = outputs['loss']
                    if args.with_tracking:
                        total_loss += loss.item()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                neighbor_total_correct += outputs['neighbor_correct_predictions']
                neighbor_total_predict += outputs['neighbor_total_predictions']
                neighbor_auc.append(outputs['neighbor_f1'])
                # lam.append(outputs["lam"].item())
                lam.append(outputs["lam"])
                logits = outputs['prediction_logits']  # Assumed to be present
                labels = batch['labels']  # Assumed labels are provided in the batch
                # correct, total = compute_accuracy(logits, labels)
                topkcorrect, total = compute_topk_accuracy(logits, labels, k=5)
                correct, total = compute_accuracy(logits, labels)
                topk_depth_correct, total = compute_accuracy_depthlimit(logits, labels, depth_tensor)
                total_correct += correct
                train_topk_correct += topkcorrect
                train_topk_depth_correct += topk_depth_correct
                total_masked_tokens += total
                if accelerator.sync_gradients and is_master:
                    progress_bar.update(1)
                    completed_steps += 1


                # if isinstance(checkpointing_steps, int) and is_master:
                #     if completed_steps % checkpointing_steps == 0:
                #         save_checkpoint(accelerator, model, optimizer, lr_scheduler, (epoch + step/len(active_dataloader)), step, args.output_dir, tokenizer, safe_serialization=False)
                #         if accelerator.is_main_process:
                #             tokenizer.save_pretrained(f"{args.output_dir}/tokenizer_epoch_{epoch + step/len(active_dataloader)}")

                if is_master:
                    if step !=0:
                        if step %10 == 0 or step == len(active_dataloader):
                            epoch_accuracy = (total_correct / total_masked_tokens).item() if total_masked_tokens > 0 else 0
                            topktrain_accuracy = (train_topk_correct / total_masked_tokens).item() if total_masked_tokens > 0 else 0
                            topk_depth_train_accuracy = (train_topk_depth_correct / total_masked_tokens).item() if total_masked_tokens > 0 else 0
                            epoch_loss = (total_loss / step) if total_masked_tokens > 0 else 0
                            current_lr = optimizer.param_groups[0]['lr']
                            if resume_step != None and epoch==starting_epoch:
                                print(step, resume_step)
                                print(f"Training step {step+(resume_step)} | Epoch {(step+(resume_step)) /len(train_dataloader): .3f} | ACC: {epoch_accuracy:.4f} | Top k ACC: {topktrain_accuracy:.4f} | Topk depth ACC: {topk_depth_train_accuracy} | LOSS: {epoch_loss:.6f} | LR: {current_lr} | Neighbor accuracy {(neighbor_total_correct/neighbor_total_predict):.4f} | Neighbor F1 {np.mean(neighbor_auc):.4f} | lam {np.mean(lam):.4f}")
                                logger.info(f"Training step {step+(resume_step)} | Epoch {(step+(resume_step)) /len(train_dataloader): .3f} | ACC: {epoch_accuracy:.4f} | Top k ACC: {topktrain_accuracy:.4f} | Topk depth ACC: {topk_depth_train_accuracy} | LOSS: {epoch_loss:.6f} | LR: {current_lr} | Neighbor accuracy {(neighbor_total_correct/neighbor_total_predict):.4f} | Neighbor F1 {np.mean(neighbor_auc):.4f} | lam {np.mean(lam):.4f}")
                            else:
                                logger.info(f"Training step {step} | Epoch {epoch + (step/len(train_dataloader)): .3f} | ACC: {epoch_accuracy:.4f} | Top k ACC: {topktrain_accuracy:.4f} | Topk depth ACC: {topk_depth_train_accuracy} |  LOSS: {epoch_loss:.6f} | LR: {current_lr} | Neighbor accuracy {(neighbor_total_correct/neighbor_total_predict):.4f} | Neighbor F1 {np.mean(neighbor_auc):.4f} | lam {np.mean(lam):.4f}")
                                print(f"Training step {step} | Epoch {(epoch + step/len(train_dataloader)): .3f} | ACC: {epoch_accuracy:.4f} | Top k ACC: {topktrain_accuracy:.4f} | Topk depth ACC: {topk_depth_train_accuracy} |  LOSS: {epoch_loss:.6f} | LR: {current_lr} | Neighbor accuracy {(neighbor_total_correct/neighbor_total_predict):.4f} | Neighbor F1 {np.mean(neighbor_auc):.4f} | lam {np.mean(lam):.4f}")

        model.eval()
        losses = []
        dist.barrier()
        eval_total_correct = 0
        eval_topk_correct = 0
        eval_topk_depth_correct = 0
        eval_total_masked_tokens = 0
        eval_top1_depth_correct = 0
        unseen_prediction = 0
        rand_mean = 0
        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states= True)             
                logits = outputs['prediction_logits']
                labels = batch['labels']
                topkcorrect, total = compute_topk_accuracy(logits, labels, k=5)
                correct, total = compute_accuracy(logits, labels)
                topk_depth_correct, total = compute_accuracy_depthlimit(logits, labels, depth_tensor, k=5)
                top1_depth_correct, total = compute_accuracy_depthlimit(logits, labels, depth_tensor, k=1)
                eval_total_correct += correct
                eval_topk_correct += topkcorrect
                eval_topk_depth_correct += topk_depth_correct
                eval_top1_depth_correct += top1_depth_correct
                eval_total_masked_tokens += total
                loss = outputs['loss']
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
                # unseen_prediction += compute_probability_of_unseen_predictions(logits=logits, input_ids = batch['input_ids'], labels = batch['labels'], depth_tensor=depth_tensor)
                unseen_prediction += compute_probability_of_unseen_predictions(logits=logits, input_ids = batch['input_ids'], labels = batch['labels'])

        losses = torch.cat(losses)
        
        if is_master:
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")
            
            eval_accuracy = (eval_total_correct / eval_total_masked_tokens).item() if eval_total_masked_tokens > 0 else 0
            topkeval_accuracy = (eval_topk_correct / eval_total_masked_tokens).item() if eval_total_masked_tokens > 0 else 0
            topk_depth_eval_accuracy = (eval_topk_depth_correct / eval_total_masked_tokens).item() if eval_total_masked_tokens > 0 else 0
            top1_depth_eval_accuracy = (eval_top1_depth_correct / eval_total_masked_tokens).item() if eval_total_masked_tokens > 0 else 0
            if args.eval_only:
                print(f"Test: Seed:{args.mask_seed} | Test Loss {eval_loss.item():.4f} | Accuracy: {eval_accuracy:.4f} | Top 5 Acc: {topkeval_accuracy:.4f} | Top1 depth acc:{top1_depth_eval_accuracy:.4f} Top 5 depth acc:{topk_depth_eval_accuracy: .4f}")
                print(f"Test: epoch {epoch}: Test perplexity: {perplexity} Test eval_loss: {eval_loss}, Test unseen={unseen_prediction}, Test total_mask={eval_total_masked_tokens} Test unseen probability: {unseen_prediction/eval_total_masked_tokens:.4f}, Test random cls cos sim: {rand_mean:.4f}")
                logger.info(f"Test: Seed:{args.mask_seed} | Test Loss {eval_loss.item():.4f} | Accuracy: {eval_accuracy:.4f} | Top 5 Acc: {topkeval_accuracy:.4f} | Top1 depth acc:{top1_depth_eval_accuracy:.4f} Top 5 depth acc:{topk_depth_eval_accuracy: .4f}")
                logger.info(f"Test: epoch {epoch}: Test perplexity: {perplexity} Test eval_loss: {eval_loss}, Test unseen={unseen_prediction}, Test total_mask={eval_total_masked_tokens} Test unseen probability: {unseen_prediction/eval_total_masked_tokens:.4f}, Test random cls cos sim: {rand_mean:.4f}")
            else:
                print(f"Evaluation Loss: {eval_loss.item():.4f}, Accuracy: {eval_accuracy:.4f} | Top k Acc: {topkeval_accuracy:.4f} | Top K depth acc:{topk_depth_eval_accuracy: .4f}")
                print(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}, unseen={unseen_prediction}, total_mask={eval_total_masked_tokens} unseen probability: {unseen_prediction/eval_total_masked_tokens:.4f}, random cls cos sim: {rand_mean:.4f}")
                logger.info(f"Evaluation Loss: {eval_loss.item():.4f}, Accuracy: {eval_accuracy:.4f} | Top k Acc: {topkeval_accuracy:.4f} | Top K depth acc:{topk_depth_eval_accuracy: .4f}")
                logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss},  unseen={unseen_prediction}, total_mask={eval_total_masked_tokens} unseen probability: {unseen_prediction/eval_total_masked_tokens:.4f}, random cls cos sim: {rand_mean:.4f}")
        if is_master and eval_accuracy>best_valid_acc:
            if args.eval_only:
                pass
            else:
                best_valid_acc = eval_accuracy
                save_checkpoint(accelerator, model, optimizer, lr_scheduler, epoch+1, step, args.output_dir, tokenizer, safe_serialization=False)
                tokenizer.save_pretrained(f"{args.output_dir}/tokenizer_epoch_{epoch+1}")
                # tokenizer.save_pretrained(f"{args.output_dir}/tokenizer_best_valid")

        if args.push_to_hub and epoch < args.num_train_epochs - 1 and not args.eval_only:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )


        if args.checkpointing_steps == "epoch" and not args.eval_only:
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None and not args.eval_only:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

if __name__ == "__main__":
    main()
    cleanup()
