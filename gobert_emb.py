import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, BertConfig, BertForPreTraining
import torch.nn as nn
import os
import random
import pandas as pd
import pickle
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR


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
                output_hidden_states=True, return_dict=None, sepcial_tokens_mask=None):

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
#         lam=self.lam

# #----------------------------------------------------------------------------------------------------------------
#         loss_fct = nn.CrossEntropyLoss()
#         masked_lm_loss = loss_fct(outputs.prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))

# #----------------------------------------------------------------------------------------------------------------
#         neighbor_labels = (self.ex_label_buffer[input_ids]).view(-1, self.config.vocab_size)
#         downsample_rate = 0.001  # Keep 10% of negatives
#         random_mask = (torch.rand(neighbor_labels.shape, device=neighbor_labels.device) < downsample_rate).float()
#         filtered_labels = (neighbor_labels + (1 - neighbor_labels) * random_mask).clamp_(0, 1).view(-1, self.config.vocab_size)

#         loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1337.7).to(neighbor_labels.device))
#         neighbor_loss = loss_fct(outputs.prediction_logits.view(-1, self.config.vocab_size), filtered_labels)

# #----------------------------------------------------------------------------------------------------------------

#         total_loss = lam * masked_lm_loss.mean() + (1 - lam) * neighbor_loss.mean()
#         neighbor_probs = torch.sigmoid(outputs.prediction_logits.view(-1, self.config.vocab_size))
#         neighbor_preds = (neighbor_probs > 0.5).float()
#         neighbor_correct_predictions = (neighbor_preds == filtered_labels).sum().item()
#         neighbor_total_predictions = filtered_labels.numel()

#         true_positives = (neighbor_preds * filtered_labels).sum(dim=0)
#         predicted_positives = neighbor_preds.sum(dim=0)
#         actual_positives = filtered_labels.sum(dim=0)
#         precision = true_positives / (predicted_positives + 1e-8)
#         recall = true_positives / (actual_positives + 1e-8)
#         neighbior_f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
#         neighbior_f1 = neighbior_f1.mean().item() 
        return outputs.hidden_states[-1]

def gobert_emb(model, tokenizer, input_sequence, mask_token_id, device):
    if not input_sequence:
        raise ValueError("The input sequence is empty.")

    # Convert input sequence to token IDs
    input_ids = tokenizer(" ".join(input_sequence))['input_ids']

    # Prepare input tensor
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_tensor).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
        mean_pooled = outputs.squeeze(0)
        return mean_pooled.cpu().numpy()
        

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Inference for GoBERTModel_Ex_Im")
    parser.add_argument('--device', type=int, default=0, help='CUDA device index to use for inference')
    parser.add_argument('--input_file', type=str, default="example/input_goids.txt", help='Input file containing GO IDs, each line as a sample')
    parser.add_argument('--gene_name_list', type=str, default="example/input_gene_list.txt", help="list of a gene name, each line each name")
    parser.add_argument('--output_path', type=str, default='example/function_level_GoBERT_emb', help='Directory to save prediction outputs, file name be gene name')
    parser.add_argument('--model_path', type=str, default="ckpt/epoch_1", help='Path to the directory containing the model')
    parser.add_argument('--tokenizer_path', type=str, default="ckpt/tokenizer_best_epoch_1", help='Path to the directory containing the tokenizer')
    parser.add_argument('--obo_file', type=str, default="go-basic.obo", help='Path to the go-basic.obo file for GO term mapping')
    parser.add_argument('--seed', type=int, default=0, help="the seed")
    args = parser.parse_args()

    # Set device
    seed_all(seed_value=args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False, trust_remote_code=True)   
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
    model = GoBERTModel_Ex_Im(config=config, lam=0.5, semantic_path="")
    accelerator_log_kwargs = {}

    accelerator = Accelerator(gradient_accumulation_steps=1, **accelerator_log_kwargs)
    no_decay = ["bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-4)
    
    lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-4,
            total_steps=1000 * 10,
            pct_start=0.5,  # Fraction of the cycle for the increasing phase
            anneal_strategy='linear',  # 'linear' or 'cos'
            last_epoch=-1,
            # cycle_momentum=args.cycle_momentum,  # Whether to cycle momentum
        )
    model, lr_scheduler = accelerator.prepare(model, lr_scheduler)


    
    

    optimizer, lr_scheduler, start_epoch, start_step = load_checkpoint(accelerator, optimizer, lr_scheduler, args.model_path)
    # model.load_state_dict(torch.load(f"{args.model_path}/pytorch_model.bin", map_location=device, weights_only=True))
    model.to(device)


    with open(args.input_file, 'r') as f:
        samples = f.readlines()

    mask_token_id = tokenizer.mask_token_id

    os.makedirs(args.output_path, exist_ok=True)
    input_entrez_list = list(pd.read_csv(args.gene_name_list, header=None, dtype=str)[0])

    for idx, line in tqdm(enumerate(samples), total=len(input_entrez_list)):
        print(input_entrez_list[idx])
        input_sequence = line.strip().split()
        logit = gobert_emb(
            model=model,
            tokenizer=tokenizer, 
            input_sequence=input_sequence,
            mask_token_id=mask_token_id,
            device=device
        )
        with open(f"{args.output_path}/{input_entrez_list[idx]}.pkl", "wb") as f:
            pickle.dump(logit, f)
        


if __name__ == "__main__":
    main()
