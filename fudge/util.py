
import math
import torch
import numpy as np
import pickle
from tqdm import tqdm
import string
from collections import defaultdict
from typing import Dict, List, Optional
import heapq
import random
import torch.nn.functional as F
from argparse import ArgumentParser
import logging
from typing import List

import csv 

# logging.basicConfig(
#     # encoding="utf-8",
#     level=logging.DEBUG,
#     handlers=[logging.FileHandler("output.log"), logging.StreamHandler()],
# )
from dec.bfs import assemble_pad, assemble_pad_plus, vanilla_heap_pop


from dec.util import BeamNode


from fudge.constants import PAD_TOKEN
from transformers import AutoTokenizer,AutoModelWithLMHead



def write_output_file_header(log_file, fields, prefix=''):
    with open(f"{prefix}{log_file}", 'w') as wf:
        writer = csv.DictWriter(wf, fieldnames=fields)
        writer.writeheader()


def write_output_file_line(log_file, fields, output_rows:List[dict], prefix=''):
    with open(f"{prefix}{log_file}", 'a') as wf:
        writer = csv.DictWriter(wf, fieldnames=fields)
        for row in output_rows:
            writer.writerow(row)


def write_topic_output_line(log_file, fields, cr_group, hyp_num):
    with open(log_file, 'a') as wf:
        writer = csv.DictWriter(wf, fieldnames=fields )
        for cr in cr_group[2]:
            writer.writerow({'category': cr_group[1], 'input_text': cr_group[0], 'generation': cr, 'num_hypo':hyp_num})


def unsqueeze_expand_sec_dim(inp, exp_size):
    num_axis = len(inp.size())
    if num_axis == 1:
        return inp.unsqueeze(1).expand(-1, exp_size)
    elif num_axis == 2:
        return inp.unsqueeze(1).expand(-1, exp_size, -1)
    else:
        raise NotImplementedError
    


def visualize_logits(logits, raw_indicies, name, tokenizer, idx=0, topk=10):
    # assume logits in batch
    tmp_logit = logits[idx]

    values, indices = torch.topk(tmp_logit, k=topk, dim=-1)  # batch x pretopk
    post_probs = F.softmax(values, dim=-1)
    post_probs_list = post_probs.tolist()
    tmp_raw = raw_indicies[idx]
    out = []
    for i in range(5):
        out.append(str(post_probs_list[i])[:4] + "  "+ tokenizer.decode(tmp_raw[indices[i]]) )
        # out.append( tokenizer.decode(tmp_raw[indices[i]]) )
    logging.info(name + ":\n" + "\n".join(out) + "\n-----------")


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def define_log_file_name(base, args):
    options =[args.task, args.algo, args.max_len, args.beam_size, args.group_size,args.constraint, args.task_rwd, args.temp_decay, args.heap_top_k, args.typical_p, args.top_p, args.num_beam_groups, args.threshold, args.len_factor]
    options = [str(x) for x in options]
    if args.demo:
        return f"demo_{'_'.join(options)}_{base}"
    else:
        return f"{'_'.join(options)}_{base}"

def setup_gpt(model_string,device):
    gpt_tokenizer = AutoTokenizer.from_pretrained(model_string)
    gpt_tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    gpt_pad_id = gpt_tokenizer.encode(PAD_TOKEN)[0]
    
    # when generating, we will use the logits of right-most token to predict the next token
    # so the padding should be on the left
    gpt_tokenizer.padding_side = "left" 
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token # to avoid an error

    gpt_model = AutoModelWithLMHead.from_pretrained(model_string).to(device)
    gpt_model.config.pad_token_id = gpt_model.config.eos_token_id
    gpt_model.eval()
    return gpt_model, gpt_tokenizer, gpt_pad_id

def gpt_decode(model, device, expansion_list:List[BeamNode], doc_input_token, tokenizer, doc_input_token_len=None):
    
    dec_prefixes = [x.get_token_idx() for x in expansion_list]
    concat_prefixes = [doc_input_token + x for x in dec_prefixes]

    (
        concat_input_tensor,
        padding_pos,
        attention_mask_concat_input_tensor,
    ) = assemble_pad_plus(
        concat_prefixes, device=device, pad_token_idx=tokenizer.bos_token_id
    )  # batch x seq      store [PAD] * ? + enc + dec
    batch_size, full_length = concat_input_tensor.size()
    # enc + dec length, exclude padding
    enc_dec_length = [full_length - x for x in padding_pos]
    if doc_input_token_len:
        # dec only length, exclude padding
        dec_length = [x - doc_input_token_len for x in enc_dec_length]

    dec_input_tensor = assemble_pad(
        dec_prefixes, device, tokenizer.bos_token_id
    )  # only store the decoded tokens after enc, batch x dec_seq_len

    output = model(
        input_ids=concat_input_tensor, attention_mask=attention_mask_concat_input_tensor        # shold use encoder_attention_mask instead of attention_mask!!!!!
    )
    logits = output.logits[:, -1, :]  # batch x seq_len x vocab =>  batch x vocab
    return logits, concat_input_tensor, dec_input_tensor, batch_size, padding_pos, enc_dec_length