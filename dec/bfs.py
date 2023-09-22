"""
best-first search
"""

import torch
import math
from typing import Dict, List, Optional
import logging
import heapq
import random

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from dec.viz_center import proc
from dec.util import BeamNode,run_inference_step

# Best first search with k-expansion


def vanilla_heap_pop(cand_heap):
    """Just pop the 1st scored item from heap"""
    score, seed = heapq.heappop(cand_heap)
    return seed

def init_search(tokenizer, tok_dec_prefix, heap, stop_token_idx_list, T=0, force_words_ids=None):
    last = BeamNode(
        prob=1.0,
        token_idx=tokenizer.eos_token_id,  # empty
        token_str='',
        glob_step=proc(),
        prev=[],
        prev_score=[],
        stop_token_idx_list=stop_token_idx_list,
        batch_step=T
    )
    if force_words_ids:
        last.set_initial_constraint(force_words_ids)
    for prefix in tok_dec_prefix:
        tmp = BeamNode(
            prob=1.0,
            token_idx=prefix,
            token_str=tokenizer.decode(prefix),
            glob_step=proc(),
            prev=[last],
            prev_score=[0],
            stop_token_idx_list=stop_token_idx_list,
            batch_step=T
        )
        last = tmp
    init_seed = last
    heap.append([0,T, init_seed])
    # heapq.heappush(heap, (0, T, init_seed))


# one step of best first search. expand k hypothesis. the expansion strategy depends.
def step_bfs(model, heap, doc_input_ids,scorer, max_len, grp_size=5,k_best=10, book=None):
    # obtain a list of top nodes from heap.
    # the size of heap <= grp_size
    expansion_list = []
    finished_hypos = []
    cnt = 0
    while cnt < grp_size and heap:
        seed:BeamNode = vanilla_heap_pop(heap)
        expansion_list.append(seed)
        cnt += 1
    dec_prefixes = [ x.get_token_idx() for x in expansion_list]
    dec_input_tensors = assemble_pad(dec_prefixes,device=doc_input_ids.device)
    output_probs, _, _ = run_inference_step(model, doc_input_ids, decoder_input_ids=dec_input_tensors, device=doc_input_ids.device, output_dec_hid=False)
    values, indices = torch.topk(output_probs, k=k_best, dim=-1)
    batch_size = values.size()[0]
    values_np = values.cpu().tolist()
    indices_np = indices.cpu().tolist()
    for b in range(batch_size):
        for rank in range(k_best):
            v = values_np[b][rank]
            tok = indices_np[b][rank]
            tmp_state = BeamNode(
                prob=v, token_idx=tok, prev=[expansion_list[b]], prev_score=[math.log(v)],glob_step=proc())
            book.add_child()
            if tmp_state.finished:  # if this branch is a completed one, just put it in the outputs.
                finished_hypos.append(tmp_state)
                continue
            if tmp_state.length >= max_len:
                continue
            
            model_score = scorer.get_model_score(tmp_state)

            if random.random() < 0.01:
                logging.info(f"Score: {model_score}")

            heapq.heappush(heap, (-model_score, tmp_state))
    return finished_hypos, cnt

# assemble k decoding prefix
# need to verify if the padding makes sense to a model
def assemble_pad(list_of_tokens, device, pad_token_idx=1):
    cur_max_len = max([len(x) for x in list_of_tokens])
    padded_tokens = [[] for _ in range(len(list_of_tokens))]
    for idx in range(len(list_of_tokens)):
        raw = list_of_tokens[idx]
        padded = [pad_token_idx] * (cur_max_len - len(raw)) + raw
        padded_tokens[idx] = padded
    output = torch.tensor(padded_tokens,device=device, dtype=torch.long)
    # print(output)
    return output


# assemble k decoding prefix
# need to verify if the padding makes sense to a model
def assemble_pad_plus(list_of_tokens, device, pad_token_idx=50256):
    cur_max_len = max([len(x) for x in list_of_tokens])
    padded_tokens = [[] for _ in range(len(list_of_tokens))]
    end_of_paddings = [0 for _ in range(len(list_of_tokens))]
    attention_mask = torch.zeros((len(list_of_tokens), cur_max_len),device=device,dtype=torch.float)
    for idx in range(len(list_of_tokens)):
        raw = list_of_tokens[idx]
        pads_len = cur_max_len - len(raw)
        padded = [pad_token_idx] * pads_len + raw
        end_of_paddings[idx] = cur_max_len - len(raw)
        padded_tokens[idx] = padded
        attention_mask[idx][pads_len:] = 1
    output = torch.tensor(padded_tokens,device=device, dtype=torch.long)
    # print(output)
    return output, end_of_paddings, attention_mask

def right_side_padding(list_of_tokens, device, pad_token_idx=50256):
    cur_max_len = max([len(x) for x in list_of_tokens])
    padded_tokens = [[] for _ in range(len(list_of_tokens))]
    extract_position = [0 for _ in range(len(list_of_tokens))]  # where we get the prediction
    attention_mask = torch.ones((len(list_of_tokens), cur_max_len),device=device,dtype=torch.float)
    for idx in range(len(list_of_tokens)):
        raw = list_of_tokens[idx]
        len_non_pad = len(raw)
        pads_len = cur_max_len - len(raw)
        padded = raw +  [pad_token_idx] * pads_len
        extract_position[idx] = len(raw) - 1
        padded_tokens[idx] = padded
        attention_mask[idx][len_non_pad:] = 0
    output = torch.tensor(padded_tokens,device=device, dtype=torch.long)
    extract_position_tensor = torch.tensor(extract_position, device=device, dtype=torch.long)
    return output, extract_position_tensor, attention_mask

def best_first_search(model, tokenizer,
           doc_input_ids: torch.LongTensor,
        #    config_heu: Optional[Dict],
           config_search: Optional[Dict]=None,
           dec_prefix: Optional[List[int]]=None,
            scorer = None,
           max_len: Optional[int]=30,
           grp_size:Optional[int]=5,
           k_best: Optional[int]=5,
           comp_budget: Optional[int]=300,
           book=None,
           eos_idx:int = 2):
    ncalls = 0
    # heu_func = DeployHeu(config_heu)
    
    heap = []  # nodes at the frontier of search
    finished_hypos = []
    # config_search.in: each time we expand a node, we always extend to end
    # config_search.post: after exploration, we try to extend all of the non-finished nodes until reach the budget

    last = None
    for prefix in dec_prefix:
        if last:
            init_seed = BeamNode(prob=1., token_idx=prefix, glob_step=proc(),
                                 prev=[last], prev_score=[0], eos_idx=eos_idx)
        else:
            init_seed = BeamNode(prob=1., token_idx=prefix,glob_step=proc(),
                                 prev=[], prev_score=[], eos_idx=eos_idx)
            last = init_seed

    heapq.heappush(heap, (0, init_seed))

    while ncalls < comp_budget:
        completed_hyps, added_num_calls = step_bfs(model=model, heap=heap, doc_input_ids=doc_input_ids, max_len=max_len, grp_size=grp_size,k_best=k_best,scorer=scorer)

        ncalls += added_num_calls  

        if completed_hyps:
            finished_hypos += completed_hyps

    return finished_hypos, heap