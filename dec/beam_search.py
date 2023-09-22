"""
best-first search
"""

import torch
import math
from typing import Dict, List, Optional
import logging
import heapq
import random
from viz_center import proc
from typing import Any, Callable, Dict, Iterable, List, Optional

from dec.util import BeamNode,run_inference_step,Scorer
# beam search

# one step of best first search. expand k hypothesis. the expansion strategy depends.
def step_bs(model, hypo, doc_input_ids, scorer, max_len, beam_size):
    # obtain a list of top nodes from heap.

    finished_hypos = []
    new_hypos = []

    dec_prefixes = [ x.get_token_idx() for x in hypo]
    dec_input_tensors = assemble_pad(dec_prefixes, device=doc_input_ids.device)
    output_probs, _, _ = run_inference_step(model, doc_input_ids, decoder_input_ids=dec_input_tensors, device=doc_input_ids.device, output_dec_hid=False)
    values, indices = torch.topk(output_probs, k=beam_size, dim=-1)
    batch_size = values.size()[0]
    values_np = values.cpu().tolist()
    indices_np = indices.cpu().tolist()
    for b in range(batch_size):
        for rank in range(beam_size):
            v = values_np[b][rank]
            tok = indices_np[b][rank]
            tmp_state = BeamNode(
                prob=v, token_idx=tok, prev=[hypo[b]], glob_step=proc(), prev_score=[math.log(v)])
            if tmp_state.finished:  # if this branch is a completed one, just put it in the outputs.
                finished_hypos.append(tmp_state)
                continue
            if tmp_state.length >= max_len:
                continue
            model_score = scorer.get_model_score(tmp_state)

            if random.random() < 0.01:
                logging.info(f"Score: {model_score}")

            new_hypos.append([-model_score, tmp_state])
    ordered = sorted(new_hypos, key=lambda tup:tup[0])
    ordered = ordered[:beam_size]
    ordered = [ x[1] for x in ordered]
    return finished_hypos, ordered

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



def beam_search(model, tokenizer,
           doc_input_ids: torch.LongTensor,
        #    config_heu: Optional[Dict],
            scorer,
           config_search: Optional[Dict]=None,
           dec_prefix: Optional[List[int]]=None,
           max_len: Optional[int]=30,
           beam_size: Optional[int]=10,
           book=None,
           eos_idx:int = 2):
    finished_hypos = []
    # config_search.in: each time we expand a node, we always extend to end
    # config_search.post: after exploration, we try to extend all of the non-finished nodes until reach the budget

    last = None

    for prefix in dec_prefix:
        if last:
            init_seed = BeamNode(prob=1., token_idx=prefix,glob_step=proc(),
                                 prev=[last], prev_score=[0], eos_idx=eos_idx)
        else:
            init_seed = BeamNode(prob=1., token_idx=prefix,glob_step=proc(),
                                 prev=[], prev_score=[], eos_idx=eos_idx)
            last = init_seed

    hypos = [init_seed]
    for idx in range(max_len):
        completed_hyps, new_hypos = step_bs(model=model, hypo=hypos, doc_input_ids=doc_input_ids, scorer=scorer,
        max_len=max_len, beam_size=beam_size)

        if completed_hyps:
            finished_hypos += completed_hyps
        hypos = new_hypos
        if not hypos:
            break
    
    ordered = sorted(finished_hypos, key=lambda hypo: scorer.get_model_score(hypo))
    # ordered = ordered[:beam_size]
    assert scorer.get_model_score(ordered[0]) <= scorer.get_model_score(ordered[-1])

    # for hypo in finished_hypos:
    #     if not hypo.finished:
    #         logging.info(f"Not finished: {hypo}")
    #         continue
    #     logging.info(f"\n\n {hypo}")
    #     print(tokenizer.decode(hypo.token_idxs))

    return ordered