from dataclasses import dataclass, field
from typing import List
import math
import string
from pydantic import constr
import torch
import random
import logging
import os

import argparse

from bbfs.constraint import CustomPhrasalConstraint

def gen_rand_id(N=10):
    return "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(N)
    )

class OutputMetaData:
    def __init__(self) -> None:
        pass

def read_mt_data(path="/mnt/data1/jcxu/lattice-sum/mt-data/use", name="zh-en"):
    src = name[:2]
    tgt = name[3:]
    with open(os.path.join(path, f"{name}.{src}"), "r") as fd:
        slines = fd.read().splitlines()
    with open(os.path.join(path, f"{name}.{tgt}"), "r") as fd:
        tlines = fd.read().splitlines()
    print(slines[:5])
    print(tlines[:5])
    assert len(slines) == len(tlines)
    return zip(slines, tlines)


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", default="cuda:0")
    parser.add_argument("-task", default="sum", choices=["sum", "mt1n", "mtn1"])
    parser.add_argument("-dataset", default="xsum", choices=["xsum", "default"])
    parser.add_argument("-algo", default="bs", choices=["bs", "bfs", "batch_bfs"])

    parser.add_argument("-len_factor", default=1.05, type=float)
    parser.add_argument("-max_len", type=int, default=30)
    parser.add_argument("-beam_size", type=int, default=10)
    parser.add_argument("-group_size", type=int, default=5)
    args = parser.parse_args()
    return args


def print_list(inp):
    inp_in_str = []
    for x in inp:
        if isinstance(x, float):
            inp_in_str.append("{:.4f}".format(x))
        else:
            inp_in_str.append(x)

    txt = ";".join(inp_in_str)
    print(txt)
    return txt


@dataclass
class BeamNode:
    # prob: float = field(default=0, metadata={"help": "probability of the token"})
    # token_idx: int = field(default=0)
    # prev = field(default=[])
    # prev_score = field(default=[])
    # finished: bool = field(default=False)
    # min_len: int = field(default=10, metadata={
    #                      "help": "min length of a target sequnce, used to determine the finished state."})
    # eos_idx: int = field(default=2)

    def __repr__(self) -> str:
        return self.token_strs
    def __init__(
        self,
        prob,
        glob_step,
        token_idx,
        prev,
        prev_score,
        token_str,
        batch_step,
        stop_token_idx_list,
        min_len=8
    ) -> None:

        self.prob = prob
        self.token_idx = token_idx
        self.token_str = token_str if token_str else ""
        self.prev = prev
        self.prev_score = prev_score
        self.min_len = min_len
        self.stop_token_idx_list = stop_token_idx_list
        self.uid = gen_rand_id()
        self.glob_step = glob_step
        self.batch_step = batch_step if batch_step else glob_step
        # self.depth = depth
        self.score = -math.log(self.prob)
        if self.prev:
            self.token_idxs = self.prev[0].token_idxs + [self.token_idx]
            self.token_strs = self.prev[0].token_strs + self.token_str
            self.length = self.prev[0].length + 1
            self.acc_score = self.prev[0].acc_score + self.score

            if hasattr(self.prev[0], "constraint"):
                self.constraint = self.prev[0].constraint.copy(stateful=True)
                prev_completed = self.constraint.completed
                self.stepped, _completed_status, reset = self.constraint.update(self.token_idx)
                self.completed = _completed_status or prev_completed
        else:
            # start of sequence
            self.token_idxs = []
            self.token_strs = ''
            self.length = 0
            # self.prev_score = [self.score]
            self.acc_score = self.score

        self.has_finished()

    
    def get_constraint_status(self,):
        pass
    
    def set_initial_constraint(self, forced_tokens):
        self.constraint = CustomPhrasalConstraint(forced_tokens)
        self.stepped, self.completed = False, False
        
    def get_token_idx(self) -> List[int]:
        return self.token_idxs

    def get_token_idx_as_input(self):
        tokens = self.all_token_idx
        dec_prefix = torch.tensor([tokens], dtype=torch.long)
        return dec_prefix
    def has_finished(self):
        if self.token_idx in self.stop_token_idx_list and self.length >= self.min_len:
            self.finished = True
        else:
            self.finished = False

    def visualization(self, tokenizer):
        nodes, edges = {}, {}
        seen = {}

        def dfs(node: BeamNode):
            if not node:
                return

            if node.uid in seen:
                return
            seen[node.uid] = True

            my_prev, my_prev_score = node.prev, node.prev_score
            for p, ps in zip(my_prev, my_prev_score):

                edge_info = {"src": p.uid, "tgt": node.uid, "score": ps}
                edges[f"{p.uid}_{node.uid}"] = edge_info
                # edges.append(edge_info)
            # print(node.token_idx)
            nodes[node.uid] = {
                "uid": node.uid,
                "text": tokenizer.decode(node.token_idx),
                "tok_idx": node.token_idx,
            }
            # nodes.append({'uid': node.uid,'text': node.token_str})

            prevs = node.prev
            for p in prevs:
                dfs(p)

        dfs(self)
        return nodes, edges


class Scorer:
    # determine how score is calculated
    def __init__(self, len_factor) -> None:
        self.len_factor = len_factor  # 1: average, 0: sum, -1: memoryless

    def get_model_score(self, node: BeamNode):
        if node.length == 0:
            return node.acc_score
        if self.len_factor < 0:
            return node.score
        else:
            return node.acc_score / (node.length) ** self.len_factor

    def get_model_score_from_list(self, inp: List[float]):
        return sum(inp) / (len(inp)) ** self.len_factor

    def __repr__(self) -> str:
        return f"Model score: sum(log(prob)) / length ** {self.len_factor}"


@torch.no_grad()
def run_inference_step(
    model,
    input_ids,
    attention_mask=None,
    decoder_input_ids=None,
    targets=None,
    device="cuda:0",
    output_dec_hid=False,
    T=1,
):
    decoder_input_ids = decoder_input_ids.to(device)
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if decoder_input_ids.size()[0] != input_ids.size()[0]:
        target_batch_size = decoder_input_ids.size()[0]
        batch_input_ids = input_ids.expand(target_batch_size, input_ids.size()[1])
    else:
        batch_input_ids = input_ids
    assert decoder_input_ids.size()[0] == batch_input_ids.size()[0]

    model_inputs = {
        "input_ids": batch_input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
    }

    outputs = model(
        **model_inputs,
        output_hidden_states=output_dec_hid,
        use_cache=False,
        return_dict=True,
    )

    # batch, dec seq, vocab size
    next_token_logits = outputs.logits[:, -1, :]
    if targets is not None:
        targets = targets.to(device)
        loss = torch.nn.functional.cross_entropy(
            input=next_token_logits, target=targets, reduction="none"
        )
    else:
        loss = 0

    prob = torch.nn.functional.softmax(next_token_logits / T, dim=-1)
    return prob, next_token_logits, loss


@torch.no_grad()
def run_inference_step_lm_head(
    model,
    input_ids,
    attention_mask=None,
    targets=None,
    device="cuda:0",
    output_dec_hid=False,
    T=1,
):

    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model_inputs = {
        "input_ids": input_ids,
        "encoder_attention_mask": attention_mask,
    }

    outputs = model(**model_inputs)

    # batch, dec seq, vocab size
    next_token_logits = outputs.logits[:, -1, :]
    if targets is not None:
        targets = targets.to(device)
        loss = torch.nn.functional.cross_entropy(
            input=next_token_logits, target=targets, reduction="none"
        )
    else:
        loss = 0

    prob = torch.nn.functional.softmax(next_token_logits / T, dim=-1)
    return prob, next_token_logits, loss


def obtain_ref_model_score(model, tokenizer, doc, ref, device) -> List:
    doc_input_ids = torch.tensor(
        tokenizer(doc)["input_ids"], dtype=torch.long, device=device
    ).unsqueeze(0)
    ref_ids_list = tokenizer(ref)["input_ids"][1:]
    ref_len = len(ref_ids_list)
    # ref_input_ids = torch.tensor(ref_ids_list ,dtype=torch.long,device=device)
    logging.info("Reference: {ref}")
    # print(doc_input_ids,ref_ids_list, ref_input_ids)
    dec_prefixes_id = [tokenizer.eos_token_id]
    oracle_probs = []
    for t in range(ref_len):
        target = ref_ids_list[t]
        # print(dec_prefixes_id)
        dec_input_tensors = torch.tensor(
            dec_prefixes_id, dtype=torch.long, device=device
        ).unsqueeze(0)
        prob, next_token_logits, loss = run_inference_step(
            model=model,
            input_ids=doc_input_ids,
            decoder_input_ids=dec_input_tensors,
            device=device,
        )
        oracle_prob = prob[0][target].tolist()
        oracle_probs.append(oracle_prob)
        dec_prefixes_id.append(target)
    return oracle_probs

