"""
best-first search
"""

from argparse import ArgumentParser
import torch
import math
from typing import Dict, List, Optional
import logging
import heapq
import random
import torch.nn.functional as F
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from dec.bfs import assemble_pad, assemble_pad_plus, init_search, vanilla_heap_pop
from dec.viz_center import proc
from dec.util import BeamNode, Scorer, run_inference_step, run_inference_step_lm_head

import numpy as np
import pickle
from tqdm import tqdm
import string
from collections import defaultdict


# Best first search with k-expansion


from fudge.constants import POETRY_BANNED_TOKENS, UNKNOWN_RHYME_GROUP
from fudge.run_poetry import predict_couplet, setup_poetry_models
from fudge.score_poem import *
from fudge.util import gpt_decode, setup_gpt




# one step of best first search. expand k hypothesis. the expansion strategy depends.
def fudge_step_bfs(
    model,
    tokenizer,
    doc_input_token,
    heap,
    scorer,
    scorer_rhyme,
    scorer_newline,
    scorer_iambic,
    max_len,
    grp_size,
    k_best,
    condition_lambda,
    precondition_topk,
    device,
    banned_tokens=POETRY_BANNED_TOKENS,
    book=None,
    verbose=True,
):
    # obtain a list of top nodes from heap.
    # the size of heap <= grp_size
    expansion_list = []
    finished_hypos = []
    cnt = 0

    while cnt < grp_size and heap:
        seed: BeamNode = vanilla_heap_pop(heap)
        expansion_list.append(seed)
        cnt += 1

    doc_input_token_len = len(doc_input_token)

    ##
    # dec_prefixes = [x.get_token_idx() for x in expansion_list]
    # concat_prefixes = [doc_input_token + x for x in dec_prefixes]

    # (
    #     concat_input_tensor,
    #     padding_pos,
    #     attention_mask_concat_input_tensor,
    # ) = assemble_pad_plus(
    #     concat_prefixes, device=device, pad_token_idx=tokenizer.bos_token_id
    # )  # batch x seq      store [PAD] * ? + enc + dec
    # batch_size, full_length = concat_input_tensor.size()
    # # enc + dec length, exclude padding
    # enc_dec_length = [full_length - x for x in padding_pos]
    # # dec only length, exclude padding
    # dec_length = [x - doc_input_token_len for x in enc_dec_length]

    # dec_input_tensor = assemble_pad(
    #     dec_prefixes, device, tokenizer.bos_token_id
    # )  # only store the decoded tokens after enc, batch x dec_seq_len

    # output = model(
    #     concat_input_tensor, attention_mask=attention_mask_concat_input_tensor
    # )
    # logits = output.logits[:, -1, :]  # batch x seq_len x vocab =>  batch x vocab
    ###
    (
        logits,
        concat_input_tensor,
        dec_input_tensor,
        batch_size,
        padding_pos,
        enc_dec_length,
    ) = gpt_decode(model, device, expansion_list, doc_input_token, tokenizer)

    logits[:, banned_tokens] = -1e8  # TODO do we need?

    values, indices = torch.topk(logits, k=precondition_topk, dim=-1)  # batch x pretopk

    expanded_concat_input = concat_input_tensor.unsqueeze(1).expand(
        -1, precondition_topk, -1
    )  # batch x pretopk x e+d_seq
    expanded_dec_input = dec_input_tensor.unsqueeze(1).expand(
        -1, precondition_topk, -1
    )  # batch, prek, seq
    expanded_top_indicies = indices.unsqueeze(-1)
    extended_concat_input_candidates = torch.cat(
        [expanded_concat_input, expanded_top_indicies], dim=2
    )  # batch  x pretopk  x seq+1
    extended_dec_candidates = torch.cat(
        [expanded_dec_input, expanded_top_indicies], dim=2
    )  # batch
    extended_concat_input_candidates_list = (
        extended_concat_input_candidates.cpu().tolist()
    )

    # expanded_dec_only

    lengths = torch.LongTensor(enc_dec_length).to(device)
    expanded_lengths = (
        (lengths + 1).unsqueeze(1).expand(batch_size, precondition_topk)
    )  # expandlen is original + 1
    expanded_future_words = (
        scorer_rhyme.future_words.unsqueeze(0)
        .unsqueeze(1)
        .expand(batch_size, precondition_topk, -1)
    )  # batch x topk x N

    # update syllabus for all batch x precondition k
    candidate_syllables_to_go = []
    for b in range(batch_size):
        cur_batch_dec_seq = extended_concat_input_candidates_list[b]  # pretopk x seq+1
        for k in range(precondition_topk):  # TODO do we really need this?
            seqs = cur_batch_dec_seq[k]
            generated_words = seqs[doc_input_token_len + padding_pos[b] :]
            candidate_until_last_word_text = " ".join(
                tokenizer.decode(generated_words).split()[:-1]
            )
            candidate_syllables_to_go.append(
                10 - count_syllables(candidate_until_last_word_text)
            )
    expanded_syllables_to_go = (
        torch.LongTensor(candidate_syllables_to_go)
        .to(device)
        .view(batch_size, precondition_topk)
    )  # batch, preconditionk

    iambic_logits = scorer_iambic.score(
        extended_dec_candidates, expanded_lengths, doc_input_token_len
    )

    rhyme_logits = scorer_rhyme.score(
        extended_concat_input_candidates,
        expanded_future_words,
        expanded_lengths,
        expanded_syllables_to_go,
        batch_size,
        precondition_topk,
    )

    newline_logits = scorer_newline.score(
        extended_concat_input_candidates,
        expanded_future_words,
        expanded_lengths,
        expanded_syllables_to_go,
        batch_size,
        precondition_topk,
        scorer_rhyme.log_probs,
    )

    full_logits = (
        values
        + condition_lambda * iambic_logits
        + condition_lambda * rhyme_logits
        + condition_lambda * newline_logits
    )

    post_logits, post_indices = full_logits.topk(k_best, dim=1)
    post_probs = F.softmax(post_logits, dim=1)

    index_into_top_indices = post_indices[
        torch.arange(batch_size).to(post_indices.device), :k_best
    ]  # batch, k_best
    # logging.info(index_into_top_indices.size())

    for b in range(batch_size):
        next_indices = (
            indices[b][index_into_top_indices[b]].cpu().tolist()
        )  # k best options
        cur_seq = concat_input_tensor[b].cpu().tolist()  # seq_len

        if verbose and random.random() > 0.95:

            logging.info(
                f"Prefix: {tokenizer.decode(doc_input_token)}\t"
                + f"///\t{tokenizer.decode(dec_prefixes[b])}"
            )
            logging.info(
                f"Original top 5:"
                + " ".join([tokenizer.decode(x) for x in indices[b][:5]])
            )
            visualize_logits(iambic_logits, indices, "iambic", tokenizer, b, 5)
            visualize_logits(rhyme_logits, indices, "rhyme", tokenizer, b, 5)
            visualize_logits(newline_logits, indices, "newline", tokenizer, b, 5)
            visualize_logits(full_logits, indices, "full", tokenizer, b, 5)

        for rank in range(k_best):
            v = post_probs[b][rank]
            tok = next_indices[rank]
            tmp_state = BeamNode(
                prob=v,
                token_idx=tok,
                prev=[expansion_list[b]],
                prev_score=[math.log(v)],
                glob_step=proc(),
            )
            # book.add_child()

            # ending state
            generated_so_far = tokenizer.decode(
                cur_seq[doc_input_token_len + padding_pos[b] :] + [next_indices[rank]]
            )
            syllables_to_go = POETRY_LINE_SYLLABLES - count_syllables(
                generated_so_far
            )  # if we get very unlucky with a partial word that the syllable counter doesn't recognize we might end early, but it's unlikely
            if syllables_to_go <= 0 and tokenizer.decode(tok)[-1] in PHRASE_ENDS:
                logging.info(
                    f"Ending: {tokenizer.decode(cur_seq[doc_input_token_len + padding_pos[b]:] + [next_indices[rank]])}"
                )
                finished_hypos.append(tmp_state)
                continue
            if syllables_to_go < 0:
                # a bad hypo?
                # encoded_input = encoded_input[:, :-1]
                continue

            # if tmp_state.finished:  # if this branch is a completed one, just put it in the outputs.
            #     finished_hypos.append(tmp_state)
            #     continue
            # if tmp_state.length >= max_len:
            #     continue

            model_score = scorer.get_model_score(tmp_state)  # stateless?
            # model_score = math.log(v)
            if random.random() < 0.01:
                logging.info(f"Score: {model_score}")

            heapq.heappush(heap, (-model_score, tmp_state))
    return finished_hypos, cnt


def fudge_best_first_search(
    model,
    tokenizer,
    scorer,
    scorer_rhyme,
    scorer_newline,
    scorer_iambic,
    doc_input,
    dec_prefix="",
    dataset_info=None,
    condition_lambda: float = 1.0,
    precondition_topk: int = 200,
    k_best: Optional[int] = 15,
    max_len: Optional[int] = 30,
    grp_size: Optional[int] = 5,
    comp_budget: Optional[int] = 300,
    device=None,
    book=None,
    eos_idx: int = 50256,
):

    ncalls = 0

    heap = []  # nodes at the frontier of search
    finished_hypos = []

    # combined_input = doc_input + dec_prefix

    tok_input: List[int] = tokenizer.encode(doc_input)
    tok_dec_prefix: List[int] = tokenizer.encode(dec_prefix)

    init_search(tok_dec_prefix, heap, eos_idx=50256)

    # init_seed only has the decoding prefix

    while ncalls < comp_budget and heap:
        completed_hyps, added_num_calls = fudge_step_bfs(
            model=model,
            tokenizer=tokenizer,
            doc_input_token=tok_input,
            heap=heap,
            scorer=scorer,
            scorer_rhyme=scorer_rhyme,
            scorer_newline=scorer_newline,
            scorer_iambic=scorer_iambic,
            max_len=max_len,
            grp_size=grp_size,
            k_best=k_best,
            condition_lambda=condition_lambda,
            precondition_topk=precondition_topk,
            device=device,
        )

        ncalls += added_num_calls

        if completed_hyps:
            finished_hypos += completed_hyps

    return finished_hypos, heap


def main(args):
    with open(args.dataset_info, "rb") as rf:
        dataset_info = pickle.load(rf)
    with open(args.rhyme_info, "rb") as rf:
        rhyme_info = pickle.load(rf)
    if args.debug:
        args.model_string = "distilgpt2"
        logging.info(f"Change model to distilgpt2 ...")

    model, tokenizer, gpt_pad_id = setup_gpt(args.model_string, args.device)

    iambic_model, rhyme_model, newline_model = setup_poetry_models(
        args.iambic_ckpt,
        args.rhyme_ckpt,
        args.newline_ckpt,
        gpt_pad_id,
        dataset_info,
        rhyme_info,
        args.device,
        verbose=True,
    )
    scorer_newline = NewLine(newline_model, args.device)
    scorer_rhyme = Rhyme(rhyme_model, device=args.device, rhyme_info=rhyme_info)
    scorer_iambic = Iambic(iambic_model, args.device)

    score_module = Scorer(len_factor=args.len_factor)

    with open(args.prefix_file, "r") as rf:
        lines = rf.readlines()
    with open(args.pred_file, "w") as wf:
        pass
    predictions = []
    for line in tqdm(lines, total=len(lines)):
        current_text = line
        current_line_text = ""
        all_lines = [current_text]
        logging.info(f"\n {current_text}")
        dec_prefix = tokenizer.encode(current_text)
        comp_budget = len(dec_prefix) * args.beam_size
        logging.info(f"Comp budget: {comp_budget}")

        scorer_rhyme.add_rhyme_group(current_text)

        finished_hypos, heap = fudge_best_first_search(
            model=model,
            tokenizer=tokenizer,
            scorer=score_module,
            scorer_rhyme=scorer_rhyme,
            scorer_newline=scorer_newline,
            scorer_iambic=scorer_iambic,
            doc_input=current_text,
            dec_prefix=current_line_text,
            dataset_info=dataset_info,
            condition_lambda=args.condition_lambda,
            precondition_topk=args.precondition_topk,
            k_best=args.topk,
            max_len=args.max_len,
            grp_size=args.group_size,
            comp_budget=comp_budget,
            device=args.device,
            eos_idx=tokenizer.eos_token_id,
        )
        # only use the top1 for now
        assert len(finished_hypos) > 0
        top1_hypo_tok_id = finished_hypos[0].get_token_idx()
        print("--")
        for hypo in finished_hypos:
            output_string = tokenizer.decode(hypo.get_token_idx())
            print(output_string.strip().replace("\n", ""))
        print("--")
        output_string = tokenizer.decode(top1_hypo_tok_id)
        # assert len(couplet) == 2
        # logging.info(couplet[1].strip().replace('\n', ''))
        output_str_post = output_string.strip().replace("\n", "")
        predictions.append(output_str_post)
        with open(args.pred_file, "a") as wf:
            wf.write(output_str_post + "\n")
        exit()


if __name__ == "__main__":
    parser = ArgumentParser()

    # DATA
    parser.add_argument(
        "--iambic_ckpt",
        type=str,
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/iambic_predictor/model.pth.tar",
    )
    parser.add_argument(
        "--rhyme_ckpt",
        type=str,
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/rhyme_predictor/model.pth.tar",
    )
    parser.add_argument(
        "--newline_ckpt",
        type=str,
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/newline_predictor/model.pth.tar",
    )
    parser.add_argument(
        "--dataset_info",
        type=str,
        help="saved dataset info",
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/rhyme_predictor/dataset_info",
    )
    parser.add_argument(
        "--rhyme_info",
        type=str,
        help="saved rhyme info",
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/rhyme_predictor/rhyme_info",
    )
    parser.add_argument("--model_string", type=str, default="gpt2-medium")

    parser.add_argument(
        "--prefix_file",
        type=str,
        help="file of prefix lines for couplets",
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/poetry_data/couplet_prefixes.txt",
    )

    parser.add_argument(
        "--precondition_topk",
        type=int,
        default=200,
        help="consider top k outputs from gpt at each step before conditioning and re-pruning",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="consider top k outputs from gpt at each step",
    )
    parser.add_argument(
        "--condition_lambda",
        type=float,
        default=1.0,
        help="lambda weight on conditioning model",
    )

    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "-device", type=str, default="cuda:0", choices=["cpu", "cuda:0", "cuda:1"]
    )
    parser.add_argument("--debug", action="store_true", default=False)

    # parser.add_argument('-task', default='sum', choices=['sum','mt1n','mtn1'])
    # parser.add_argument('-dataset', default='xsum', choices=['xsum','default'])
    parser.add_argument("-algo", default="bs", choices=["bs", "bfs", "batch_bfs"])

    parser.add_argument("-len_factor", default=1.4, type=float)
    parser.add_argument("-max_len", type=int, default=30)
    parser.add_argument("-beam_size", type=int, default=10)
    parser.add_argument("-group_size", type=int, default=2)
    parser.add_argument("-pred_file", default="my_pred.txt")
    # /export/home/experimental/naacl-2021-fudge-controlled-generation/poetry_data/couplet_prefixes.txt

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
