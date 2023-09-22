import sys
import time
import pickle
import os
import random
from datasets import load_dataset
from copyreg import pickle
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelWithLMHead
from dataclasses import dataclass, field
from typing import List
import math
import string
from dec.beam_search import beam_search
import logging
from dec.bfs import best_first_search
from dec.discriminator import Book
from dec.visual import visualize_fixed, viz_result
from dec.util import Scorer, obtain_ref_model_score, process_args




def setup_model(task='sum', dataset='xsum', model_name='facebook/bart-large-xsum', device_name='cuda:0'):
    device = torch.device(device_name)
    logging.info(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if task == 'custom':
        # you need to store the input under the path_dataset folder
        dec_prefix = [tokenizer.eos_token_id]
        with open(os.path.join(dataset, 'input.txt'), 'r') as fd:
            slines = fd.read().splitlines()
        with open(os.path.join(dataset, 'output.txt'), 'r') as fd:
            tlines = fd.read().splitlines()
        dataset = zip(slines, tlines)
    elif task == 'sum':
        logging.info('Loading dataset')
        if dataset == 'xsum':
            dataset = load_dataset("xsum", split='validation')
        elif dataset == 'cnndm':
            raise NotImplementedError("not supported")
            dataset = load_dataset("cnn_dailymail", split='validation')
            print("CNNDM mean token in ref 56")
        dec_prefix = [tokenizer.eos_token_id]
    elif task == 'mt1n':
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

        with open('examples/src.en', 'r') as fd:
            lines = fd.read().splitlines()
        dataset = [ {'document':l, 'summary':'N/A', 'id':idx} for idx,l in enumerate(lines)]
        from transformers.models.mbart.tokenization_mbart import FAIRSEQ_LANGUAGE_CODES
        # match = [x for x in FAIRSEQ_LANGUAGE_CODES if x.startswith(tgt_lang)]
        # assert len(match) == 1
        # lang = match[0]
        lang = 'zh_CN'
        logging.info(f"Lang: {lang}")
        dec_prefix = [tokenizer.eos_token_id, tokenizer.lang_code_to_id[lang]]
        logging.info(f"{tokenizer.decode(dec_prefix)}")
    elif task == 'mtn1':
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-one-mmt", )
        tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-one-mmt")
        # dataset should be like "xx-en"
        assert dataset.endswith('-en')
        src_lang = dataset[:2]
        from transformers.models.mbart.tokenization_mbart import FAIRSEQ_LANGUAGE_CODES
        match = [x for x in FAIRSEQ_LANGUAGE_CODES if x.startswith(src_lang)]
        assert len(match) == 1
        lang = match[0]
        tokenizer.src_lang = lang
        dataset = read_mt_data(name=dataset)
        dec_prefix = [tokenizer.eos_token_id,
                      tokenizer.lang_code_to_id["en_XX"]]
        logging.info(f"{tokenizer.decode(dec_prefix)}")
    model = model.to(device)
    return tokenizer, model, dataset, dec_prefix


def run_example(args, tokenizer, model, device, dec_prefix, score: Scorer, doc, ref, uid, algo='bfs'):
    doc_input_ids = torch.tensor(
        tokenizer(doc)['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
    cur_len = doc_input_ids.size()[1]
    if args.task != 'sum':  # reset max len to each example
        args.max_len = cur_len * 2
    comp_budget = args.max_len * args.beam_size

    book = Book()

    time_start = time.time()
    if algo == 'batch_bfs':

        completed, frontier = best_first_search(
            model,
            tokenizer,
            doc_input_ids,
            dec_prefix=dec_prefix,
            scorer=score,
            grp_size=args.group_size,  # batch BFS
            k_best=3,
            comp_budget=comp_budget,book=book,
        )
    elif algo == 'bfs':
        completed, frontier = best_first_search(
            model, tokenizer, doc_input_ids, dec_prefix=dec_prefix,
            scorer=score,
            grp_size=1,  # vanilla BFS
            k_best=3,
            comp_budget=comp_budget,book=book,
        )
    elif algo == 'bs':
        frontier = None
        completed = beam_search(model,
                                tokenizer,
                                doc_input_ids,
                                dec_prefix=dec_prefix,
                                scorer=score,
                                max_len=args.max_len,
                                beam_size=args.beam_size,
                                book=book,
                                eos_idx=2)
    else:
        raise NotImplementedError
    time_passed = time.time() - time_start

    oracle_probs = obtain_ref_model_score(model, tokenizer, doc, ref, device)
    reference_model_score = score.get_model_score_from_list(oracle_probs)
    search_output = {
        'completed': completed,
        'frontier': frontier,
        'time': time_passed,            
        'doc': doc,
        'ref': ref,
        'uid': uid,
        'ref_score_raw':oracle_probs,
        'ref_score': reference_model_score
    }

    return search_output, time_passed

import pickle
from viz_center import proc
def main(args) -> int:
    tokenizer, model, dataset, dec_prefix = setup_model(
        device_name=args.device, task=args.task, dataset=args.dataset)
    cnt = 0
    time = 0
    algo = args.algo

    score_module = Scorer(len_factor=args.len_factor)

    for ex in dataset:
        document = ex['document']
        summary = ex['summary']
        ex_id = ex['id']
        proc.reset()
        search_output, time_added = run_example(args, tokenizer, model, args.device, score=score_module,
                                                doc=document, ref=summary, uid=ex_id, algo=algo, dec_prefix=dec_prefix)
        name = f"{args.task}_{algo}_{args.len_factor}_{args.beam_size}_{args.group_size}_{ex_id}"

        with open(f"vizs/{name}.pkl", 'wb') as fd:
            pickle.dump(search_output, fd)
        net = viz_result(completed=search_output['completed'], tokenizer=tokenizer)
        net.show(f"vizs/{name}-ez.html")
        
        net = visualize_fixed(search_output, tokenizer)
        net.show(f"vizs/{name}-full.html")
        cnt += 1
        time += time_added
        if cnt >= 20:
            break
    print(time, args)
    return 0


if __name__ == '__main__':
    args = process_args()
    args.task = 'mt1n'
    args.beam_size = 4
    args.algo = 'bs'
    main(args)
    args.algo = 'batch_bfs'
    args.group_size = 2
    main(args)
    args.group_size = 5
    main(args)
    args.group_size = 10
    main(args)
    args.algo = 'bfs'
    main(args)
    exit()

    # args = process_args()
    args.task = 'sum'
    args.algo = 'bs'
    main(args)
    args.algo = 'batch_bfs'
    args.group_size = 2
    main(args)
    args.group_size = 5
    main(args)
    args.group_size = 10
    main(args)
    args.algo = 'bfs'
    main(args)
