import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
import string
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model

from data import Dataset, load_rhyme_info
from fudge_model import Model

from constants import *
from poetry_util import get_rhymes, count_syllables
from fudge.ref_predict_portry import predict_couplet

def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_string)
    gpt_tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    gpt_pad_id = gpt_tokenizer.encode(PAD_TOKEN)[0]
    gpt_model = AutoModelWithLMHead.from_pretrained(args.model_string).to(args.device)
    gpt_model.eval()

    checkpoint = torch.load(args.iambic_ckpt, map_location=args.device)
    model_args = checkpoint['args']
    iambic_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    iambic_model.load_state_dict(checkpoint['state_dict'])
    iambic_model = iambic_model.to(args.device)
    iambic_model.eval()
    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.iambic_ckpt, checkpoint['epoch']))


    with open(args.rhyme_info, 'rb') as rf:
        rhyme_info = pickle.load(rf)
    checkpoint = torch.load(args.rhyme_ckpt, map_location=args.device)
    model_args = checkpoint['args']
    rhyme_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word), rhyme_group_size=len(rhyme_info.index2rhyme_group), verbose=args.verbose) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    rhyme_model.load_state_dict(checkpoint['state_dict'])
    rhyme_model = rhyme_model.to(args.device)
    rhyme_model.eval()
    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.rhyme_ckpt, checkpoint['epoch']))

    
    checkpoint = torch.load(args.newline_ckpt, map_location=args.device)
    model_args = checkpoint['args']
    newline_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    newline_model.load_state_dict(checkpoint['state_dict'])
    newline_model = newline_model.to(args.device)
    newline_model.eval()
    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.newline_ckpt, checkpoint['epoch']))


    with open(args.prefix_file, 'r') as rf:
        lines = rf.readlines()
    for line in tqdm(lines, total=len(lines)):
        couplet = predict_couplet(gpt_model, 
                gpt_tokenizer, 
                iambic_model, 
                rhyme_model,
                newline_model,
                [line], 
                dataset_info, 
                rhyme_info,
                args.precondition_topk,
                args.topk, 
                condition_lambda=args.condition_lambda,
                device=args.device)
        assert len(couplet) == 2
        print(couplet[1].strip().replace('\n', ''))


if __name__=='__main__':
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
    parser.add_argument('--model_string', type=str, default='gpt2-medium')

    parser.add_argument(
        "--prefix_file",
        type=str,
        help="file of prefix lines for couplets",
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/poetry_data/couplet_prefixes.txt",
    )

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--topk', type=int, default=10, help='consider top k outputs from gpt at each step')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)