import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import defaultdict
import string
import csv

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification


from fudge.fudge_model import Model
from fudge.constants import *

def tw_topic_eval(sentences, category, tw_dir, cap=None):
    # num matches of distinct words
    words = []
    with open(os.path.join(tw_dir, category + '.txt'), 'r') as rf:
        for line in rf:
            words.append(line.strip().lower())
    num_match = 0
    for sent in sentences:
        sent_match = 0
        sent = sent.strip().lower().split()
        sent = [tok.strip(string.punctuation) for tok in sent]
        for word in words:
            if word in sent:
                sent_match += 1
        if cap is None:
            num_match += sent_match
        else:
            num_match += min(cap, sent_match)
    return num_match


def perplexity(sentences, tokenizer, model, device='cuda'):
    # calculate perplexity 
    with torch.no_grad():
        ppl = []
        sos_token = tokenizer.decode([0])
        for sentence in tqdm(sentences, total=len(sentences)):
            full_tensor_input = tokenizer.encode(sos_token + sentence.replace(EOT_TOKEN, ' ').strip(), return_tensors='pt').to(device)
            full_loss = model(full_tensor_input, labels=full_tensor_input)[0].mean()
            ppl.append(torch.exp(full_loss).flatten().cpu().item())
    return np.mean(ppl), np.std(ppl)


def grammaticality(sentences, tokenizer, model, device='cuda'):
    with torch.no_grad():
        total_good = 0
        for sent in tqdm(sentences, total=len(sentences)):
            good_prob = F.softmax(model(tokenizer.encode(sent, return_tensors='pt').to(device))[0].flatten(), dim=0)[1]
            total_good += good_prob
        return total_good / len(sentences) # avg probability of grammaticality according to model


def distinctness(results):
    d1, d2, d3 = defaultdict(lambda: set()), defaultdict(lambda: set()), defaultdict(lambda: set())
    total_words = defaultdict(lambda: 0)
    for cw, outputs in results.items():
        for o in outputs:
            o = o.replace(EOT_TOKEN, ' ').strip().split(' ')
            o = [str(x) for x in o]
            total_words[cw] += len(o)
            d1[cw].update(o)
            for i in range(len(o) - 1):
                d2[cw].add(o[i] + ' ' + o[i+1])
            for i in range(len(o) - 2):
                d3[cw].add(o[i] + ' ' + o[i+1] + ' ' + o[i+2])
    return_info = []
    avg_d1, avg_d2, avg_d3 = 0, 0, 0
    for cw in total_words.keys():
        return_info.append((cw, 'DISTINCTNESS', len(d1[cw]) / total_words[cw], len(d2[cw]) / total_words[cw], len(d3[cw]) / total_words[cw]))
        avg_d1 += len(d1[cw]) / total_words[cw]
        avg_d2 += len(d2[cw]) / total_words[cw]
        avg_d3 += len(d3[cw]) / total_words[cw]
    avg_d1, avg_d2, avg_d3 = avg_d1 / len(total_words.keys()), avg_d2 / len(total_words.keys()), avg_d3 / len(total_words.keys())
    return return_info, (avg_d1, avg_d2, avg_d3)

def eval_api(log_file, tw_dir='/export/home/experimental/naacl-2021-fudge-controlled-generation/topic_data/test_wordlists', cap_per_example=None, device='cuda:0'):
    total_rs_dict = {}

    tw_topic_match_c_total = 0
    category_totals_c = defaultdict(lambda:0)
    results = defaultdict(lambda: [])
    with open(log_file, 'r') as rf:
        data = list(csv.DictReader(rf))
        for line in data:
            results[line['category']].append(line['generation'])

    all_c_sents = []
    for category, condition_results in results.items():
        tw_topic_match_c = tw_topic_eval(condition_results, category, tw_dir, cap=cap_per_example)
        tw_topic_match_c_total += tw_topic_match_c
        category_totals_c[category] += tw_topic_match_c
        all_c_sents += condition_results

    total_rs_dict['topic_match'] = tw_topic_match_c_total
    print('Test wordlist matches (divide by num outputs to get the Success metric):', tw_topic_match_c_total)
    print('per category:', category_totals_c)

    dist_info_by_category, dist_overall = distinctness(results)
    print('Overall avg distinctness:', dist_overall)
    print('per category:', dist_info_by_category)
    total_rs_dict['distinctness_1'] = dist_overall[0]
    total_rs_dict['distinctness_2'] = dist_overall[1]
    total_rs_dict['distinctness_3'] = dist_overall[2]

    grammar_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA')
    grammar_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA').to(device)
    grammar_model.eval()
    gram_output = grammaticality(all_c_sents, grammar_tokenizer, grammar_model, device=device)
    gram_output = gram_output.tolist()
    print('grammaticality:', gram_output)
    total_rs_dict['gram'] = gram_output
    del grammar_model; del grammar_tokenizer
    
    eval_tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    eval_model = AutoModelWithLMHead.from_pretrained('openai-gpt').to(device)
    eval_model.eval()
    gpt_ppl_mean, gpt_ppl_std =  perplexity(all_c_sents, eval_tokenizer, eval_model)
    print(f'GPT perplexity:{gpt_ppl_mean} {gpt_ppl_std}')
    del eval_model; del eval_tokenizer
    total_rs_dict['gpt_ppl'] = gpt_ppl_mean
    
    eval_tokenizer = AutoTokenizer.from_pretrained('transfo-xl-wt103')
    eval_model = AutoModelWithLMHead.from_pretrained('transfo-xl-wt103').to(device)
    eval_model.eval()
    tfxl_mean, tfxl_std = perplexity(all_c_sents, eval_tokenizer, eval_model)
    print('TFXL perplexity:',tfxl_mean, tfxl_std )
    del eval_model; del eval_tokenizer
    total_rs_dict['tfxl_ppl'] = tfxl_mean
    total_rs_dict['name'] = log_file
    return total_rs_dict

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--log_file', type=str, default="topic_preds.log", help='where to load results from')
    parser.add_argument('--tw_dir', type=str, default='/export/home/experimental/naacl-2021-fudge-controlled-generation/topic_data/test_wordlists', help='test wordlists')
    parser.add_argument('--batch_size', type=int, default=20, help='max samples at a time')
    parser.add_argument('--cap_per_example', type=int, default=None, help='max matches to count per sentence')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda:0','cuda:1'])
    args = parser.parse_args()
    eval_api(args.log_file, tw_dir=args.tw_dir,cap_per_example=args.cap_per_example, device=args.device)
