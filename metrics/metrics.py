import statistics
import random
import torch
import mauve 
import os
import csv
import itertools
from typing import List
from collections import defaultdict
from multiprocessing import Pool
from datasets import load_metric
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification
from random import sample
from math import pow
import evaluate
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from evaluate import load
import editdistance
# mauve = load_metric('mauve')
rouge = load_metric('rouge')
meteor = load_metric( 'meteor')
# bleu = load_metric('bleu')

# bertscore = load("bertscore")
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'], use_stemmer=True)

# LOAD models
device_id = input("Assign device ID")
device_id = int(device_id)
device = f"cuda:{device_id}"
grammar_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA')
grammar_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA').to(device)
grammar_model.eval()

gpt_tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
gpt_model = AutoModelWithLMHead.from_pretrained('openai-gpt').to(device)
gpt_model.eval()

# tfxl_tokenizer = AutoTokenizer.from_pretrained('transfo-xl-wt103')
# tfxl_model = AutoModelWithLMHead.from_pretrained('transfo-xl-wt103').to(device)
# tfxl_model.eval()

def back_fill_xsum(raw_pack):
    pre = raw_pack['output']
    
    if pre[0]:
        return raw_pack['output']
    context = raw_pack["context"]
    sents = context.split('\n')[:3]
    return sents

def run_ed():
    distance = editdistance.eval('banana', 'bahama')
    pass
def read_lines(fname):
    with open(fname,'r') as fd:
        lines  = fd.read().splitlines()
    return lines

def run_meteor(pred_lines, gt_lines):
    assert len(pred_lines) == len(gt_lines)
    results = meteor.compute(predictions=pred_lines, references=gt_lines)
    return results["meteor"]

def run_bertscore_all(predictions,references):
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    return results['f1']


def run_metric(met, pred_lines, gt_lines):
    met.add(predictions=pred_lines, reference=gt_lines)
    # for p,g in zip(pred_lines, gt_lines):
        
    score = met.compute()

    return score

def distinctness(outputs, EOT_TOKEN='.'):
    cw = "default"
    d1, d2, d3 = defaultdict(lambda: set()), defaultdict(lambda: set()), defaultdict(lambda: set())
    total_words = defaultdict(lambda: 0)

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
    return  (avg_d1, avg_d2, avg_d3)

def read_json(fname):
    with open(fname,'r') as fd:
        import json
        data = json.load(fd)
    return data


def perplexity(sentences, tokenizer, model, device='cuda:2'):
    # calculate perplexity 
    EOT_TOKEN = '</s>'
    with torch.no_grad():
        ppl = []
        sos_token = tokenizer.decode([0])
        for sentence in tqdm(sentences, total=len(sentences)):
            full_tensor_input = tokenizer.encode(sos_token + sentence.replace(EOT_TOKEN, ' ').strip(), return_tensors='pt').to(device)
            full_loss = model(full_tensor_input, labels=full_tensor_input)[0].mean()
            ppl.append(torch.exp(full_loss).flatten().cpu().item())
    return np.mean(ppl), np.std(ppl)

def gector():
    command = f"sh src/recom_search/evaluation/bash_gector.sh {model_repo} {input_file} {outpuf_file} {outpuf_cnt_file}"

def run_bleu(predictions, references):
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    return results['bleu']

def call_rouge(pred, ref):
    scores = scorer.score(pred, ref)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def call_score_multi(preds, refs):
    rouge_scores = defaultdict(list)
    for p,r in zip(preds, refs):
        r1,r2,rl = call_rouge(p,r)
        rouge_scores['r1'].append(r1)
        rouge_scores['r2'].append(r2)
        rouge_scores['rl'].append(rl)
    return statistics.mean(rouge_scores['r1']), statistics.mean(rouge_scores['r2']), statistics.mean(rouge_scores['rl'])


def grammaticality(sentences, tokenizer, model, device='cuda:2'):
    with torch.no_grad():
        total_good = 0
        for sent in tqdm(sentences, total=len(sentences)):
            enc = tokenizer.encode(sent, return_tensors='pt')
            calib_len = len(enc)
            good_prob = F.softmax(model(enc.to(device))[0].flatten(), dim=0)[1]
            total_good += pow(good_prob, 1/calib_len)
        return total_good / len(sentences) # avg probability of grammaticality according to model

def run_gpt_ppl(all_c_sents,device='cuda:2'):

    gpt_ppl_mean, gpt_ppl_std =  perplexity(all_c_sents, gpt_tokenizer, gpt_model,device=device)
    print(f'GPT perplexity:{gpt_ppl_mean} {gpt_ppl_std}')
    return gpt_ppl_mean

def run_tfppl(all_c_sents,device='cuda:2'):

    tfxl_mean, tfxl_std = perplexity(all_c_sents, tfxl_tokenizer, tfxl_model,device=device)
    print('TFXL perplexity:',tfxl_mean, tfxl_std )
    return tfxl_mean

def run_gramma(sentences,device='cuda:2'):
    gram_output = grammaticality(sentences, grammar_tokenizer, grammar_model, device=device)
    gram_output = float(gram_output)
    print('grammaticality:', gram_output)
    return gram_output

def get_oracle(preds:List[List], ref:List):
    assert len(preds) == len(ref)
    result = defaultdict(list)
    for pred, r in zip(preds, ref):
        # print(pred,r)
        outputs = [call_rouge(p, r) for p in pred]
        ourput_r1 = [x[0] for x in outputs]
        ourput_r2 = [x[1] for x in outputs]
        ourput_rl = [x[2] for x in outputs]
        result['oracle_r1'].append(max(ourput_r1))
        result['oracle_r2'].append(max(ourput_r2))
        result['oracle_rl'].append(max(ourput_rl))
    return result


def evaluate_pair(prediction, reference, prefix, lang='en',task='xsum-bart'):
    summary = {}

    # prediction can be List[str], or List[List[str]]
    assert len(prediction) == len(reference)
    if not prediction:
        raise NotImplementedError("Missing data")
    if isinstance(prediction[0], List) :
        flatten_prediction = list(itertools.chain(*prediction))
        flatten_ref =  list(itertools.chain(*reference))
        assert len(flatten_prediction) == len(flatten_ref)
        prediction = flatten_prediction
        reference = flatten_ref

    # ROUGE
    # rouge_results = rouge.compute(predictions=prediction,references=reference)
    # r1,r2,rl = rouge_results["rouge1"].mid.fmeasure, rouge_results["rouge2"].mid.fmeasure, rouge_results["rougeL"].mid.fmeasure
    r1,r2,rl = call_score_multi(prediction, reference)
    summary[f"r1"] = r1
    summary[f"r2"] = r2
    summary[f"rl"] = rl
    if lang == 'en':
        
        total_len = len(prediction)
        if total_len > 2000:
            sel_index = sample(range(total_len), 2000)
            print(sel_index[:30])
            prediction = [prediction[x] for x in sel_index]
            reference = [reference[x] for x in sel_index]
        try:
            assert len(prediction) == len(reference) > 100
        except:
            raise AssertionError
            exit()
        print(prediction[:5])
        
        # filter
        new_pred, new_ref = [], []
        for p,r in zip(prediction, reference):
            if p and r:
                new_pred.append(p)
                new_ref.append(r)
        prediction = new_pred
        reference = new_ref
        
        # call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to 256 tokens
        if task not in ['']:
            mauve_results = mauve.compute_mauve(p_text=prediction, q_text=reference, device_id=device_id, max_text_length=40, verbose=True, batch_size=16, kmeans_max_iter=300, kmeans_num_redo=3).mauve
            summary['mauve'] = mauve_results
            
            gram_score = run_gramma(prediction, device)
            summary['gram'] = gram_score
            # summary['gpt'] = run_gpt_ppl(prediction, device)
        # summary['tfxl'] = run_tfppl(prediction, device)
    
    summary['meteor'] = run_meteor(prediction, reference)
    summary['bleu'] = run_bleu(prediction, reference)
    # update keys
    new_dict = {}
    for k, v in summary.items():
        new_dict[prefix+"_"+k] = v
    return new_dict

def cg_match_test(keywords, candidates):
    rate = []
    for cand in candidates:
        # cand is a sentence
        
        cnt = 0
        for key in keywords:
            
            for k in key:
                if k in cand:
                    cnt += 1
                    break
            
        ratio = cnt / len(keywords)
        rate.append(ratio)
    return rate

def sort_outputs(output, scores):
    score_w_index = [ (idx, score) for idx, score in enumerate(scores)]
    sorted_scores = sorted(score_w_index, key=lambda x: x[1], reverse=True)
    sorted_outputs = [ output[ss[0]] for ss in sorted_scores]
    return sorted_outputs
    
def run_one_to_one(data, fname, task, debug=False, tgt_num=10,  lang='en'):
    # ref == pred == 1
    dg = data_generator(data,debug)
    result = defaultdict(list)
    summary = {'name':fname}
    pred_11, pred_11_sp,  pred_1n_sp, ref_11, ref_n = [], [], [], [], []
    for instance in dg:
        ref, pred, raw_pack = instance
        if task in ['rank_enfr','rank_ende']:
            new_pred = sort_outputs(raw_pack['output'], raw_pack['rank'])
            pred = new_pred
        if task.startswith('xsum'):
            pred = back_fill_xsum(raw_pack)
            # remove those not ending with .!?
            pred = [sel for sel in pred if sel.endswith('.') or sel.endswith('!') or sel.endswith('?')]
            if not pred:
                pred = [""]
        pred = [ o.strip() for o in pred]
        ref = [ o.strip() for o in ref]

        result['num'].append(len(pred))
        
        uniq_pred = len(set(pred))
        result['uniq_num'].append(uniq_pred)
        result['null'].append(int(bool(pred[0])))
        if len(ref) > 1:
            tgt_num = len(ref)
        ref_11.append(random.choices(ref, k=1)[0])
        # ref_11.append(ref[0])                           # top 1 from ref, applied for cases len(ref) = 1
        if len(ref) != tgt_num:
            ref = random.choices(ref, k=tgt_num)
        ref_n.append(ref)
        pred_11.append(pred[0])                         # default top 1 from pred
        pred_11_sp.append(random.choices(pred, k=1)[0]) # sample 1 from pred
        pred_1n_sp.append(random.choices(pred, k=tgt_num))  # sample tgtnum from pred

        # compute in group diversity
        # SAMPLE, if output is more than target_num
        samples = random.choices(pred, k=tgt_num) # will upsample if needed
        avg_d1, avg_d2, avg_d3 = distinctness(samples)
        for i in range(tgt_num):
            result['distinct_1'].append(avg_d1)
            result['distinct_2'].append(avg_d2)
            result['distinct_3'].append(avg_d3)
        # compute in group diversity of reference
        avg_d1, avg_d2, avg_d3 = distinctness(ref)
        for i in range(tgt_num):
            result['ref_distinct_1'].append(avg_d1)
            result['ref_distinct_2'].append(avg_d2)
            result['ref_distinct_3'].append(avg_d3)
        if task == 'cg':
            matches = raw_pack['match']
            sp_outputs = random.choices(pred, k=tgt_num)
            rates  = cg_match_test(matches, sp_outputs)
            result['match'] += rates
        if task == 'gec':
            # rerank outputs with edit distance.
            pass
    if task in ['squad','drop','quoref', 'xsum-bart','cg','xsum-bart-regular','enfr','ende']:
        oracle_summary = get_oracle(pred_1n_sp, ref_11)
        result = {**result, **oracle_summary}
    
    for k,v in result.items():
        avg = statistics.mean(v)
        summary[k] = avg
    # rouge_res = run_metric(rouge, pred_lines, ref_lines)
    # bleu_res = run_metric(bleu, pred_lines, ref_lines)
    rt_top1, rt_sp1, rt_nn = {}, {}, {}
    if task in ['cg', 'enfr','ende']:
        rt_nn = evaluate_pair(pred_1n_sp,ref_n,prefix='n_n', lang=lang, task=task)
    if task in [ 'rank_enfr', 'rank_ende']:
        rt_top1 = evaluate_pair(pred_11, ref_11, prefix='top1_1', lang=lang,task=task)
        rt_sp1 = evaluate_pair(pred_11_sp, ref_11, prefix='sp1_1', lang=lang,task=task)
    if task in ['xsum-bart','squad','drop', 'quoref','xsum-bart-regular']:
        rt_sp1 = evaluate_pair(pred_11_sp, ref_11, prefix='sp1_1', lang=lang,task=task)
        
    summary = {**summary, **rt_top1, **rt_sp1, **rt_nn}
    return summary


def data_generator(data, debug=False):
    cnt = 0 
    for d in data:
        yield (d['ref'], d['output'], d)
        cnt += 1
        if debug and cnt >=20:
            break
import time
def work(x):
    result_board, task, f, tgt_num, lang, dir = x
    file_exists = os.path.exists(result_board)

    # check existing
    if  file_exists:
        with open(result_board, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                exist = row['name']
                if exist == f:
                    print(f"Found existing {f}")
                    return
    
    fname = os.path.join(dir, f)
    print(fname)
    data = read_json(fname)
    result = run_one_to_one(data, f, task=task, tgt_num=tgt_num, lang=lang)
    file_exists = os.path.exists(result_board)
    if  file_exists:
        with open(result_board, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                exist = row['name']
                if exist == f:
                    print(f"Found existing {f}")
                    return
    if not file_exists:
        with open(result_board, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=result.keys())
            writer.writeheader()
    with open(result_board, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result.keys())
        writer.writerow(result)


def multi_run(files,result_board, task, tgt_num, lang, dir, parallel=False):
    # prepare
    pre_input = [ (result_board, task, f, tgt_num, lang, dir) for f in files]
    print(pre_input)
    
    if not parallel:
        for pre_in in pre_input:
            print(pre_in)
            work(pre_in)
    else:
        with Pool(processes=4) as pool:
            # print same numbers in arbitrary order
            for i in pool.imap_unordered(work, pre_input):
                pass
    
def main():
    prefix = input("name of the task, like cg, squad, drop, rank_opus, etc.")  # Python 3

    task = prefix
    tgt_num = input("target number. if there is a floating number of reference, use -1 (cg) or 8 for MT. ")
    tgt_num = int(tgt_num)

    print(prefix, tgt_num)
    dir = '/export/home/cond-text-gen/outputs'
    dir  = os.path.join(dir, prefix)
    files = os.listdir(dir)

    files = [ f for f in files if f.endswith('.json')]
    # files = [ f for f in files if f.endswith('.json') and f.startswith("")]
    # files = ['cg_sample_25_10_2_0.0_5_0.2_1.0_1_0.0_output.json']
    if prefix in ['rank_opus','op','mt','enfr','ende','rank_enfr','rank_ende']:
        lang = 'xx'
        print(lang)
    else:
        lang = 'en'
    # files = [os.path.join(dir, f) for f in files if f.endswith('.json')]
    print(files)
    random.shuffle(files)
    result_board = f"{prefix}_results_all.csv"
    
    use_ref = False
    
    multi_run(files,result_board, task, tgt_num, lang,dir)


if __name__ == '__main__':
    main()

# run_metric(rouge)
# run_metric(meteor)
# run_metric(bleu)
# evaluation
# reference: 1 to n
# pred: 1 to m, 
# sample. oracle. BLEU/ROUGE
# repetition novel ngram