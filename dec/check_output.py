import statistics
import random
from collections import defaultdict

import pickle
import os

from transformers import AutoTokenizer

from dec.util import Scorer, print_list

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score('The quick brown fox jumps over the lazy dog',
                      'The quick brown dog jumps on the log.')
print(scores['rouge1'].fmeasure)

p = os.path.join(os.getcwd(), 'vizs')
files = os.listdir(p)
files = [f for f in files if f.endswith('pkl')]
score_func = Scorer(1.05)
model_name = 'facebook/bart-large-xsum'
tokenizer = AutoTokenizer.from_pretrained(model_name)


def extract_hypo(hypos, calib_len):
    d = defaultdict(list)
    for hyp in hypos:
        tokens = []
        scores = []
        node = hyp
        while node.prev:
            tokens.append(node.token_idx)
            scores.append(node.score)
            node = node.prev[0]
        # reverse
        rev_toks = tokens[::-1]
        rev_scores = scores[::-1]
        original_score = score_func.get_model_score_from_list(rev_scores)
        calib_score = score_func.get_model_score_from_list(
            rev_scores[:calib_len])
        origin_text = tokenizer.decode(rev_toks)
        caliv_text = tokenizer.decode(rev_toks[:calib_len])
        d['score'].append(original_score)
        d['score_calib'].append(calib_score)
        d['text'].append(origin_text)
        d['text_caib'].append(caliv_text)
    return d


def self_rouge(group):
    random.shuffle(group)
    if len(group) > 10:
        group = group[:10]
    board = []
    l = len(group)
    for idx in range(l):
        for jdx in range(idx+1, l):
            scores = scorer.score(group[idx], group[jdx])
            board.append(scores['rouge1'].fmeasure)
    return statistics.mean(board)


def examine(sel_files):
    num_hypos = []
    avg_model_score = []
    avg_model_score_ref_len = []
    avg_self_rouge = []
    avg_rouge = []
    best_rouge = []

    for f in sel_files:
        with open(os.path.join(p, f), 'rb') as fd:
            data = pickle.load(fd)

        reference_summary = data['ref']
        hypos = data['completed']
        num_hypos.append(len(hypos))

        ref_len = len(data['ref_score_raw'])
        output_dict = extract_hypo(hypos, ref_len)
        avg_self_rouge.append(self_rouge(output_dict['text']))
        avg_model_score.append(statistics.mean(output_dict['score']))
        avg_model_score_ref_len.append(
            statistics.mean(output_dict['score_calib']))
        rouge_scores = [scorer.score(reference_summary, x)[
            'rouge1'].fmeasure for x in output_dict['text']]
        avg_rouge.append(statistics.mean(rouge_scores))
        best_rouge.append(max(rouge_scores))
    final = {
        'num_hypos': num_hypos,
        'avg_model_score': avg_model_score,
        'avg_model_score_ref_len':   avg_model_score_ref_len,
        'avg_self_rouge': avg_self_rouge,
        'avg_rouge': avg_rouge,
        'best_rouge': best_rouge,
    }
    for k in final.keys():
        final[k] = statistics.mean(final[k])
    return final

prefix_keys = defaultdict(list)
for f in files:
    prefix = "_".join(f.split('_')[:-1])
    prefix_keys[prefix].append(f)

for k in prefix_keys.keys():
    selected_files = [x for x in files if x.startswith(k)]
    out = examine(selected_files)
    print(k)
    
    print_list(list(out.values()) )
print_list(list(out.keys()))

