import itertools
import random

from typing import List

from tqdm import tqdm
from datasets import load_dataset
from transformers import PreTrainedTokenizer

import json
import os
from dataset.cg.convert import get_test_references
from fudge.analyze_output import SearchOutput
from collections import Counter

def extract_ngram(x:List, n=2, split_key="_"):
    if len(x) < n:
        return []
    l = len(x)
    output = []
    for idx in range(l+1-n):
        span = x[idx:idx+n]
        # key = split_key.join(span)
        # output.append(key)
        output.append(tuple(span))
    return output

def set_constaint(inputs, ngram=1, split_key="_"):

    # inputs: [[1,2,3,41,2,2], [22,23,32,21], ...] # find the novel ngrams
    all_ngrams = [ extract_ngram(x,ngram) for x in inputs]
    flatten = list(itertools.chain(*all_ngrams))
    cnt = Counter(flatten)  # cnt of all ngrams
    constraints = [[] for _ in range(len(inputs))]
    for idx, it in enumerate(all_ngrams):
        # unable to extract ngram
        if not it:
            constraints[idx] = [inputs[idx][0] ]   # if we can't extract any ngram, just use the first token
            continue
        novel_ngrams = [x for x in it if cnt[x]<=1]
        if not novel_ngrams:
            constraints[idx] = list(all_ngrams[idx][0])
        else:
            constraints[idx] = list(novel_ngrams[0])
    return constraints

def mp_add_constraint(packet):
    const = set_constaint(packet['ref_tok'], ngram=2)
    packet['const'] = const
    return packet
    
def add_tokenized_ref(inps:List, tokenizer):
    for idx  in range(len(inps)):
        inp = inps[idx]
        txts = inp['ref']
        inputs = tokenizer(txts)['input_ids']
        inps[idx]['ref_tok'] = inputs
    return inps

def load_cg(split='validation', debug:bool=True, constrain=False, tokenizer=None):
    # one concept id corresponds to mulitple outputs
    print('Preprocessing data')
    if debug:
        print('DEBUG mode on. Truncate to 50.')
        dataset = load_dataset('common_gen',split=f"{split}[40%:60%]")
    else:
        dataset = load_dataset('common_gen', split=split)
        # dataset = full_dataset[split]
    processed_data = {}
    for data in tqdm(dataset):
        concept_set_idx = data['concept_set_idx']
        concepts = data['concepts']
        target = data['target']
        target = target.strip()
        if concept_set_idx in processed_data:
            processed_data[concept_set_idx]['ref'].append(target)
        else:
            processed_data[concept_set_idx] = {
                'uid':concept_set_idx,
                'concepts':concepts,
                'input':" ".join(concepts),
                'ref':[target]
            }
    print('Complete preprocessing Phase I.')
    # for test data, add test manually
    if split == 'test':
        test_refs,constrain_match_dict = get_test_references()
        for k, v in processed_data.items():
            processed_data[k]['ref'] = test_refs[k]
            processed_data[k]['match'] = constrain_match_dict[k]
            if random.random() > 0.99:
                print(v['concepts'], v['ref'])
    processed_data = list(processed_data.values())
    print(processed_data[2])
    if constrain:
        processed_data = add_tokenized_ref(processed_data, tokenizer)
        ex_output = mp_add_constraint(processed_data[0])
        print('Running example')
        print(processed_data[0])
        print(ex_output)
        import multiprocessing as mp
        pool = mp.Pool(processes=8)
        results = pool.map(mp_add_constraint, processed_data)
        processed_data = results
        print('Done adding constraints.')
    return processed_data

def prepare_output_cg(data_pack, search_outputs: SearchOutput)->List[dict]:
    if 'const' in data_pack:
        del data_pack['const']
    if 'ref_tok' in data_pack:
        del data_pack['ref_tok']
    data_pack['output'] = search_outputs.all_hypos
    return data_pack

def write_to_json(filename, new_entry):
    
    if not os.path.exists(filename):
        with open(filename, "w") as file:
            
            json.dump([], file)
    # 1. Read file contents
    with open(filename, "r") as file:
        data = json.load(file)
    # 2. Update json object
    data.append(new_entry)
    # 3. Write json file
    with open(filename, "w") as file:
        json.dump(data, file)