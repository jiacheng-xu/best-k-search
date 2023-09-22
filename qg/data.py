from datasets import load_dataset

from tqdm import tqdm
from typing import List
import multiprocessing as mp
from cg.data import add_tokenized_ref, mp_add_constraint
import random
def load_quoref( split='validation', debug:bool=True,  constrain=False, tokenizer=None):
    # one concept id corresponds to mulitple outputs
    print('Preprocessing data')
    
    if debug:
        print('DEBUG mode on. Truncate to 50.')
        dataset = load_dataset('quoref',split=f"{split}[50%:60%]")
    else:
        dataset = load_dataset('quoref', split=split)

    processed_data = {}
    for idx, data in enumerate(dataset):
        answer = data["answers"]["text"][0]
        context = data['context'][:1500]
        q = data['question']
        input = f"{answer} \\n {context}"
        key = input[:20]
        tgt = q.strip()
        if key in processed_data:
            processed_data[key]['ref'].append(tgt)
        else:
            processed_data[key] = {
            'input':input,
            'ref':[tgt],
            'uid':idx,
            'context':context,
            'answer':answer
        }

    print('Complete preprocessing Phase I.')
    processed_data = list(processed_data.values())
    print(processed_data[2])
    if constrain:
        processed_data = add_tokenized_ref(processed_data, tokenizer)
        ex_output = mp_add_constraint(processed_data[0])
        print('Running example')
        print(processed_data[0])
        print(ex_output)
        
        pool = mp.Pool(processes=8)
        results = pool.map(mp_add_constraint, processed_data)
        processed_data = results
        print('Done adding constraints.')
    return processed_data

def format_inputs(context: str, answer: str):
    return f"{answer} \\n {context}"


def prepare_output_quoref(ex_id, input_text, search_outputs)->List[dict]:
    # 'field':['uid','ex_id','concepts','input','output']
    num_of_outputs = len(search_outputs.top_hypos)
    rts = []
    for i in range(num_of_outputs):
        data = {
            'uid': f"{ex_id}_{i}",
            'ex_id':ex_id,
            'concepts':input_text,
            'input':input_text,
            'output':search_outputs.top_hypos[i],
            'total_hypo':search_outputs.total_hypos_num
        }
        rts.append(data)
    return rts

import random
random.seed(2022)

def load_squad(split='validation', debug:bool=True,  constrain=False, tokenizer=None):

    # one concept id corresponds to mulitple outputs
    print('Preprocessing data')
    
    if debug:
        print('DEBUG mode on. Truncate to 50.')
        dataset = load_dataset('squad_v2',split=f"{split}[50%:60%]")
    else:
        dataset = load_dataset('squad_v2', split=split)

    processed_data = []
    for idx, data in enumerate(dataset):
        if data["answers"]["text"]:
            answer = data["answers"]["text"][0]
        else:
            continue
        context = data['context'][:1500]
        q = data['question']
        input = f"{answer} \\n {context}"
        tgt = q
        ref_tok = tokenizer(tgt)['input_ids']
        data_packet = {
            'input':input,
            'ref':[tgt],
            'ref_tok':[ref_tok],
            'uid':idx,
            'context':context,
            'answer':answer
        }
        # constraint is a random bigram
        if constrain:
            l = len(ref_tok)
            pos = random.randint(0, l-2)
            assert l-2 >= 0
            const = ref_tok[pos:pos+2]
            data_packet['const'] = [const]            
        processed_data.append(data_packet)

    print('Complete preprocessing. ')
    print(processed_data[2])
    return processed_data


def load_drop(split='validation', debug:bool=True,  constrain=False, tokenizer=None):

    # one concept id corresponds to mulitple outputs
    print('Preprocessing data')
    
    if debug:
        print('DEBUG mode on. Truncate to 50.')
        dataset = load_dataset('drop',split=f"{split}[50%:60%]")
    else:
        dataset = load_dataset('drop', split=split)

    processed_data = []
    for idx, data in enumerate(dataset):
        answer = data['answers_spans']['spans'][0]
        # if data["answers"]["text"]:
        #     answer = data["answers"]["text"][0]
        # else:
        #     continue
        context = data['passage'][:1500]
        q = data['question']
        input = f"{answer} \\n {context}"
        tgt = q
        ref_tok = tokenizer(tgt)['input_ids']
        data_packet = {
            'input':input,
            'ref':[tgt],
            'ref_tok':[ref_tok],
            'uid':idx,
            'context':context,
            'answer':answer
        }
        # constraint is a random bigram
        if constrain:
            l = len(ref_tok)
            pos = random.randint(0, l-2)
            assert l-2 >= 0
            const = ref_tok[pos:pos+2]
            data_packet['const'] = [const]            
        processed_data.append(data_packet)

    print('Complete preprocessing. ')
    print(processed_data[2])
    return processed_data




def load_xsum(split='test', debug:bool=True,  constrain=False, tokenizer=None):
    random.seed(2022)
    # one concept id corresponds to mulitple outputs
    print('Preprocessing data')
    
    if debug:
        print('DEBUG mode on. Truncate to 50.')
        dataset = load_dataset('xsum',split=f"{split}[0%:100%]")
    else:
        dataset = load_dataset('xsum', split=split)

    processed_data = []
    for idx, data in enumerate(dataset):
        if debug and random.random() < 0.95:
            continue
            
        context = data['document'][:1200]
        q = data['summary']
        input = f"{context}"
        tgt = q
        ref_tok = tokenizer(tgt)['input_ids']
        data_packet = {
            'input':input,
            'ref':[tgt],
            'ref_tok':[ref_tok],
            'uid':idx,
            'context':context,
            'datset_uid':data['id']
        }
        # constraint is a random bigram
        if constrain:
            l = len(ref_tok)
            pos = random.randint(0, l-2)
            assert l-2 >= 0
            const = ref_tok[pos:pos+2]
            data_packet['const'] = [const]            
        processed_data.append(data_packet)

    print('Complete preprocessing. ')
    print(processed_data[2])
    return processed_data