from datasets import load_dataset
from tqdm import tqdm
from typing import List

def set_gec_constraint():
    # given one input and a list of corrections, return a set of constraints which do not exist in the input
    pass

def load_gec(tokenizer,split='validation', debug:bool=True):

    # one concept id corresponds to mulitple outputs
    print('Preprocessing data')
    
    if debug:
        print('DEBUG mode on. Truncate to 50.')
        dataset = load_dataset('jfleg',split=f"{split}[50%:70%]")
    else:
        dataset = load_dataset('jfleg', split=split)

    processed_data = []
    for idx, data in enumerate(dataset):
        sent = data["sentence"]
        corrections = data['corrections']
        processed_data.append({
            'input':sent,
            'ref':corrections,
            'uid':idx,
        })
    
    print('Complete preprocessing. ')
    print(processed_data[2])
    return processed_data