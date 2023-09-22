from typing import List
from dataclasses import dataclass
from fudge.eval_topic_metrics import eval_api

@dataclass
class SearchOutput:
    complete:bool
    total_hypos_num: int
    all_hypos_node:List
    all_hypos:List
    top_hypos: List
    
from fudge.util import *

def handle_output(finished_hypos:List,  input_text:str, tokenizer, target_output_num:int) -> SearchOutput:
    if finished_hypos != [] and isinstance(finished_hypos, torch.Tensor):
        decoded = [tokenizer.decode(x,skip_special_tokens=True) for x in finished_hypos]
        # traditional output
        return SearchOutput(
        complete=True, 
        total_hypos_num=len(decoded), 
        all_hypos_node=None,
        all_hypos=decoded, 
        top_hypos=decoded[:target_output_num]
        )
    complete = True if finished_hypos else False
    total_num = len(finished_hypos)
    logging.info(f"Set of finished hypo: {len(finished_hypos)}")
    random.shuffle(finished_hypos)
    outputs = []
    if not finished_hypos:
        outputs = [""] * target_output_num
        return SearchOutput(complete=False, total_hypos_num=0, all_hypos=outputs, all_hypos_node=[],top_hypos= outputs)
    elif len(finished_hypos) >= target_output_num:
        # finished_hypos = finished_hypos[:target_output_num]
        finished_hypos = [x[0] for x in finished_hypos]
    else:
        finished_hypos = [item for sublist in finished_hypos for item in sublist]
        random.shuffle(finished_hypos)
    
    cur_pred = []
    for hypo in finished_hypos:
        hypo_tok_id = hypo.get_token_idx()
        output_string = tokenizer.decode(hypo_tok_id, skip_special_tokens=True)
        logging.info(f"Input | gen: {input_text[:50]}|{output_string}")
        cur_pred.append(output_string)
    return SearchOutput(
        complete=complete, 
        total_hypos_num=total_num, 
        all_hypos_node=finished_hypos,
        all_hypos=cur_pred, 
        top_hypos=cur_pred[:target_output_num]
        )

def down_sample():
    pass
def up_sample():
    pass

import os
import csv

def main():
    dir = '/export/home/cond-text-gen/outputs'
    files = os.listdir(dir)
    files = [os.path.join(dir, f) for f in files if f.endswith('preds.log')]
    print(files)
    result_board = 'results.csv'

    for f in files:
        result = eval_api(f)
        file_exists = os.path.exists(result_board)
        if not file_exists:
            with open(result_board, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                writer.writeheader()
        with open(result_board, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=result.keys())
            writer.writerow(result)

if __name__ == "__main__":
    main()