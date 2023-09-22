
import csv
from transformers import AutoTokenizer
from bbfs.constant import TASKS

model_name = TASKS['cg']['model']
field_name = TASKS['cg']['field']
tokenizer = AutoTokenizer.from_pretrained(model_name)

import os
print(os.getcwd())
fname = 'cg_bfs_20_10_5_0.1_10_0.0_output.csv'

outputs = []
def read_output_file_line(log_file, fields):
    with open(log_file, 'r') as wf:
        reader = csv.DictReader(wf)
        for row in reader:
            txt = row['output']
            ref_txt = tokenizer.decode(tokenizer.encode(txt),skip_special_tokens=True)
            ref_txt = ref_txt.strip()
            outputs.append(ref_txt)
    new_fname = fname[:-3] + 'txt'
    with open(new_fname, 'w') as fd:
        fd.write('\n'.join(outputs))
    print(f"Lines: {len(outputs)}")
read_output_file_line(fname, field_name)