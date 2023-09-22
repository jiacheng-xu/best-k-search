dir = "/export/home/experimental/analyzing-uncertainty-nmt"
# files = ["wmt14-en-de.extra_refs", "wmt14-en-fr.extra_refs"]
fmap = {"ende": "wmt14-en-de.extra_refs", "enfr": "wmt14-en-fr.extra_refs"}

import os
from collections import defaultdict


def process_file(path, fname):
    with open(os.path.join(path, fname), "r") as fd:
        lines = fd.readlines()
    data = {}
    for line in lines:
        prefix, content = line.split("\t")
        content = content.strip()
        type_inp, idx = prefix.split("-")
        if idx not in data:
            assert type_inp == "S"
            data[idx] = {"input": content, "uid": idx, "ref": []}
        else:
            data[idx]["ref"].append(content)
    print("Complete preprocessing.")
    processed_data = list(data.values())
    print(processed_data[:2])
    return processed_data


def load_mt(task_name, debug=True):

    output = process_file(dir, fmap[task_name])
    
    if debug:
        import random
        random.shuffle(output)
        output = output[:200]
        print("truncate to 200")
    return output