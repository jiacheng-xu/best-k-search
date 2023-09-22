import shlex, subprocess
from argparse import ArgumentParser
import os

import subprocess
import multiprocessing as mp
from tqdm import tqdm
import time
import os
os.chdir('/export/home/cond-text-gen')

num_gpu = 8

def work(data):
    data = ["/opt/conda/bin/python","bbfs/run.py"] +data
    data = [str(x) for x in data if x]
    print(data)
    # command = ['python', 'worker.py', sec_sleep]
    subprocess.call(data)

from multiprocessing import Pool
import json
import random

def main():
    program_name = input("program name")
    with open(f"scripts/program-{program_name}.json",'r') as fd:
        data = json.load(fd)
    run_data = data["args"]
    print(run_data)

    random.shuffle(run_data)
    # add device
    run_data = [ x + ["--device", f"cuda:{str(idx % num_gpu)}"] for idx, x in enumerate(run_data)]

    with Pool(processes=num_gpu*3) as pool:
        # print same numbers in arbitrary order
        for i in pool.imap_unordered(work, run_data):
            pass

if __name__ == "__main__":
    print("Excuting jobs.")
    # [args.task, args.algo, args.max_len, args.beam_size, args.group_size, args.temp_decay, args.heap_top_k, args.typical_p, args.top_p, args.num_beam_groups, args.task_rwd]

    main()
    

