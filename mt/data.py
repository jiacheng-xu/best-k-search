import json
import tqdm

DATA_DIR='dataset/machine_translation'
DATA_PREFIX='iate.414'

MODEL_NAME='Helsinki-NLP/opus-mt-en-de'
from transformers import MarianTokenizer, MarianMTModel
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)


src_file = f"{DATA_DIR}/newstest2017-iate/{DATA_PREFIX}.terminology.tsv.en"
tgt_file = f"{DATA_DIR}/newstest2017-iate/{DATA_PREFIX}.terminology.tsv.de"
constraint_file = f"{DATA_DIR}/constraint/iate.414.json"


def tokenize_constraints(tokenizer, raw_cts):
    def tokenize(phrase):
        token_ids = [tokenizer.encoder.get(x) for x in tokenizer.spm_target.EncodeAsPieces(phrase)]
        if phrase.startswith(' ('):
            token_ids = token_ids[1:]
        assert all([x is not None for x in token_ids]), f'unrecognized token in {phrase} {type}'
        return token_ids, True
    return [[list(map(tokenize, clause)) for clause in ct] for ct in raw_cts]

def read_constraints(file_name):
    cons_list = []
    with open(file_name, 'r') as f:
        for line in f:
            cons = []
            for concept in json.loads(line):
                cons.append([f' {c}' for c in concept])
            cons_list.append(cons)
    return cons_list


def load_mt(split='validation', debug:bool=True, constrain=False, tokenizer=None):
    # one concept id corresponds to mulitple outputs
    print('Preprocessing data')
    
    input_lines = [l.strip() for l in open(src_file, 'r').readlines()]
    target_lines = [l.strip() for l in open(tgt_file, 'r').readlines()]
    constraints_list = read_constraints(constraint_file)
    constraints_list = tokenize_constraints(tokenizer, constraints_list)
    processed_data = {}
    for idx, data in enumerate(input_lines):
        input = input_lines[idx]
        ref = target_lines[idx]
        constraint = constraints_list[idx]
        constraint = [c[0]  for c in constraint]
        processed_data[idx] = {
            'ref':[ref],
            'input':input,
            'uid':idx,
            'const':constraint
        }

    print('Complete preprocessing.')
    processed_data = list(processed_data.values())

    return processed_data