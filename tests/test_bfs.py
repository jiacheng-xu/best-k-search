from transformers import AutoConfig,AutoModelForSeq2SeqLM,AutoTokenizer
import torch
import heapq


task='sum'
dataset='xsum'
model_name='facebook/bart-large-xsum'
device_name='cuda:2'

device = torch.device(device_name)
print(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

heap = []
doc_input_ids 
max_len=30 
grp_size=5
k_best=10

expansion_list = []
cnt = 0
while cnt < grp_size and heap:
    seed:BeamNode = vanilla_heap_pop(heap)
    expansion_list.append(seed)
    cnt += 1
dec_prefixes = [ x.get_token_idx() for x in expansion_list]
dec_input_tensors = assemble_pad(dec_prefixes,device=doc_input_ids.device)
output_probs, _, _ = run_inference_step(model, doc_input_ids, decoder_input_ids=dec_input_tensors, device=doc_input_ids.device, output_dec_hid=False)
values, indices = torch.topk(output_probs, k=k_best)