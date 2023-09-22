from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from bbfs.util import step_model_bart_decode, step_model_decode, step_model_seq2seq_decode

from fudge.util import *
from dec.bfs import init_search, vanilla_heap_pop
from dec.viz_center import proc
from dec.util import BeamNode, Scorer
from fudge.temp_decay import decay
from transformers import MarianMTModel

from scipy.special import softmax as sp_softmax
from scipy.stats.mstats import mquantiles
from transformers import T5ForConditionalGeneration,BartForConditionalGeneration,PegasusForConditionalGeneration
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class BestFirstSearch:
    def __init__(self, model, tokenizer, device, max_len, k_best, grp_size, temp_decay, heap_top_k,  use_heap_sample:bool, max_heap_size,end_token_idx_list, min_len:int, task_rwd, threshold=0.01, verbose=True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.k_best=k_best
        self.use_heap_sample = use_heap_sample
        self.grp_size = grp_size
        self.max_heap_size = max_heap_size
        self.verbose = verbose
        self.top_k = heap_top_k
        self.min_tokens_to_keep = grp_size
        self.filter_value = -float("Inf")
        self.temp_decay = temp_decay
        self.min_len = min_len
        self.end_token_idx_list = end_token_idx_list
        self.task_rwd = task_rwd
        self.threshold = threshold
        # self.score_mode = score

    def heap_sample(self, current_time_stamp, heap):
        expansion_list = []
        cnt = 0

        def comp_score(item):
            original_score, time_stamp, node = item
            time_offset = decay(current=current_time_stamp, stamp=time_stamp,name='poly', weight=self.temp_decay)   # also positive, larger, worse
            
            score_meet_constraint = 0
            if self.task_rwd > 0:
                if node.stepped:
                    score_meet_constraint -= self.task_rwd
                if node.completed:
                    score_meet_constraint -= 0.5 * self.task_rwd
            if self.verbose and  random.random()>0.9999 :
                logging.info(f"model score: {round(original_score,3)}\ttime:{round(time_offset,3)}\tconst:{round(score_meet_constraint,3)}")
            return original_score + time_offset + score_meet_constraint

        processed_heap = list(map(lambda x: comp_score(x), heap))
        
        if self.verbose and random.random() > 0.99 and len(processed_heap) > 5:
            m = sp_softmax(processed_heap)
            # logging.info(f"Quantile: {[round(q, 2) for q in quantiles(processed_heap, n=4)]}\n")
            logging.info(f"Quantile after softmax: {mquantiles(m)}") 
            
        scores = torch.tensor(processed_heap,dtype=torch.float,requires_grad=False)
        
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k

        # heap prune
        cut_off_size = min(self.max_heap_size, scores.size(-1))
        indices_prune = scores > torch.topk(scores, cut_off_size, largest=False)[0][ -1, None]  # mask. everything True means they will be pruned after this iteration.
        
        candidate_indices = scores > torch.topk(scores, top_k, largest=False)[0][ -1, None]     # mask. everything True means they will not be considered as candidates. 
        scores = scores.masked_fill(candidate_indices, self.filter_value)

        non_inf_idx = (scores != -float("inf")).nonzero(as_tuple=True)[0]  # index possible for candidates
        samples_idx = non_inf_idx.tolist()
        # if non_inf_idx.size()[-1] > self.grp_size:
            # non_inf_idx = torch.multinomial(non_inf_idx, num_samples=self.grp_size)    
        # sel_index_list = non_inf_idx.tolist()
        # samples_idx = [ x[1] for x in sel_index_list]
        # logging.info(f"Sample from top {len(samples_idx)}")
        if len(samples_idx) > self.grp_size:
            random.shuffle(samples_idx)
            samples_idx = samples_idx[:self.grp_size]
        
        
        # Now non_inf_idx is the set of candidates we are going to explore
        # sel_candidates = non_inf_idx.tolist()
        rm_nodes = indices_prune.nonzero(as_tuple=True)[0]
        rm_list = rm_nodes.tolist()
        expansion_list = [ heap[idx][2] for idx in samples_idx] # 0 is original score, 1 is time, 2 is node
        cnt = len(expansion_list)
        
        for idx in samples_idx:
            heap[idx] = None
        for idx in rm_list:
            heap[idx] = None
        heap = list(filter(None, heap))

        return expansion_list, cnt,heap
    
    def heap_sort(self, inp):
        # convert heap to a real heap
        copy_inp = inp.copy()
        heapq.heapify(copy_inp)
        expansion_list = []
        cnt = 0
        # regular mode
        while cnt < self.grp_size and copy_inp:
            seed: BeamNode = vanilla_heap_pop(copy_inp)
            expansion_list.append(seed)
            cnt += 1
        heap = list(copy_inp)
        return expansion_list, cnt, heap
    
    # entrance
    def get_expansion_candidates(self, current_time_stamp, heap):
        # Get grp size of candidates from heap.
        if self.use_heap_sample:
            exp, cnt, heap = self.heap_sample(current_time_stamp, heap)
        else:
            # regular mode
            exp, cnt, heap =  self.heap_sort(heap)
        return exp, cnt, heap
    
    def step_bfs(self,
        doc_input_token,
        heap,
        default_scorer,
        task_scores,
        precondition_topk:int=200,
        T=None, # batch time
        verbose=True,
    ):
        expansion_list, cnt , heap= self.get_expansion_candidates(T, heap)
        finished_hypos = []
        
        doc_input_token_len = len(doc_input_token) # length of input (shared, encoding side)
        if isinstance(self.model, T5ForConditionalGeneration) or isinstance(self.model, MarianMTModel) or isinstance(self.model, PegasusForConditionalGeneration):
            logits, enc_input_tensor, dec_input_tensor, batch_size, padding_pos, enc_dec_length = step_model_seq2seq_decode(self.model, self.device, expansion_list, doc_input_token, self.tokenizer)
            top_logits, indices = torch.topk(logits, k=precondition_topk, dim=-1)  # batch x pretopk
        elif isinstance(self.model,BartForConditionalGeneration) or isinstance(self.model,MBartForConditionalGeneration) :

            logits,  dec_input_tensor, batch_size,  = step_model_bart_decode(self.model, self.device, expansion_list, doc_input_token, self.tokenizer)
            top_logits, indices = torch.topk(logits, k=precondition_topk, dim=-1)  # batch x pretopk

        else:
            (
                logits, # b x v
                concat_input_tensor,    # b x e_d, padded input of enc+dec
                dec_input_tensor,       # b x d, padded input of dec
                batch_size,             
                padding_pos,            # list, size = b, the start position of non pad token
                enc_dec_length,         # list, size = b, the length of e+d excluding PAD
            ) = step_model_decode(self.model, self.device, expansion_list, doc_input_token, self.tokenizer)
        
            top_logits, indices = torch.topk(logits, k=precondition_topk, dim=-1)  # batch x pretopk
        
            exp_concat_input_tensor = unsqueeze_expand_sec_dim(concat_input_tensor, precondition_topk)
            exp_dec_input_tensor = unsqueeze_expand_sec_dim(dec_input_tensor, precondition_topk)
        
            # candidates come with the newly decoded content
            extend_concat_input_tensor = torch.cat(
            [exp_concat_input_tensor, indices.unsqueeze(2) ], dim=2)  # batch  x pretopk  x seq+1
        
            lengths = torch.LongTensor(enc_dec_length).to(self.device)
            expanded_lengths = unsqueeze_expand_sec_dim(lengths + 1, precondition_topk)     ## expandlen is original + 1
        
        full_logits = 0
        aux_logits = []

        full_logits += top_logits

        post_logits, post_indices = full_logits.topk(self.k_best, dim=1)
        post_probs = F.softmax(post_logits, dim=1)
        post_probs_list = post_probs.cpu().tolist()
        # index_into_top_indices = post_indices[torch.arange(batch_size).to(post_indices.device), torch.multinomial(post_probs, 1).flatten()] # batch
        index_into_top_indices = post_indices[
            torch.arange(batch_size).to(post_indices.device), :self.k_best
        ]  # batch, k_best

        
        for b in range(batch_size):
            next_indices = indices[b][index_into_top_indices[b]].cpu().tolist() # k best options
            # cur_seq = concat_input_tensor[b].cpu().tolist()  # seq_len
            
            # visualization
            if verbose and random.random() > 0.99:
                logging.info(
                    f"Prefix: {self.tokenizer.decode(doc_input_token)}"
                    + f"///{self.tokenizer.decode(dec_input_tensor[b].cpu())}"
                )
            
                visualize_logits(top_logits, indices,"RAW", self.tokenizer,b,10)
                for aux in aux_logits:
                    visualize_logits(aux[0], indices, aux[2], self.tokenizer, b, 10)
                visualize_logits(full_logits, indices, "full", self.tokenizer, b, 10)
            # end of visualization
            
            # trick: for the same token's variations (upper/lower), pick the one with higher score
            seen = []
            
            for rank in range(self.k_best):
                v = post_probs_list[b][rank]
                tok = next_indices[rank]
                tok_str_unique = self.tokenizer.decode(tok).lower().strip()
                if tok_str_unique in seen:
                    continue
                if expansion_list[b].prob == 1 and self.tokenizer.decode(tok).isalpha() and self.tokenizer.decode(tok).islower():
                    # if this is the beginning of a sentence, force to upper case it
                    old_tok_str = self.tokenizer.decode(tok)
                    tok = self.tokenizer.convert_tokens_to_ids( tok_str_unique.capitalize().strip() )
                    logging.info(f"Manually uppercase: {old_tok_str}=>{ self.tokenizer.decode(tok) }")

                tmp_state = BeamNode(
                    prob=v, token_idx=tok,
                    token_str=self.tokenizer.decode(tok),
                    prev=[expansion_list[b]],
                    prev_score=[math.log(v)],
                    glob_step=proc(), batch_step=T,
                    stop_token_idx_list=self.end_token_idx_list, min_len=self.min_len
                )
                if tok in self.end_token_idx_list and tmp_state.length < self.min_len:
                    # logging.info(f"trash. {tmp_state}")
                    continue
                if tmp_state.length >= self.max_len and not tmp_state.finished:
                    # logging.info(f"trash. {tmp_state}")
                    continue
                if tmp_state.finished:
                    if verbose and random.random() > 0.99:
                        logging.info(f"Ending: {self.tokenizer.decode(dec_input_tensor[b].cpu().tolist() + [next_indices[rank]])}"
                    )
                    finished_hypos.append(tmp_state)
                    continue
                
                if v < self.threshold:
                    break
                
                seen.append(tok_str_unique)
                
                # seq_model_score = default_scorer.get_model_score(tmp_state)  # stateless?
                
                # if self.score_mode == "last":
                #     model_score = -math.log(v)
                # elif self.score_mode == "avg":
                #     seq_model_score = default_scorer.get_model_score(tmp_state)  # stateless?
                # elif self.score_mode == "sum":
                #     pass
                heap_score = default_scorer.get_model_score(tmp_state)  # stateless?
                heap.append(( heap_score, T, tmp_state))
        
        return finished_hypos, cnt, heap

    def f_best_first_search(self,
        default_scorer,
        task_scores,
        doc_input,
        dec_prefix="",
        precondition_topk: int = 200,
        comp_budget: Optional[int] = 300,
        force_words_ids=None,
        verbose=True
    ):
        ncalls = 0
        proc.reset()    # this is a global counter
        heap = []  # nodes at the frontier of search    it can be a list instead of a heap
        finished_hypos = []

        tok_input: List[int] = self.tokenizer.encode(doc_input)
        tok_dec_prefix: List[int] = self.tokenizer.encode(dec_prefix,add_special_tokens=False)
        T = 0
        init_search(self.tokenizer,tok_dec_prefix, heap, stop_token_idx_list=self.end_token_idx_list, T=T, force_words_ids=force_words_ids)
        T += 1
        
        # init_seed only has the decoding prefix

        with torch.no_grad():
            while ncalls < comp_budget and heap:

                completed_hyps, added_num_calls,heap = self.step_bfs(tok_input,heap,default_scorer,task_scores,precondition_topk,T=T,verbose=verbose)
                ncalls += added_num_calls
                T += 1
                if completed_hyps:
                    finished_hypos.append( completed_hyps) 
        return finished_hypos, heap


