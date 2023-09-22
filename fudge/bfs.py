
import statistics
from tabnanny import verbose
from fudge.data_control import load_topic_data
from fudge.run_poetry import setup_poetry_models
from fudge.score_poem import NewLine,Iambic,Rhyme
from fudge.score_topic import ScoreTopic, load_topic_model
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from fudge.util import *
from dec.bfs import assemble_pad, assemble_pad_plus, init_search, vanilla_heap_pop
from dec.viz_center import proc
from dec.util import BeamNode, Scorer
from fudge.temp_decay import decay

from statistics import quantiles
from scipy.special import softmax as sp_softmax
from scipy.stats.mstats import mquantiles


class BestFirstSearch:
    def __init__(self, model, tokenizer, device, max_len, k_best,grp_size, temp_decay, heap_top_k,  use_heap_sample:bool,  max_heap_size, verbose=True) -> None:
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
        
        
    def heap_sample(self, current_time_stamp, heap):
        expansion_list = []
        cnt = 0
        if len(heap) < 3:
            print(heap)

        def comp_score(item):
            original_score, time_stamp, _ = item
            time_offset = decay(current=current_time_stamp, stamp=time_stamp,name='poly', weight=self.temp_decay)   # also positive, larger, worse
            if random.random()>0.995 and self.verbose:
                logging.info(f"model score: {round(original_score,3)}\t time:{round(time_offset,3)}")
            return original_score + time_offset

        processed_heap = list(map(lambda x: comp_score(x), heap))
        if self.verbose and random.random()>0.9 and len(processed_heap) > 5:
            m = sp_softmax(processed_heap)
            # logging.info(f"Max: {max(processed_heap)}\tMin:{min(processed_heap)}\tMean:{statistics.mean(processed_heap)}")
            logging.info(f"Quantile: {[round(q, 2) for q in quantiles(processed_heap, n=4)]}\n")
            logging.info(f"Quantile after softmax: {mquantiles(m)}") 
            
        scores = torch.tensor([processed_heap],dtype=torch.float,requires_grad=False)
        
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        # print(torch.topk(scores, top_k, largest=False))
        indices_to_remove = scores > torch.topk(scores, top_k, largest=False)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        
        # sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        # cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        
        # # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        # sorted_indices_to_remove = cumulative_probs > top_p

        # # Shift the indices to the right to keep also the first token above the threshold
        # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        # sorted_indices_to_remove[..., 0] = 0

        # # scatter sorted tensors to original indexing
        # indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        # scores = scores.masked_fill(indices_to_remove,  -float("Inf"))
        
        non_inf_idx = (scores != -float("inf")).nonzero()
        sel_index_list = non_inf_idx.tolist()
        samples_idx = [ x[1] for x in sel_index_list]
        # logging.info(f"Sample from top {len(samples_idx)}")
        if len(samples_idx) > self.grp_size:
            random.shuffle(samples_idx)
            samples_idx = samples_idx[:self.grp_size]
        expansion_list = [ heap[idx][2] for idx in samples_idx] # 0 is original score, 1 is time, 2 is node
        cnt = len(expansion_list)
        for idx in samples_idx:
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
        (
            logits, # b x v
            concat_input_tensor,    # b x e_d, padded input of enc+dec
            dec_input_tensor,       # b x d, padded input of dec
            batch_size,             
            padding_pos,            # list, size = b, the start position of non pad token
            enc_dec_length,         # list, size = b, the length of e+d excluding PAD
        ) = gpt_decode(self.model, self.device, expansion_list, doc_input_token, self.tokenizer)
        
        top_logits, indices = torch.topk(logits, k=precondition_topk, dim=-1)  # batch x pretopk
        
        exp_concat_input_tensor = unsqueeze_expand_sec_dim(concat_input_tensor, precondition_topk)
        exp_dec_input_tensor = unsqueeze_expand_sec_dim(dec_input_tensor, precondition_topk)
        
        # candidates come with the newly decoded content
        extend_concat_input_tensor = torch.cat(
            [exp_concat_input_tensor, indices.unsqueeze(2) ], dim=2
        )  # batch  x pretopk  x seq+1
        
        extend_dec_candidates = torch.cat(
            [exp_dec_input_tensor, indices.unsqueeze(2)], dim=2
        )  # batch x prek x dec+1
        
        extend_concat_input_candidates_list = (
            extend_concat_input_tensor.cpu().tolist()
        )
        
        lengths = torch.LongTensor(enc_dec_length).to(self.device)
        expanded_lengths = unsqueeze_expand_sec_dim(lengths + 1, precondition_topk)     ## expandlen is original + 1
        
        full_logits = 0
        aux_logits = []
        for k, score_module in task_scores.items():

            if score_module.name == 'topic' and k =='cond_topic':
                score_module.ext_feat(self.max_len, lengths-doc_input_token_len)
                task_logits = score_module.score(extend_concat_input_tensor, expanded_lengths)
                aux_logits.append([task_logits, score_module.condition_lambda, score_module.name])
                full_logits += task_logits * score_module.condition_lambda
            # elif k == 'rwd_topic':
            #     score_module.score()
            else:
                pass
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
            cur_seq = concat_input_tensor[b].cpu().tolist()  # seq_len

            if verbose and random.random() > 0.99:
                logging.info(
                    f"Prefix: {self.tokenizer.decode(doc_input_token)}"
                    + f"///{self.tokenizer.decode(dec_input_tensor[b].cpu())}"
                )

                visualize_logits(top_logits, indices,"RAW", self.tokenizer,b,10)
                for aux in aux_logits:
                    visualize_logits(aux[0], indices, aux[2], self.tokenizer, b, 10)
                visualize_logits(full_logits, indices, "full", self.tokenizer, b, 10)
                print('')
            for rank in range(self.k_best):
                v = post_probs_list[b][rank]
                tok = next_indices[rank]
                tmp_state = BeamNode(
                    prob=v,
                    token_idx=tok,
                    token_str=self.tokenizer.decode(tok),
                    prev=[expansion_list[b]],
                    prev_score=[math.log(v)],
                    glob_step=proc(),
                    batch_step=T,
                )
                # book.add_child()

                if tmp_state.length >= self.max_len:
                    if verbose and random.random() > 0.99:
                        logging.info(
                        f"Ending: {self.tokenizer.decode(cur_seq[doc_input_token_len + padding_pos[b]:] + [next_indices[rank]])}"
                    )
                    finished_hypos.append(tmp_state)
                    continue
                if  tmp_state.token_idx == 50256 or v < 0.1:
                    continue
                # model_score = default_scorer.get_model_score(tmp_state)  # stateless?
                model_score = -math.log(v)
                
                # add guidance score
                for k, score_module in task_scores.items():
                    if  k =='rwd_topic':
                        topic_meet_score = score_module.score(tmp_state)

                heap_score = model_score + topic_meet_score # smaller the better
                heap.append(( heap_score, T, tmp_state))
        
        return finished_hypos, cnt, heap

    def f_best_first_search(self,
        default_scorer,
        task_scores,
        doc_input,
        dec_prefix="",
        precondition_topk: int = 200,
        comp_budget: Optional[int] = 300,
        eos_idx: int = 50256,
    ):
        ncalls = 0
        proc.reset()    # this is a global counter
        heap = []  # nodes at the frontier of search    it can be a list instead of a heap
        finished_hypos = []

        tok_input: List[int] = self.tokenizer.encode(doc_input)
        tok_dec_prefix: List[int] = self.tokenizer.encode(dec_prefix)
        T = 0
        init_search(self.tokenizer,tok_dec_prefix, heap, T=T, eos_idx=eos_idx)
        T += 1

        # init_seed only has the decoding prefix

        
        with torch.no_grad():
            while ncalls < comp_budget and heap:
                if verbose:
                    logging.info(f"")
                completed_hyps, added_num_calls,heap = self.step_bfs(tok_input,heap,default_scorer,task_scores,precondition_topk,T=T)
                ncalls += added_num_calls
                T += 1
                if completed_hyps:
                    finished_hypos.append( completed_hyps) 
        return finished_hypos, heap


