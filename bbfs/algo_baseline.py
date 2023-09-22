
import statistics

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from bbfs.util import step_model_decode, step_model_seq2seq_decode

from fudge.util import *
from dec.bfs import assemble_pad, assemble_pad_plus, init_search, vanilla_heap_pop
from dec.viz_center import proc
from dec.util import BeamNode, Scorer
from fudge.temp_decay import decay




class BeamSearch:
    def __init__(self, model, tokenizer, device, max_len,  beam_size, task_rwd, do_sample=False, typical_p=1., top_p=1.0, num_beam_groups=1,diversity_penalty=0.0, verbose=True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.beam_size = beam_size
        self.verbose = verbose
        self.min_tokens_to_keep = beam_size
        self.filter_value = -float("Inf")
        self.do_sample = do_sample
        self.typical_p = typical_p
        self.top_p = top_p
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.task_rwd = task_rwd
        
    def f_beam_search(self,
        default_scorer,
        task_scores,
        doc_input,
        dec_prefix="",
        precondition_topk: int = 200,
        comp_budget: Optional[int] = 300,
        force_words_ids = None,
        verbose=False
    ):
        input_text = doc_input
        features = self.tokenizer([input_text], return_tensors='pt').to(self.device)
        
        output = self.model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=self.max_len, 
               num_beams=self.beam_size, 
               do_sample=self.do_sample, 
               typical_p=self.typical_p,
               top_p=self.top_p,
               diversity_penalty=self.diversity_penalty,
               num_beam_groups=self.num_beam_groups,
               num_return_sequences=self.beam_size,
               forced_bos_token_id=force_words_ids
               )
        return output, None

    def f_sample(self,
        default_scorer,
        task_scores,
        doc_input,
        dec_prefix="",
        precondition_topk: int = 200,
        comp_budget: Optional[int] = 300,
        force_words_ids = None,
        verbose=False
    ):
        input_text = doc_input
        features = self.tokenizer([input_text], return_tensors='pt').to(self.device)
        output = self.model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=self.max_len, 
               num_beams=1, 
               do_sample=self.do_sample, 
               typical_p=self.typical_p,
               top_p=self.top_p,
               diversity_penalty=self.diversity_penalty,
               num_beam_groups=self.num_beam_groups,
               num_return_sequences=self.beam_size,
               forced_bos_token_id=force_words_ids
               )
        return output, None

    def const_beam_search(self,
        default_scorer,
        task_scores,
        doc_input,
        dec_prefix="",
        force_words_ids=None,
        precondition_topk: int = 200,
        comp_budget: Optional[int] = 300,
        verbose=False
    ):
        assert self.top_p == 1.0
        assert not self.do_sample
        assert self.typical_p == 1.0 or self.typical_p is None
        
        input_text = doc_input
        features = self.tokenizer([input_text], return_tensors='pt').to(self.device)
        output = self.model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=self.max_len, 
               force_words_ids=[force_words_ids],
               num_beams=self.beam_size, 
               do_sample=self.do_sample, 
               typical_p=self.typical_p,
               top_p=self.top_p,
               num_return_sequences=self.beam_size
               )
        return output ,None