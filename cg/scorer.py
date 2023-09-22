import string
from typing import List
import re
class RewardTokenMatch:
    
    def __init__(self, task_rwd:float = 0.5) -> None:
        self.task_rwd = task_rwd
        self.name = 'token'
        
    def __repr__(self) -> str:
        return f"Rwd token match\t{self.task_rwd}"

    def set_tgt_words(self, tgt_words:List[str]):
        self.wl = tgt_words

    def score(self, current_node):
        if not self.task_rwd:
            return 0
        
        # use current wl
        inp = current_node.token_strs
        # remove special tokens
        inp = re.sub('</s>', '', inp)
        inp = re.sub('<pad>', '', inp)
        
        sent = inp.strip().lower().split()
        sent = set([tok.strip(string.punctuation) for tok in sent])
        sent_match = len(sent.intersection(self.wl))
        # if sent_match > 0 and random.random() > 0.9:
        #     logging.info(f"word found. {sent_match}\t {inp}")
        return (sent_match/ len(self.wl)) * (-self.task_rwd)
    