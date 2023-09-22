from fudge.fudge_model import Model
from util import *
from fudge.score import ScoreModule

def load_topic_model(args, gpt_pad_id, dataset_info):
    checkpoint = torch.load(args.topic_ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.topic_ckpt, checkpoint['epoch']))
    return conditioning_model

import os

class RewardTopic:
    
    def __init__(self, task_rwd) -> None:
        self.task_rwd = task_rwd
        self.name = 'rwd_topic'
        self.tw_dir = '/export/home/experimental/naacl-2021-fudge-controlled-generation/topic_data/test_wordlists'
        self.categories = {}
        self.wl = None
    
    def set_category(self, category):
        self.wl = self.find_cat_topic_words(category)
        
    def find_cat_topic_words(self, category)-> set:
        if category not in self.categories:
            words = []
            with open(os.path.join(self.tw_dir, category + '.txt'), 'r') as rf:
                for line in rf:
                    words.append(line.strip().lower())
            self.categories[category] = set(words)
        return self.categories[category]
    
    def score(self,  current_node):
        if not self.task_rwd:
            return 0
        
        # use current wl
        inp = current_node.token_strs
        sent = inp.strip().lower().split()
        sent = set([tok.strip(string.punctuation) for tok in sent])
        sent_match = len(sent.intersection(self.wl))
        if sent_match > 0 and random.random() > 0.9:
            logging.info(f"word found. {sent_match}\t {inp}")
        return sent_match * (-self.task_rwd)
    
class ScoreTopic(ScoreModule):
    def __init__(self, model, device, condition_lambda) -> None:
        super().__init__(model, device, condition_lambda)
        self.name = 'topic'

    def score(self, extend_concat_input_tensor, expanded_lengths):
        batch_size, precondition_topk, _ = extend_concat_input_tensor.size()
        expanded_future_words = self.future_words.unsqueeze(0).unsqueeze(1).expand(batch_size, precondition_topk, -1) # batch x topk x N
        expanded_tokens_left = self.tokens_left.unsqueeze(1).expand(-1, precondition_topk) # batch x topk
        # expanded_lengths = (lengths + 1).unsqueeze(1).expand(batch_size, precondition_topk) # batch x topk

        if self.condition_lambda == 0:
            condition_logits = torch.zeros_like(expanded_future_words).float()
        else:
            condition_logits = self.model(extend_concat_input_tensor.flatten(0, 1), # batch*topk x seq+1
                                                expanded_lengths.flatten(0, 1), # batch*topk
                                                expanded_future_words.flatten(0, 1), # batch*topk x N
                                                self.log_probs, # N
                                                expanded_tokens_left.flatten(0, 1)) # batch*topk
            condition_logits = condition_logits.view(batch_size, precondition_topk, -1) # batch x topk x N
            condition_logits = condition_logits - torch.log(1 + torch.exp(condition_logits)) # get correct log probs

        condition_logits = torch.mean(condition_logits, dim=2)
        return condition_logits
        
    def prepare(self, condition_words, dataset_info):

        self.condition_words = condition_words.split()
        future_words = torch.LongTensor([dataset_info.word2index[cw] for cw in self.condition_words]).to(self.device) # N
        log_probs = torch.Tensor([math.log(dataset_info.vocab[cw] / dataset_info.total_words) for cw in self.condition_words]).to(self.device) # N
        self.future_words=future_words
        self.log_probs=log_probs
        
    def ext_feat(self, length_cutoff, decoded_length):
        # tokens_left = torch.LongTensor([length_cutoff - lengths.max() for _ in range(batch_size)]).to(device)
        self.tokens_left = length_cutoff - decoded_length