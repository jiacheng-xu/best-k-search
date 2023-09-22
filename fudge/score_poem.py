import torch, math
from collections import defaultdict
import string
from fudge.constants import *


class Iambic:
    def __init__(self, model, device, condition_lambda=1) -> None:
        self.condition_lambda = condition_lambda
        self.device = device
        self.model = model

    def score(self, extended_dec_candidates,expanded_lengths, doc_input_token_len):
        # expanded_dec_input: batch x prek x dec_seq
        # extended_dec_candidates: batch x prek x dec_seq+1
        # expanded_lengths: batch x prek
        if self.condition_lambda == 0:
            iambic_logits = torch.zeros_like(expanded_lengths).float()
        else:
            batch_size, precondition_topk, _ = extended_dec_candidates.size()
            # truncate prefix because we trained on single lines
            iambic_logits = self.model(
                extended_dec_candidates.flatten(0, 1),
                expanded_lengths.flatten(0, 1) - doc_input_token_len,
                None,
                None,
                None,
            )[
                :, -1
            ]  # batch*topk x seq+1 -> batch*topk
            iambic_logits = iambic_logits.view(batch_size, precondition_topk)
            iambic_logits = iambic_logits - torch.log(1 + torch.exp(iambic_logits))
        return iambic_logits


class Rhyme:
    def __init__(self, model, device, rhyme_info, condition_lambda=1) -> None:
        self.model = model
        self.rhyme_info = rhyme_info
        self.device = device
        self.condition_lambda = condition_lambda

    def add_rhyme_group(self, current_text):
        ending_word = current_text.split()[-1].strip(string.punctuation)
        word2rhyme_group = defaultdict(
            lambda: UNKNOWN_RHYME_GROUP, self.rhyme_info.word2rhyme_group
        )
        rhyme_group = word2rhyme_group[ending_word]
        rhyme_group_index = self.rhyme_info.rhyme_group2index[rhyme_group]
        future_words = torch.LongTensor([rhyme_group_index]).to(self.device)  # 1
        self.log_probs = torch.Tensor(
            [
                math.log(
                    self.rhyme_info.rhyme_group_counts[rhyme_group]
                    / self.rhyme_info.total_rhyme_groups
                )
            ]
        ).to(
            self.device
        )  # 1
        self.future_words = future_words

    def score(
        self,
        new_input_candidates,
        expanded_future_words,
        expanded_lengths,
        expanded_syllables_to_go,
        batch_size,
        precondition_topk,
    ):
        if self.condition_lambda == 0:
            rhyme_logits = torch.zeros_like(expanded_lengths).float()
        else:
            rhyme_logits = self.model(
                new_input_candidates.flatten(0, 1),  # batch*topk x seq+1
                expanded_lengths.flatten(0, 1),  # batch*topk
                expanded_future_words.flatten(0, 1),  # batch*topk x N
                self.log_probs,  # N
                expanded_syllables_to_go.flatten(0, 1),
            )  # batch*topk
            rhyme_logits = rhyme_logits.view(
                batch_size, precondition_topk, -1
            )  # batch x topk x N
            rhyme_logits = rhyme_logits - torch.log(
                1 + torch.exp(rhyme_logits)
            )  # batch x topk x N
            rhyme_logits = rhyme_logits.squeeze(2)  # batch x topk
        return rhyme_logits


class NewLine:
    def __init__(self, model, device, condition_lambda=1) -> None:
        self.device = device
        self.model = model
        self.condition_lambda = condition_lambda

    def score(
        self,
        new_input_candidates,
        expanded_future_words,
        expanded_lengths,
        expanded_syllables_to_go,
        batch_size,
        precondition_topk,
        log_probs,
    ):
        if self.condition_lambda == 0:
            newline_logits = torch.zeros_like(expanded_lengths).float()
        else:
            newline_logits = self.model(
                new_input_candidates.flatten(0, 1),  # batch*topk x seq+1
                expanded_lengths.flatten(0, 1),  # batch*topk
                expanded_future_words.flatten(0, 1),  # batch*topk x N
                log_probs,  # N
                expanded_syllables_to_go.flatten(0, 1),
            )  # batch*topk
            newline_logits = newline_logits[:, -1].view(
                batch_size, precondition_topk, -1
            )  # batch x topk x N
            newline_logits = newline_logits - torch.log(
                1 + torch.exp(newline_logits)
            )  # batch x topk x N
            newline_logits = newline_logits.squeeze(2)  # batch x topk
        return newline_logits


class PoemScorers:
    def __init__(self) -> None:
        self.iambic = Iambic()
        self.rhyme = Rhyme()
        self.newline = NewLine()

