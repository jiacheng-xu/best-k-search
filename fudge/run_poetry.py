path = "/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt"

from email.policy import default
from constants import PAD_TOKEN,POETRY_BANNED_TOKENS

from fudge_model import Model
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelWithLMHead
import torch,pickle
from constants import UNKNOWN_RHYME_GROUP,MAX_SYLLABLES_PER_WORD,POETRY_LINE_SYLLABLES,PHRASE_ENDS
import torch.nn.functional as F
# python -u evaluate_poetry.py --iambic_ckpt ckpt/poetry/iambic_predictor/model.pth.tar --rhyme_ckpt ckpt/poetry/rhyme_predictor/model.pth.tar --newline_ckpt ckpt/poetry/newline_predictor/model.pth.tar --dataset_info ckpt/poetry/rhyme_predictor/dataset_info --rhyme_info ckpt/poetry/rhyme_predictor/rhyme_info --prefix_file poetry_data/couplet_prefixes.txt --precondition_topk 200 > poetry_preds.log
# python eval_poetry_metrics.py --pred_file poetry_preds.log --prefix_file poetry_data/couplet_prefixes.txt
import pronouncing



def count_syllables(words):
    syllables = 0
    for word in words.split():
        word = word.strip().strip(string.punctuation)
        try:
            phones_list = pronouncing.phones_for_word(word)
            stresses = pronouncing.stresses(phones_list[0])
            syllables += min(MAX_SYLLABLES_PER_WORD, len(stresses))
        except:
            # if we don't know, just do a quick approximation here; it shouldn't come up too often
            syllables += min(MAX_SYLLABLES_PER_WORD, round(len(word) / 3))
    return syllables




def setup_poetry_models(iambic_ckpt, rhyme_ckpt,newline_ckpt, gpt_pad_id,dataset_info, rhyme_info, device, verbose=True):

    checkpoint = torch.load(iambic_ckpt, map_location=device)
    model_args = checkpoint['args']
    iambic_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    iambic_model.load_state_dict(checkpoint['state_dict'])
    iambic_model = iambic_model.to(device)
    iambic_model.eval()
    if verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(iambic_ckpt, checkpoint['epoch']))

    # with open(rhyme_info, 'rb') as rf:
    #     rhyme_info = pickle.load(rf)
    checkpoint = torch.load(rhyme_ckpt, map_location=device)
    model_args = checkpoint['args']
    rhyme_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word), rhyme_group_size=len(rhyme_info.index2rhyme_group), verbose=verbose) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    rhyme_model.load_state_dict(checkpoint['state_dict'])
    rhyme_model = rhyme_model.to(device)
    rhyme_model.eval()
    if verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(rhyme_ckpt, checkpoint['epoch']))
    
    checkpoint = torch.load(newline_ckpt, map_location=device)
    model_args = checkpoint['args']
    newline_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    newline_model.load_state_dict(checkpoint['state_dict'])
    newline_model = newline_model.to(device)
    newline_model.eval()
    if verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(newline_ckpt, checkpoint['epoch']))
    return iambic_model,rhyme_model,newline_model 
from collections import defaultdict
import string,math


def print_size(inp):
    print(inp.size())


def predict_iambic_pentameter_line(model, tokenizer, iambic_model, rhyme_model, newline_model,current_text, current_line_text, rhyme_group, dataset_info, rhyme_info, precondition_topk, postcondition_topk, banned_tokens=POETRY_BANNED_TOKENS, condition_lambda=1.0, device='cuda', length_cutoff=30):

    with torch.no_grad():
        batch_size = 1
        rhyme_group_index = rhyme_info.rhyme_group2index[rhyme_group]
        future_words = torch.LongTensor([rhyme_group_index]).to(device) # 1
        log_probs = torch.Tensor([math.log(rhyme_info.rhyme_group_counts[rhyme_group] / rhyme_info.total_rhyme_groups)]).to(device) # 1

        # assumes initially all same length.
        previous_encoded_text = [tokenizer.encode(it, return_tensors='pt').to(device) for it in [current_text]]
        previous_enc_len = previous_encoded_text[0].shape[1]
        encoded_input = [tokenizer.encode(it, return_tensors='pt').to(device) for it in [current_text + current_line_text]] # batch x seq
        encoded_input = torch.cat(encoded_input, dim=0)  #unsqueeze first dim
        lengths = torch.LongTensor([encoded_input.shape[1]]).to(device)

        line_syllable_count = count_syllables(current_line_text)
        assert line_syllable_count < POETRY_LINE_SYLLABLES # assume we started with less than one full line
        syllables_to_go = POETRY_LINE_SYLLABLES - line_syllable_count

        for _ in range(length_cutoff): # really shouldn't have a line this long anyway
            gpt_logits = model(encoded_input)[0][:, -1, :] # batch x vocab
            gpt_logits[:, banned_tokens] = -1e8
            top_logits, top_indices = gpt_logits.topk(precondition_topk, dim=1) # 1, 200

            new_input_candidates = torch.cat([encoded_input.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2) # batch x topk x seq+1
            expanded_lengths = (lengths + 1).unsqueeze(1).expand(batch_size, precondition_topk) # batch x topk
            expanded_future_words = future_words.unsqueeze(0).unsqueeze(1).expand(batch_size, precondition_topk, -1) # batch x topk x N
            candidate_syllables_to_go = []
            for candidate in new_input_candidates[0]:
                candidate_until_last_word_text = ' '.join(tokenizer.decode(candidate[previous_enc_len:]).split()[:-1])
                candidate_syllables_to_go.append(10 - count_syllables(candidate_until_last_word_text))
                # usually these are all the same, but run them all for correctness. could do more efficiently but it's not too slow anyway.
            expanded_syllables_to_go = torch.LongTensor(candidate_syllables_to_go).to(device).view(1, precondition_topk)



            if condition_lambda == 0:
                iambic_logits = torch.zeros_like(expanded_lengths).float()
            else:
                # truncate prefix because we trained on single lines
                input1 = new_input_candidates[:, :, previous_enc_len:].flatten(0, 1)    # batch,prek,14 =>index batch,prek,add_seq_len 200,1
                input2 = expanded_lengths.flatten(0, 1) - previous_enc_len  # (batch, prek) => batch * prek
                iambic_logits = iambic_model(input1, input2, None, None, None)[:, -1] # batch*topk x seq+1 -> batch*topk
                iambic_logits = iambic_logits.view(batch_size, precondition_topk)
                iambic_logits = iambic_logits - torch.log(1 + torch.exp(iambic_logits))
            
            # score_iambic
                

            if condition_lambda == 0:
                rhyme_logits = torch.zeros_like(expanded_lengths).float()
            else:
                rhyme_logits = rhyme_model(new_input_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    expanded_future_words.flatten(0, 1), # batch*topk x N
                                                    log_probs, # N
                                                    expanded_syllables_to_go.flatten(0, 1)) # batch*topk
                rhyme_logits = rhyme_logits.view(batch_size, precondition_topk, -1) # batch x topk x N
                rhyme_logits = rhyme_logits - torch.log(1 + torch.exp(rhyme_logits)) # batch x topk x N
                rhyme_logits = rhyme_logits.squeeze(2) # batch x topk
                
                

            if condition_lambda == 0:
                newline_logits = torch.zeros_like(expanded_lengths).float()
            else:
                newline_logits = newline_model(new_input_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    expanded_future_words.flatten(0, 1), # batch*topk x N
                                                    log_probs, # N
                                                    expanded_syllables_to_go.flatten(0, 1)) # batch*topk
                newline_logits = newline_logits[:, -1].view(batch_size, precondition_topk, -1) # batch x topk x N
                newline_logits = newline_logits - torch.log(1 + torch.exp(newline_logits)) # batch x topk x N
                newline_logits = newline_logits.squeeze(2) # batch x topk
            
            full_logits = top_logits + condition_lambda * iambic_logits + condition_lambda * rhyme_logits + condition_lambda * newline_logits
            post_logits, post_indices = full_logits.topk(postcondition_topk, dim=1)
            post_probs = F.softmax(post_logits, dim=1)
            index_into_top_indices = post_indices[torch.arange(batch_size).to(post_indices.device), torch.multinomial(post_probs, 1).flatten()] # batch
            next_indices = top_indices[torch.arange(batch_size).to(top_indices.device), index_into_top_indices] # batch
            encoded_input = torch.cat([encoded_input, next_indices.unsqueeze(1)], dim=1) # batch x seq+1
            lengths = lengths + 1
            syllables_to_go = POETRY_LINE_SYLLABLES - count_syllables(tokenizer.decode(encoded_input[0][previous_enc_len:])) # if we get very unlucky with a partial word that the syllable counter doesn't recognize we might end early, but it's unlikely
            if syllables_to_go <= 0 and [tokenizer.decode(s) for s in encoded_input][0][-1] in PHRASE_ENDS:
                break
            if syllables_to_go < 0:
                # encoded_input = encoded_input[:, :-1]
                break

        return [tokenizer.decode(s) for s in encoded_input][0][len(current_text):]


def predict_couplet(gpt_model, gpt_tokenizer, iambic_model, rhyme_model, newline_model, input_text, dataset_info, rhyme_info, precondition_topk, postcondition_topk, condition_lambda=1.0, device='cuda'):
    assert len(input_text) == 1
    current_text = input_text[0]
    current_line_text = ''
    all_lines = [current_text]
    ending_word = current_text.split()[-1].strip(string.punctuation)
    word2rhyme_group = defaultdict(lambda: UNKNOWN_RHYME_GROUP, rhyme_info.word2rhyme_group)
    rhyme_group = word2rhyme_group[ending_word]

    line = predict_iambic_pentameter_line(gpt_model, 
                        gpt_tokenizer, 
                        iambic_model, 
                        rhyme_model, 
                        newline_model,
                        current_text,
                        current_line_text,
                        rhyme_group,
                        dataset_info, 
                        rhyme_info,
                        precondition_topk, 
                        postcondition_topk,
                        condition_lambda=condition_lambda,
                        device=device)
    all_lines.append(line)
    return all_lines

from tqdm import tqdm

def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    with open(args.rhyme_info, 'rb') as rf:
        rhyme_info = pickle.load(rf)
    gpt_model, gpt_tokenizer, gpt_pad_id = setup_gpt(args.model_string, args.device)
    iambic_model,rhyme_model,newline_model = setup_poetry_models(args.iambic_ckpt, args.rhyme_ckpt, args.newline_ckpt, gpt_pad_id,dataset_info, rhyme_info, args.device, verbose=True)

    with open(args.prefix_file, 'r') as rf:
        lines = rf.readlines()
    for line in tqdm(lines, total=len(lines)):
        couplet = predict_couplet(gpt_model, 
                gpt_tokenizer, 
                iambic_model, 
                rhyme_model,
                newline_model,
                [line], 
                dataset_info, 
                rhyme_info,
                args.precondition_topk,
                args.topk, 
                condition_lambda=args.condition_lambda,
                device=args.device)
        assert len(couplet) == 2
        print(couplet[1].strip().replace('\n', ''))
        
from argparse import ArgumentParser
import random
import numpy as np


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--iambic_ckpt', type=str,  default='/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/iambic_predictor/model.pth.tar')
    parser.add_argument('--rhyme_ckpt', type=str,  default='/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/rhyme_predictor/model.pth.tar')
    parser.add_argument('--newline_ckpt', type=str, default='/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/newline_predictor/model.pth.tar')
    parser.add_argument('--dataset_info', type=str,  help='saved dataset info',default='/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/rhyme_predictor/dataset_info')
    parser.add_argument('--rhyme_info', type=str,  help='saved rhyme info',default='/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/rhyme_predictor/rhyme_info')
    parser.add_argument('--model_string', type=str, default='gpt2-medium')

    parser.add_argument('--prefix_file', type=str,  help='file of prefix lines for couplets', default="/export/home/experimental/naacl-2021-fudge-controlled-generation/poetry_data/couplet_prefixes.txt")

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--topk', type=int, default=10, help='consider top k outputs from gpt at each step')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('-device', type=str, default='cuda:0', choices=['cpu', 'cuda:0', 'cuda:1'])
    parser.add_argument('--debug', action='store_true', default=False)

    # /export/home/experimental/naacl-2021-fudge-controlled-generation/poetry_data/couplet_prefixes.txt

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)