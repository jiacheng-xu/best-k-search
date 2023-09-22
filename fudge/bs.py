from functools import total_ordering
from fudge.data_control import load_topic_data
from fudge.run_poetry import setup_poetry_models
from fudge.score_poem import NewLine,Iambic,Rhyme
from fudge.score_topic import ScoreTopic, load_topic_model

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


from fudge.util import *
from dec.bfs import assemble_pad, assemble_pad_plus, init_search, vanilla_heap_pop
from dec.viz_center import proc
from dec.util import BeamNode, Scorer


def step_bs(
    model,
    tokenizer,
    device,
    doc_input_token,
    expansion_list, # replace heap
    default_scorer,
    task_scores,
    max_len,
    beam_size,
    k_best,
    precondition_topk:int=200,
    verbose=True,
):
    
    # BS
    # expansion_list, cnt = get_expansion_candidates(grp_size, heap)
    
    finished_hypos = []
    new_expansion_list = []
    # print expansion_list
    # for it in expansion_list:
    #     toks = it.get_token_idx()
    #     print(tokenizer.decode(toks))
    # print('-' * 30)
    
    doc_input_token_len = len(doc_input_token) # length of input (shared, encoding side)
    (
        logits, # b x v
        concat_input_tensor,    # b x e_d, padded input of enc+dec
        dec_input_tensor,       # b x d, padded input of dec
        batch_size,             
        padding_pos,            # list, size = b, the start position of non pad token
        enc_dec_length,         # list, size = b, the length of e+d excluding PAD
    ) = gpt_decode(model, device, expansion_list, doc_input_token, tokenizer)
    
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
    
    lengths = torch.LongTensor(enc_dec_length).to(device)
    expanded_lengths = unsqueeze_expand_sec_dim(lengths + 1, precondition_topk)     ## expandlen is original + 1
    
    full_logits = 0
    aux_logits = []
    for score_module in task_scores:
        if score_module.name == 'topic':
            score_module.ext_feat(max_len, lengths-doc_input_token_len)
            task_logits = score_module.score(extend_concat_input_tensor, expanded_lengths)
            aux_logits.append([task_logits, score_module.condition_lambda, score_module.name])
            full_logits += task_logits * score_module.condition_lambda
        else:
            pass
    full_logits += top_logits

    post_logits, post_indices = full_logits.topk(k_best, dim=1)
    post_probs = F.softmax(post_logits, dim=1)
    post_probs_list = post_probs.tolist()
    # index_into_top_indices = post_indices[torch.arange(batch_size).to(post_indices.device), torch.multinomial(post_probs, 1).flatten()] # batch
    index_into_top_indices = post_indices[
        torch.arange(batch_size).to(post_indices.device), :k_best
    ]  # batch, k_best

    for b in range(batch_size):
        next_indices = indices[b][index_into_top_indices[b]].cpu().tolist() # k best options
        cur_seq = concat_input_tensor[b].cpu().tolist()  # seq_len

        if verbose and random.random() > 0.990:
            logging.info(
                f"Prefix: {tokenizer.decode(doc_input_token)}"
                + f"///{tokenizer.decode(dec_input_tensor[b].cpu())}"
            )

            visualize_logits(top_logits, indices,"RAW", tokenizer,b,10)
            for aux in aux_logits:
                visualize_logits(aux[0], indices, aux[2], tokenizer, b, 10)
            visualize_logits(full_logits, indices, "full", tokenizer, b, 10)
            print('')
        for rank in range(k_best):
            v = post_probs_list[b][rank]
            tok = next_indices[rank]
            tmp_state = BeamNode(
                prob=v,
                token_idx=tok,
                token_str=tokenizer.decode(tok),
                prev=[expansion_list[b]],
                prev_score=[math.log(v)],
                glob_step=proc(),
            )
            # book.add_child()
            model_score = math.log(v)  # 
            # model_score = default_scorer.get_model_score(tmp_state)  # stateless?
            if tmp_state.length >= max_len:

                logging.info(
                    f"Ending: {tokenizer.decode(cur_seq[doc_input_token_len + padding_pos[b]:] + [next_indices[rank]])}"
                )
                finished_hypos.append([model_score, tmp_state])
                continue
            if  tmp_state.token_idx == 50256 or v < 0.1:
                continue
            new_expansion_list.append([model_score, tmp_state]) # make sure the model score is larger better!
    
    # sort new_expansion_list
    # larger better
    sorted_new_expansion_list = sorted(new_expansion_list, key=lambda cand: cand[0], reverse=True)
    expansion_list = sorted_new_expansion_list[:beam_size]
    expansion_list = [ x[1] for x in expansion_list]
    if finished_hypos:
        sorted_finish = sorted(finished_hypos, key=lambda cand: cand[0], reverse=True)
        finish = sorted_finish[:beam_size]
        finish = [ x[1] for x in finish]
    else:
        finish = []
    return finish, batch_size, expansion_list


def beam_search(
    model,
    tokenizer,
    default_scorer,
    task_scores,
    doc_input,
    dec_prefix="",
    precondition_topk: int = 200,
    k_best: Optional[int] = 15,
    max_len: Optional[int] = 30,
    beam_size: Optional[int] = 5,
    comp_budget: Optional[int] = 300,
    device=None,
    eos_idx: int = 50256,
):
    ncalls = 0
    
    finished_hypos = []

    tok_input: List[int] = tokenizer.encode(doc_input)
    tok_dec_prefix: List[int] = tokenizer.encode(dec_prefix)
    heap = []  # nodes at the frontier of search
    init_search(tokenizer, tok_dec_prefix, heap, eos_idx=eos_idx)
    _, seed = heapq.heappop(heap)

    expansion_list = [seed]

    
    # init_seed only has the decoding prefix
    with torch.no_grad():
        while ncalls < comp_budget and expansion_list:
            completed_hyps, added_num_calls,expansion_list  = step_bs(model,tokenizer,device,tok_input,expansion_list,default_scorer,task_scores,max_len,beam_size,k_best,precondition_topk)
            ncalls += added_num_calls
            if completed_hyps:
                finished_hypos += completed_hyps
    return finished_hypos


def main(args):
    args.log_file = define_log_file_name(args.log_file, args)
    
    with open(args.dataset_info, "rb") as rf:
        dataset_info = pickle.load(rf)
    
    if args.debug:
        args.model_string = "distilgpt2"
        logging.info(f"Change model to distilgpt2 ...")

    model, tokenizer, gpt_pad_id = setup_gpt(args.model_string, args.device)
    task_scores = []
    if args.task == 'topic':
        logging.info("Setup topic conditioning model.")
        cond_model = load_topic_model(args, gpt_pad_id, dataset_info)
        topic_scorer = ScoreTopic(cond_model, args.device, args.condition_lambda)

        task_scores.append(topic_scorer)
        logging.info("Setup topic data.")
        dict_data = load_topic_data(args, dataset_info)
        
    elif args.task == 'poetry':
        with open(args.rhyme_info, "rb") as rf:
            rhyme_info = pickle.load(rf)
        iambic_model, rhyme_model, newline_model = setup_poetry_models(
        args.iambic_ckpt,
        args.rhyme_ckpt,
        args.newline_ckpt,
        gpt_pad_id,
        dataset_info,
        rhyme_info,
        args.device,
        verbose=True,
    )
        scorer_newline = NewLine(newline_model, args.device)
        scorer_rhyme = Rhyme(rhyme_model, device=args.device, rhyme_info=rhyme_info)
        scorer_iambic = Iambic(iambic_model, args.device)
        task_scores.append(scorer_newline)
        task_scores.append(scorer_rhyme)
        task_scores.append(scorer_iambic)
    else:
        raise NotImplementedError
    default_scorer = Scorer(len_factor=args.len_factor)

    logging.info(f"Reading prefix {args.prefix_file}")
    with open(args.prefix_file, "r") as rf:
        lines = rf.readlines()
    with open(args.pred_file, "w") as wf:
        pass
    predictions = []
    
    total_len = len(list(dict_data.values())[0])
    for idx in tqdm(range(total_len), total=total_len):
        if args.task == 'topic':
            input_text = dict_data['input_texts'][idx]
            condition_words = dict_data['conditions'][idx]
            comp_budget = args.max_len * args.beam_size
            task_scores[0].prepare(condition_words, dataset_info)
            category = dict_data['categories'][idx]
            
            finished_hypos  = beam_search(
            model,
            tokenizer,
            default_scorer,task_scores,input_text,
            "",args.precondition_topk,
            args.topk,max_len=args.max_len,
            beam_size=args.beam_size,
            comp_budget=comp_budget,device=args.device,
            eos_idx=tokenizer.eos_token_id)
            
            cur_pred = []
            print(len(finished_hypos))
            assert len(finished_hypos) == args.beam_size
            for hypo in finished_hypos:
                hypo_tok_id = hypo.get_token_idx()
                output_string = tokenizer.decode(hypo_tok_id)
                logging.info(f"Input|gen: {input_text}|{output_string}")
                cur_pred.append(output_string)
            predictions.append((input_text, category, cur_pred))
            
        elif args.task == 'poetry':
            raise NotImplementedError
            for line in tqdm(lines, total=len(lines)):
                current_text = line
                current_line_text = ""
                all_lines = [current_text]
                logging.info(f"\n {current_text}")
                dec_prefix = tokenizer.encode(current_text)
                comp_budget = len(dec_prefix) * args.beam_size
                logging.info(f"Comp budget: {comp_budget}")

                # scorer_rhyme.add_rhyme_group(current_text)
                finished_hypos, heap  = f_best_first_search(
                    model,
                    tokenizer,
                    default_scorer,
                    task_scores,
                    doc_input,
                    "",
                    dataset_info,
                    args.precondition_topk,
                    args.topk,
                max_len=args.max_len,
                    grp_size=args.group_size,
                    comp_budget=comp_budget,
                    device=args.device,
                    eos_idx=tokenizer.eos_token_id)

                # only use the top1 for now
                assert len(finished_hypos) > 0
                top1_hypo_tok_id = finished_hypos[0].get_token_idx()
                print("--")
                for hypo in finished_hypos:
                    output_string = tokenizer.decode(hypo.get_token_idx())
                    print(output_string.strip().replace("\n", ""))
                print("--")
                output_string = tokenizer.decode(top1_hypo_tok_id)
                # assert len(couplet) == 2
                # logging.info(couplet[1].strip().replace('\n', ''))
                output_str_post = output_string.strip().replace("\n", "")
                predictions.append(output_str_post)
            with open(args.pred_file, "a") as wf:
                wf.write(output_string + "\n")
    logging.info(f"Done. Writting. ")
    with open(args.log_file, 'w') as wf:
        writer = csv.DictWriter(wf, fieldnames=['category', 'input_text', 'generation'])
        writer.writeheader()
        for cr_group in predictions:
            for cr in cr_group[2]:
                writer.writerow({'category': cr_group[1], 'input_text': cr_group[0], 'generation': cr})


if __name__ == "__main__":
    parser = ArgumentParser()

    # POEM
    parser.add_argument(
        "--iambic_ckpt",
        type=str,
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/iambic_predictor/model.pth.tar",
    )
    parser.add_argument(
        "--rhyme_ckpt",
        type=str,
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/rhyme_predictor/model.pth.tar",
    )
    parser.add_argument(
        "--newline_ckpt",
        type=str,
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/newline_predictor/model.pth.tar",
    )

    parser.add_argument(
        "--rhyme_info",
        type=str,
        help="saved rhyme info",
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/rhyme_predictor/rhyme_info",
    )


    ##
    
    # TOPIC
    parser.add_argument('--condition_words', type=str, default=None, help='word(s) to optimize for')
    parser.add_argument('--condition_file', type=str, default=None, help='file of inputs and conditions')

    parser.add_argument('--wordlist_dir', type=str, default='/export/home/experimental/naacl-2021-fudge-controlled-generation/topic_data/wordlists', help='dir of bow wordlists for categories')
    parser.add_argument('--topic_ckpt', type=str, default="/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/topic/future_word_predictor/model.pth.tar")
    parser.add_argument('--log_file', type=str, default='topic_preds.log', help='file to write outputs to (csv format)')
    ##
    
    parser.add_argument('--dataset_info', type=str, default='/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/topic/future_word_predictor/dataset_info', help='saved dataset info', choices=["/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/poetry/rhyme_predictor/dataset_info", '/export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/topic/future_word_predictor/dataset_info'])
    parser.add_argument(
        "--prefix_file",
        type=str,
        help="file of prefix lines for couplets",
        default="/export/home/experimental/naacl-2021-fudge-controlled-generation/poetry_data/couplet_prefixes.txt", choices=["/export/home/experimental/naacl-2021-fudge-controlled-generation/poetry_data/couplet_prefixes.txt", '/export/home/experimental/naacl-2021-fudge-controlled-generation/topic_data/topic_prefixes.txt']
    )

    parser.add_argument("--model_string", type=str, default="gpt2-medium")

    parser.add_argument(
        "--precondition_topk",type=int,default=200,
        help="consider top k outputs from gpt at each step before conditioning and re-pruning",
    )
    parser.add_argument(
        "--topk",type=int,default=10,help="consider top k outputs from gpt at each step",
    )
    parser.add_argument(
        "--condition_lambda",
        type=float,
        default=1.0,
        help="lambda weight on conditioning model",
    )

    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "-device", type=str, default="cuda:0", choices=["cpu", "cuda:0", "cuda:1"]
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)

    parser.add_argument('--task', default='topic', choices=['topic',' poetry'])
    # parser.add_argument('-dataset', default='xsum', choices=['xsum','default'])
    parser.add_argument("--algo", default="bs", choices=["bs", "bfs", "batch_bfs"])

    parser.add_argument("--len_factor", default=1., type=float)
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--pred_file", default="fudge_topic.txt")
    # /export/home/experimental/naacl-2021-fudge-controlled-generation/poetry_data/couplet_prefixes.txt

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)