

from fudge.analyze_output import SearchOutput, handle_output
from fudge.data_control import load_topic_data
from fudge.run_poetry import setup_poetry_models
from fudge.score_poem import NewLine,Iambic,Rhyme
from fudge.score_topic import RewardTopic, ScoreTopic, load_topic_model

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from fudge.util import *
from dec.util import  Scorer
from fudge.bfs import *

GPT_TASKS = ['topic', 'poetry']
MT_TASKS = ['mt']
OUTPUT_FIELDS = {
    'topic':  ['category', 'input_text', 'generation', 'num_hypo'] ,
    
}

def main(args):
    args.log_file = define_log_file_name(args.log_file, args)
    
    if args.debug:
        args.model_string = "distilgpt2"
        logging.info(f"Change model to distilgpt2 ...")
    if args.task in GPT_TASKS:
        model, tokenizer, gpt_pad_id = setup_gpt(args.model_string, args.device)
    elif args.task in MT_TASKS:
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    task_scores = {}
    
    if args.task == 'topic':
        with open(args.dataset_info, "rb") as rf:
            dataset_info = pickle.load(rf)
        logging.info("Setup topic conditioning model.")
        cond_model = load_topic_model(args, gpt_pad_id, dataset_info)
        topic_scorer = ScoreTopic(cond_model, args.device, args.condition_lambda)
        task_scores['cond_topic'] = topic_scorer

        logging.info("Setup topic data.")
        dict_data = load_topic_data(args, dataset_info)
        reward_scorer = RewardTopic(args.task_rwd)  # load it anyway. If task_rwd = 0, it won't be used. 
        task_scores['rwd_topic'] = reward_scorer

    elif args.task == 'poetry':
        with open(args.dataset_info, "rb") as rf:
            dataset_info = pickle.load(rf)
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
    
    default_scorer = Scorer(len_factor=args.len_factor) # default model score

    if args.task in GPT_TASKS:
        logging.info(f"Reading prefix {args.prefix_file}")
        with open(args.prefix_file, "r") as rf:
            lines = rf.readlines()
    elif args.task in MT_TASKS:
        raise NotImplementedError
    
    write_output_file_header(args.log_file, OUTPUT_FIELDS[args.task])
    
    predictions = []    # collect for meta data
    
    # loading search algo
    bfs_algo = BestFirstSearch(model, tokenizer, args.device,max_len=args.max_len,k_best=args.topk, grp_size=args.group_size, use_heap_sample=True, temp_decay=args.temp_decay,heap_top_k=args.heap_top_k, max_heap_size=1000)
    
    total_len = len(list(dict_data.values())[0] )
    for idx in tqdm(range(total_len), total=total_len):
        logging.info(f"index:{idx}")
        if args.task == 'topic':
            input_text = dict_data['input_texts'][idx]
            condition_words = dict_data['conditions'][idx]
            comp_budget = args.max_len * args.beam_size
            
            task_scores['cond_topic'].prepare(condition_words, dataset_info)
            
            category = dict_data['categories'][idx]
            task_scores['rwd_topic'].set_category(category)
            
            finished_hypos, heap = bfs_algo.f_best_first_search(default_scorer,task_scores,input_text,
            "", args.precondition_topk, comp_budget=comp_budget, eos_idx=tokenizer.eos_token_id)
            
            search_output:SearchOutput = handle_output(finished_hypos,heap, input_text,args.beam_size, tokenizer)

            predictions.append( (input_text, category, search_output.rt_hypos) )
            write_topic_output_line(args.log_file, OUTPUT_FIELDS[args.task], (input_text, category, search_output.rt_hypos, ),search_output.total_hypos_num)
        
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
    parser.add_argument('--log_file', type=str, default='new_topic_preds.log', help='file to write outputs to (csv format)')
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

    parser.add_argument('--task', default='topic', choices=['topic',' poetry', 'e2e', 'mt'])

    parser.add_argument("--algo", default="bs", choices=["bs", "bfs", "batch_bfs"])
    parser.add_argument("--len_factor", default=1., type=float)
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--beam_size", type=int, default=3, help='Effective beam size.')
    parser.add_argument("--group_size", type=int, default=2, help='Group size for batch bfs.')
    

    # start of BFS args
    parser.add_argument("--task_rwd",default=0.0, type=float,help='use downstream task reward as guidance. 0.0 means off.')

    parser.add_argument("--temp_decay",default=0.0,type=float, help="")
    parser.add_argument("--heap_sample", default=False)
    parser.add_argument("--heap_top_k", default=5, type=int)
    parser.add_argument("--max_heap_size", default=500, type=int)
    
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)