from ast import parse
from bbfs.algo_baseline import BeamSearch
from cg.data import load_cg, prepare_output_cg, write_to_json
from cg.scorer import RewardTokenMatch
from fudge.analyze_output import SearchOutput, handle_output


from fudge.util import *
from dec.util import  Scorer
from bbfs.algo_bfs import BestFirstSearch
from bbfs.constant import TASKS
from gec.data import load_gec

from qg.data import load_drop, load_quoref, load_xsum
from qg.data import load_squad


def setup_model(model_name, model_type, device):
    
    if 'mbart' in model_name:
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

        tokenizer.src_lang = "en_XX"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    if model_type == "AutoModelForSeq2SeqLM":
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif model_type == 'AutoModelForCausalLM':
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_type == 'MBartForConditionalGeneration':
        model = MBartForConditionalGeneration.from_pretrained(model_name)
    else:
        raise NotImplementedError
    model = model.to(device)
    if not tokenizer.pad_token_id:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        tokenizer.pad_token = tokenizer.eos_token # to avoid an error
        model.config.pad_token_id = model.config.eos_token_id
    # gpt_pad_id = gpt_tokenizer.encode(PAD_TOKEN)[0]
    
    # when generating, we will use the logits of right-most token to predict the next token
    # so the padding should be on the left
    # tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer

import os

def main(args):
    
    args.log_file = define_log_file_name(args.log_file, args)
    log_file_name = os.path.join(args.output_dir, args.task, args.log_file)
    if not os.path.exists(os.path.join(args.output_dir, args.task)):
        os.makedirs(os.path.join(args.output_dir, args.task))
    if (not args.demo) and os.path.isfile(log_file_name):
        logging.info(f"{log_file_name}exists! QUIT." )
        exit()
    logging.info(log_file_name)
    # set up model according the task
    model, tokenizer = setup_model(TASKS[args.task]['model'], TASKS[args.task]['type'],args.device )

    task_scores = {}
    if args.task == 'cg':
        dataset = load_cg(split=args.split, debug=args.debug,constrain=args.constraint, tokenizer=tokenizer)
        # reward_scorer = RewardTokenMatch(args.task_rwd)
        # task_scores['rwd'] = reward_scorer
    elif args.task == 'squad':
        dataset = load_squad(split=args.split, debug=args.debug, constrain=args.constraint, tokenizer=tokenizer)
    elif args.task == 'drop':
        dataset = load_drop(split=args.split, debug=args.debug, constrain=args.constraint, tokenizer=tokenizer)
    elif args.task == 'quoref':
        dataset = load_quoref(split=args.split, debug=args.debug)
    elif args.task in ['xsum-t5','xsum-bart', 'xsum-peg'] :
        dataset = load_xsum(split=args.split, debug=args.debug, constrain=args.constraint, tokenizer=tokenizer)
    elif args.task == 'gec':
        dataset = load_gec(tokenizer=tokenizer, split=args.split, debug=args.debug)
    elif args.task in ['opus','mt','mbart', 'enfr','ende']:
        # from mt.data import load_mt
        from mt.data_multiref import load_mt
        # dataset = load_mt(tokenizer=tokenizer)
        dataset = load_mt(args.task, debug=args.debug)
    else:
        raise NotImplementedError

    start_token = TASKS[args.task]['start']
    end_token_list = TASKS[args.task]['end']
    end_token_idx_list = [tokenizer.convert_tokens_to_ids(x) for x in end_token_list]
    
    default_scorer = Scorer(len_factor=args.len_factor) # default model score

    # write_output_file_header(log_file_name, TASKS[args.task]['field'])
    # write_output_file_header(log_file_name, TASKS[args.task]['field'], prefix='ref_')
    
    predictions = []    # collect for meta data
    if args.algo == 'bfs':
        # loading search algo
        bfs_algo = BestFirstSearch(model, tokenizer, args.device, max_len=args.max_len, k_best=args.topk, grp_size=args.group_size, use_heap_sample=args.heap_sample, temp_decay=args.temp_decay, heap_top_k=args.heap_top_k, max_heap_size=1000, end_token_idx_list=end_token_idx_list,min_len=TASKS[args.task]['min_len'], task_rwd=args.task_rwd, threshold=args.threshold, )
        func_search = bfs_algo.f_best_first_search
        algo_instance = bfs_algo
    elif args.algo =='bs':
        bs_algo = BeamSearch(model, tokenizer, args.device, args.max_len, args.beam_size, task_rwd=args.task_rwd, do_sample=False,typical_p=args.typical_p, top_p=args.top_p, num_beam_groups=args.num_beam_groups, diversity_penalty=args.diversity_penalty)
        algo_instance = bs_algo
        if args.task_rwd > 0:
            func_search = bs_algo.const_beam_search
        else:
            func_search = bs_algo.f_beam_search
    elif args.algo =='sample':
        assert args.task_rwd == 0.0
        bs_algo = BeamSearch(model, tokenizer, args.device, args.max_len, args.beam_size,task_rwd=args.task_rwd, do_sample=True,typical_p=args.typical_p, top_p=args.top_p)
        func_search = bs_algo.f_beam_search
        algo_instance = bs_algo
    elif args.algo == 'sample1':
        bs_algo = BeamSearch(model, tokenizer, args.device, args.max_len, args.beam_size,task_rwd=args.task_rwd, do_sample=True,typical_p=args.typical_p, top_p=args.top_p)
        func_search = bs_algo.f_sample
        algo_instance = bs_algo

    total_len = len(dataset)
    comp_budget = args.max_len * args.beam_size
    
    for idx in tqdm(range(total_len), total=total_len):
        if args.task == 'cg':
            concepts = dataset[idx]['concepts']   # list
            if args.demo:
                input_text = input('Type your input:')
            else:
                input_text = dataset[idx]['input']
            tgt_num_gen = len(dataset[idx]['ref'])
            if args.task_rwd > 0:
                tgts = dataset[idx]['const']
                # we are going to generate n different things
                for jdx in range(len(tgts)):
                    current_tgt = tgts[jdx]
                    # if current_tgt != [216,19]:
                    #     continue
                    finished_hypos, heap = func_search(default_scorer,task_scores,input_text, start_token, precondition_topk=args.precondition_topk, comp_budget=comp_budget, force_words_ids=current_tgt, verbose=args.verbose)
                    search_outputs:SearchOutput = handle_output(finished_hypos, input_text, tokenizer, tgt_num_gen )
                    wt_data = prepare_output_cg( dataset[idx], search_outputs )
                    wt_data['tgt_token_idx'] = current_tgt
                    wt_data['tgt_token'] = tokenizer.decode(current_tgt)
                    write_to_json(log_file_name, wt_data)
            else:
                finished_hypos, heap = func_search(default_scorer,task_scores,input_text, start_token, args.precondition_topk, comp_budget=comp_budget)
                search_outputs:SearchOutput = handle_output(finished_hypos,  input_text, tokenizer, tgt_num_gen )
                wt_data = prepare_output_cg( dataset[idx], search_outputs )
                write_to_json(log_file_name, wt_data)
        elif args.task in ['squad', 'drop', 'quoref','xsum-bart', 'xsum-peg', 'xsum-t5']:
            input_text = dataset[idx]['input']
            tgt_num_gen = args.beam_size

            if args.task_rwd > 0:
                tgts = dataset[idx]['const']
                # we are going to generate n different things
                for jdx in range(len(tgts)):
                    current_tgt = tgts[jdx]
                    # if current_tgt != [216,19]:
                    #     continue
                    finished_hypos, heap = func_search(default_scorer,task_scores,input_text, start_token, precondition_topk=args.precondition_topk, comp_budget=comp_budget, force_words_ids=current_tgt)
                    search_outputs:SearchOutput = handle_output(finished_hypos, input_text, tokenizer, tgt_num_gen )
                    wt_data = prepare_output_cg( dataset[idx], search_outputs )
                    wt_data['tgt_token_idx'] = current_tgt
                    wt_data['tgt_token'] = tokenizer.decode(current_tgt)
                    write_to_json(log_file_name, wt_data)
            else:
                finished_hypos, heap = func_search(default_scorer,task_scores,input_text, start_token, args.precondition_topk, comp_budget=comp_budget, verbose=args.verbose)
                search_outputs:SearchOutput = handle_output(finished_hypos,  input_text, tokenizer, tgt_num_gen )
                wt_data = prepare_output_cg( dataset[idx], search_outputs )
                write_to_json(log_file_name, wt_data)

        elif args.task == 'opus' or args.task == 'mbart' or args.task in ['enfr', 'ende']:
            input_text = dataset[idx]['input']
            input_text_len = len(tokenizer(input_text)['input_ids'])
            refs = dataset[idx]['ref']
            

            # tgts = dataset[idx]['const']
            tgt_num_gen = len(refs)
            algo_instance.max_len = int(input_text_len * 1.5)
            comp_budget = algo_instance.max_len * args.beam_size
            if args.task_rwd > 0:
                raise NotImplementedError
                if len(tgts) > 1:
                    print('Skip more than 1 constraint.')
                    continue
                else:
                    tgts = tgts[0][0]
                current_tgt = tgts
                finished_hypos, heap = func_search(default_scorer,task_scores,input_text, start_token, precondition_topk=args.precondition_topk, comp_budget=comp_budget, force_words_ids=current_tgt)
                search_outputs:SearchOutput = handle_output(finished_hypos, input_text, tokenizer, tgt_num_gen )
                wt_data = prepare_output_cg( dataset[idx], search_outputs )
                wt_data['tgt_token_idx'] = current_tgt
                wt_data['tgt_token'] = tokenizer.decode(current_tgt)
                write_to_json(log_file_name, wt_data)
            else:
                
                if args.task == 'enfr' or args.task == 'rank_enfr':
                    dec_prefix = 250008
                elif args.task == 'ende' or args.task == 'rank_ende':
                    dec_prefix = 250003
                else:
                    raise NotImplementedError
                finished_hypos, heap = func_search(default_scorer,task_scores,input_text, start_token, args.precondition_topk, comp_budget=comp_budget, force_words_ids=dec_prefix)
                search_outputs:SearchOutput = handle_output(finished_hypos,  input_text, tokenizer, tgt_num_gen )
                wt_data = prepare_output_cg( dataset[idx], search_outputs )
                write_to_json(log_file_name, wt_data)

            
    logging.info(f"Done. Writting. ")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--log_file", type=str,default='output.json')
    parser.add_argument("--split",type=str,default='validation')
    parser.add_argument("--rc_data_name", type=str, default='', choices=['','squad_v2','quoref'])
    parser.add_argument(
        "--precondition_topk",type=int,default=200,
        help="consider top k outputs from gpt at each step before conditioning and re-pruning",
    )
    parser.add_argument(
        "--topk",type=int,default=10,help="consider top k outputs at each step after conditioning and pruning",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cuda:0"
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)

    parser.add_argument("--algo", default="bs", choices=["bs", "bfs", "batch_bfs","sample", "sample1"])
    parser.add_argument('--task', default='cg', choices=['cg', 'qg', 'e2e', 'mt', 'squad','quoref', 'opus','mbart','drop', 'xsum-t5','xsum-peg','xsum-bart', 'enfr','ende'], help='tasks to choose from. ')
    
    parser.add_argument('--demo', action="store_true", default=False,  help="Demo mode asks for input, ignores existing file.")

    parser.add_argument('--constraint', action="store_true", default=False,  help="Whether to apply constraints to decoding.")
    parser.add_argument("--task_rwd",default=0.0, type=float,help='add constraints. 0.0 means off.')
    
    parser.add_argument("--threshold", default=0.05, type=float)
    parser.add_argument("--len_factor", default=1., type=float)
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--beam_size", type=int, default=3, help='Effective beam size.')
    
    parser.add_argument("--typical_p", type=float, default=None)
    parser.add_argument("--top_p",type=float, default=1.0)
    parser.add_argument("--num_beam_groups", type=int, default=1)
    parser.add_argument("--diversity_penalty",default=0.0,type=float)
    
    # start of BFS args
    parser.add_argument("--group_size", type=int, default=2, help='Group size for batch bfs.')
    
    parser.add_argument("--temp_decay",default=0.0,type=float, help="")
    parser.add_argument("--heap_sample", default=False, action='store_true')
    parser.add_argument("--heap_top_k", default=5, type=int, help='Sample from the heap. Typically set to be double the group size. ')
    parser.add_argument("--max_heap_size", default=500, type=int)
    # parser.add_argument("--score", default="last", choices=["avg", "sum", "last"])
    # End of BFS args
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)