#!/usr/bin/env bash

source /export/home/anaconda3/bin/activate

HEAP=('2' '4' '8')
TEMP=('0.0' '0.5' '1.0')
TASK=('0.5' '0.0')
BASE='fudge/run_bfs.py -device cuda:1 --task topic --prefix_file /export/home/experimental/naacl-2021-fudge-controlled-generation/topic_data/topic_prefixes.txt --dataset_info /export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/topic/future_word_predictor/dataset_info --verbose --condition_lambda 4 --max_len 20 --algo batch_bfs --beam_size 5 --group_size 2 '

for h in "${HEAP[@]}"; do
for t in "${TEMP[@]}"; do
for task in "${TASK[@]}"; do
: '
PYTHONPATH=./ python fudge/run_bfs.py -device cuda:1 --task topic --prefix_file /export/home/experimental/naacl-2021-fudge-controlled-generation/topic_data/topic_prefixes.txt --dataset_info /export/home/experimental/naacl-2021-fudge-controlled-generation/ckpt/topic/future_word_predictor/dataset_info --verbose --condition_lambda 4 --max_len 20 --algo batch_bfs --beam_size 5 --group_size 2 --heap_top_k  "$h" --temp_decay "$t" --task_rwd "$task"
'
echo $h
echo $t
echo "PYTHONPATH=./ python fudge/eval_topic_metrics.py --log_file topic_batch_bfs_20_5_${t}_${h}_${task}_new_topic_preds.log"
PYTHONPATH=./ python fudge/eval_topic_metrics.py --log_file topic_batch_bfs_20_5_2_${t}_${h}_${task}_new_topic_preds.log 
# PYTHONPATH=./ python fudge/eval_topic_metrics.py --log_file "topic_batch_bfs_20_5_$t_$h_$task_new_topic_preds.log"
done
done
done