task_name="qg"


PYTHONPATH=./ /export/home/miniconda3/bin/python bbfs/run.py  --task $task_name --algo bfs --max_len 25 --beam_size 10 --device cpu --group_size 5 --task_rwd 0.0 --temp_decay 0.1 --heap_sample --heap_top_k 10  --debug

task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task $task_name --algo bfs --max_len 25 --beam_size 10 --device cuda:1 --group_size 5 --task_rwd 0.0 --temp_decay 0.2 --heap_sample --heap_top_k 10   --debug

PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task $task_name --algo bfs --max_len 25 --beam_size 10 --device cuda:0 --group_size 5 --task_rwd 0.0 --temp_decay 0.1 --heap_sample --heap_top_k 15    --debug

PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task $task_name --algo bfs --max_len 25 --beam_size 10 --device cuda:3 --group_size 5 --task_rwd 0.0 --temp_decay 0.2 --heap_sample --heap_top_k 15   --debug

task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task $task_name --algo bfs --max_len 25 --beam_size 10 --device cuda:2 --group_size 5 --task_rwd 0.0 --temp_decay 0.1 --heap_sample --heap_top_k 20   --debug

task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task $task_name --algo bfs --max_len 25 --beam_size 10 --device cuda:1 --group_size 5 --task_rwd 0.0 --temp_decay 0.2 --heap_sample --heap_top_k 20   --debug

# dbs
task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task $task_name --algo bs --max_len 25 --beam_size 10 --num_beam_groups 5 --task_rwd 0.0 --device cuda:3  --debug

# BS
task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task $task_name --algo bs --max_len 25 --beam_size 10 --num_beam_groups 1 --task_rwd 0.0 --device cuda:0  --debug

#sample
task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task $task_name --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --typical_p 0.8 --device cuda:1  --debug

task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task $task_name --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --typical_p 0.9  --device cuda:3  --debug

task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task $task_name --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --typical_p 0.5 --device cuda:3  --debug

task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task $task_name --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --typical_p 0.2 --debug


task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task $task_name --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --top_p 0.9 --debug

task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task $task_name --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --top_p 0.8 --debug

task_name="qg"; PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task $task_name --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --top_p 0.7 --debug