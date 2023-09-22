prefix_sample=" bbfs/run.py  --task cg --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0"


PYTHONPATH=./ /export/home/miniconda3/bin/python bbfs/run.py  --task cg --algo bfs --max_len 25 --beam_size 10 --device cpu --group_size 5 --task_rwd 0.0 --temp_decay 0.1 --heap_sample --heap_top_k 10

PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task cg --algo bfs --max_len 25 --beam_size 10 --device cuda:2 --group_size 5 --task_rwd 0.0 --temp_decay 0.2 --heap_sample --heap_top_k 10

PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task cg --algo bfs --max_len 25 --beam_size 10 --device cuda:2 --group_size 5 --task_rwd 0.0 --temp_decay 0.1 --heap_sample --heap_top_k 15

PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task cg --algo bfs --max_len 25 --beam_size 10 --device cuda:2 --group_size 5 --task_rwd 0.0 --temp_decay 0.2 --heap_sample --heap_top_k 15

PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task cg --algo bfs --max_len 25 --beam_size 10 --device cuda:2 --group_size 5 --task_rwd 0.0 --temp_decay 0.1 --heap_sample --heap_top_k 20

PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task cg --algo bfs --max_len 25 --beam_size 10 --device cuda:2 --group_size 5 --task_rwd 0.0 --temp_decay 0.2 --heap_sample --heap_top_k 20

# dbs
PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task cg --algo bs --max_len 25 --beam_size 10 --num_beam_groups 5 --task_rwd 0.0

# BS
PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task cg --algo bs --max_len 25 --beam_size 10 --num_beam_groups 1 --task_rwd 0.0

#sample
PYTHONPATH=./ /export/home/anaconda3/bin/python bbfs/run.py  --task cg --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --typical_p 0.8

PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task cg --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --typical_p 0.9

PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task cg --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --typical_p 0.5
PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task cg --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --typical_p 0.2


PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task cg --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --top_p 0.9
PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task cg --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --top_p 0.8
PYTHONPATH=./ /export/home/anaconda3/bin/python  bbfs/run.py  --task cg --algo sample --max_len 25 --beam_size 10  --task_rwd 0.0 --top_p 0.7