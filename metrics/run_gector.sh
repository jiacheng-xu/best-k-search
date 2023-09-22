
echo '1: working directory, 2: input file, 3: output file, 4: output counts, 5: venv'
cd $1
# conda init bash
# conda activate run-gector
source $5/bin/activate
/home/jcxu/miniconda3/envs/run-gector/bin/python  predict.py --model_path  roberta_1_gectorv2.th \
                   --input_file $2 --output_file $3 --output_cnt_file $4