WORK_DIR="/export/home/experimental/CommonGen"
INPUT_FILE="$WORK_DIR"/dataset/final_data/commongen/commongen.dev.src_alpha.txt
TRUTH_FILE="$WORK_DIR"/dataset/final_data/commongen/commongen.dev.tgt.txt
PRED_FILE=$1

echo ${INPUT_FILE} 
echo ${TRUTH_FILE}
echo ${PRED_FILE}
which python


# echo "Start running ROUGE"

# cd "$WORK_DIR"/methods/unilm_based
# /opt/conda/envs/unilm_env/bin/python python unilm/src/gigaword/eval.py --pred ${PRED_FILE}   --gold ${TRUTH_FILE} --perl

echo "Start running ROUGE"
echo "BLEU/METER/CIDER/SPICE"
cd "$WORK_DIR"/evaluation/Traditional/eval_metrics/
/opt/conda/envs/coco_score/bin/python eval.py --key_file ${INPUT_FILE} --gts_file ${TRUTH_FILE} --res_file ${PRED_FILE}

echo "Start running coverage"
echo "Coverage"
cd "$WORK_DIR"/evaluation/PivotScore
/export/home/anaconda3/envs/pivot_score/bin/python evaluate.py --pred ${PRED_FILE}   --ref ${TRUTH_FILE} --cs ${INPUT_FILE} --cs_str ../../dataset/final_data/commongen/commongen.dev.cs_str.txt

