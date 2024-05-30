#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
# export PATH="/opt/conda/bin:$PATH"
# source ~/.bashrc
# ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# echo $ROOT_DIR

MODEL=$1
OUT_PREFIX=$2
DATA=./test-data/test.json
DATA=./test-data/test_0830.json
python -u inference_vllm.py --test_file ${DATA} --model_name_or_path ${MODEL} --num_beams 1 --max_new_tokens 512 --out_prefix ${OUT_PREFIX} --instruction "" --input_key "Question" --prompt_template "one_shot_sim" 2>&1 | tee $MODEL/eval.log
