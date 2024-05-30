export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PATH="/opt/conda/bin:$PATH"
# source ~/.bashrc

BasePath=/data/xfbai
MODEL=${BasePath}/data/pretrained-models/llama-7b
# MODEL=/data/model/llama-65b

DataPath=${BasePath}/data/TaskData
DataSetName=stanford-alpaca
DataSetName=pubmedqa
DataSetName=trucated-pubmedqa

export HF_DATASETS_CACHE=${DataPath}/${DataSetName}/.cache

lr=2e-5

OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-7b-GenMode-lr-${lr}-totalbsz128-decay0.1-3epoch
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-7b-GenMode-lr-${lr}-totalbsz256-stage2-nooffload-GC-1Node
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-7b-GenMode-lr-${lr}-totalbsz1024-stage3-nooffload-GC-1Node
OUTPUT_DIR=${BasePath}/output/exp.InstructTuning/Finetune-${DataSetName}-llama-7b-GenMode-lr-${lr}-totalbsz1024-zeropp-stage3-nooffload-GC-1Node

if [ ! -d ${OUTPUT_DIR} ];then
  mkdir -p ${OUTPUT_DIR}
else
  read -p "${OUTPUT_DIR} already exists, delete origin one [y/n]?" yn
  case $yn in
    [Yy]* ) rm -rf ${OUTPUT_DIR}; mkdir -p ${OUTPUT_DIR};;
    [Nn]* ) echo "exiting..."; exit;;
    * ) echo "Please answer yes or no.";;
  esac
fi

MODEL_SIZE=7B
NUM_GPUS=8
# NUM_GPUS=2
BATCH_SIZE_PER_GPU=128
TOTAL_BATCH_SIZE=1024
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed benchmark.py \
    --deepspeed ds_configs/stage3_no_offloading.conf \
    --data_path ${DataPath}/${DataSetName} \
    --model_name_or_path ${MODEL} \
    --tokenizer_name ${MODEL} \
    --use_fast_tokenizer False \
    --conditional_gen False \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --evaluation_strategy "steps" \
    --logging_steps 100 \
    --greater_is_better False \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --ignore_opt_states True \
    --num_train_epochs 3 \
    --logging_first_step True \
    --gradient_checkpointing \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --tf32 True \
    --overwrite_output_dir \
    --preprocessing_num_workers 1 \
    --data_cache_dir ${DataPath}/${DataSetName}/.cache \
    --report_to "tensorboard" 2>&1 | tee ${OUTPUT_DIR}/training.log