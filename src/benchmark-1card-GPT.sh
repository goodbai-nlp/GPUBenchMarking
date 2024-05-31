export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

ProjectDir=$(cd $(dirname $0);cd ..; pwd)
BasePath=/mnt/data/home/usera6k10
#BasePath=/home/ubuntu/gpu-test

ModelCate=gpt2-xl
MODEL=${BasePath}/data/pretrained-models/${ModelCate}

DataPath=${ProjectDir}/data
DataSetName=trucated-pubmedqa

export HF_DATASETS_CACHE=${DataPath}/${DataSetName}/.cache

lr=2e-5
MODEL_SIZE=1.5B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

OUTPUT_DIR=${ProjectDir}/output/exp.InstructTuning/Finetune-${DataSetName}-${ModelCate}-GenMode-${NUM_GPUS}GPU-lr-${lr}-totalbsz${TOTAL_BATCH_SIZE}-decay0.1-3epoch

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

deepspeed benchmark.py \
    --deepspeed ds_configs/stage1_no_offloading.conf \
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
    --data_cache_dir ${DataPath}/${DataSetName}/.cache-gptxl \
    --report_to "tensorboard" 2>&1 | tee ${OUTPUT_DIR}/training.log
