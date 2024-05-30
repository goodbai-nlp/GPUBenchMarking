#!/bin/bash
#SBATCH --job-name=xfbai-Graph2text     #作业名称
#SBATCH --partition=q_intel_share*      #选择资源分区
#SBATCH -N 1                            #申请计算节点数
#SBATCH --ntasks-per-node=32            #申请每个节点32核CPU
#SBATCH --gres=gpu:4                    #申请4张GPU卡
#SBATCH -w a10002           		       #指定GPU
#SBATCH --output=logs/run-job%j.out         #作业标准输出
#SBATCH --error=logs/run-job%j.err          #作业标准报错信息
hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
module load anaconda3
module load gcc
source activate py3.8torch2.0vllm
cd $SLURM_SUBMIT_DIR
sleep 15h
