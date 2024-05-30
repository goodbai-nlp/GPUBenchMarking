#!/bin/bash
#SBATCH --job-name=xfbai-RLHF     #作业名称
#SBATCH --partition=vip                #选择资源分区
#SBATCH -N 2                           #申请计算节点数
#SBATCH --ntasks-per-node=96           #申请每个节点96核CPU
#SBATCH --gres=gpu:8                   #申请2张GPU卡
#SBATCH -w gpu04,gpu07           		#指定GPU
#SBATCH --output=logs/run-job%j.out         #作业标准输出
#SBATCH --error=logs/run-job%j.err          #作业标准报错信息
hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
module load anaconda3
module load gcc
source activate py3.10torch2.0rlhf
cd /share/home/hubaotian/hbt_user02/mywork/exp.LLMTuning/src
sleep 15h