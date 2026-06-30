#!/bin/bash
cd ..
root='./exp/'
exp_name='census'
latent_file="${root}table/census/sampled_latent.pkl"
train_num=50000

# # 遍历 models_flip 目录下所有以 adult 开头的子文件夹
for model_dir in ${root}train_dnn/multitask_quant_models/adult*; do
    if [ -d "$model_dir" ]; then
        label_file="${model_dir}/labels.npy"
        score_file="${model_dir}/predict_scores.npy"
        
        # 执行命令
        python train_latent_boundary.py \
         --exp_name="$exp_name" \
         --latent_file="$latent_file" \
         --label_file="$label_file" \
         --score_file="$score_file" \
         --train_num=$train_num
    fi
done

exp_name='bank'
latent_file="${root}table/bank/sampled_latent.pkl"
train_num=50000

# # 遍历 models_flip 目录下所有以 adult 开头的子文件夹
for model_dir in ${root}train_dnn/multitask_quant_models/bank*; do
    if [ -d "$model_dir" ]; then
        label_file="${model_dir}/labels.npy"
        score_file="${model_dir}/predict_scores.npy"
        
        # 执行命令
        python train_latent_boundary.py \
         --exp_name="$exp_name" \
         --latent_file="$latent_file" \
         --label_file="$label_file" \
         --score_file="$score_file" \
         --train_num=$train_num
    fi
done

exp_name='credit'
latent_file="${root}table/credit/sampled_latent.pkl"
train_num=50000

# # 遍历 models_flip 目录下所有以 adult 开头的子文件夹
for model_dir in ${root}train_dnn/multitask_quant_models/german*; do
    if [ -d "$model_dir" ]; then
        label_file="${model_dir}/labels.npy"
        score_file="${model_dir}/predict_scores.npy"
        
        # 执行命令
        python train_latent_boundary.py \
         --exp_name="$exp_name" \
         --latent_file="$latent_file" \
         --label_file="$label_file" \
         --score_file="$score_file" \
         --train_num=$train_num
    fi
done

exp_name='compas'
latent_file="${root}table/compas/sampled_latent.pkl"
train_num=50000

# # 遍历 models_flip 目录下所有以 adult 开头的子文件夹
for model_dir in ${root}train_dnn/multitask_quant_models/compas*; do
    if [ -d "$model_dir" ]; then
        label_file="${model_dir}/labels.npy"
        score_file="${model_dir}/predict_scores.npy"
        
        # 执行命令
        python train_latent_boundary.py \
         --exp_name="$exp_name" \
         --latent_file="$latent_file" \
         --label_file="$label_file" \
         --score_file="$score_file" \
         --train_num=$train_num
    fi
done


exp_name='lsac'
latent_file="${root}table/lsac/sampled_latent.pkl"
train_num=50000

# # 遍历 models_flip 目录下所有以 adult 开头的子文件夹
for model_dir in ${root}train_dnn/multitask_quant_models/lsac*; do
    if [ -d "$model_dir" ]; then
        label_file="${model_dir}/labels.npy"
        score_file="${model_dir}/predict_scores.npy"
        
        # 执行命令
        python train_latent_boundary.py \
         --exp_name="$exp_name" \
         --latent_file="$latent_file" \
         --label_file="$label_file" \
         --score_file="$score_file" \
         --train_num=$train_num
    fi
done

# 