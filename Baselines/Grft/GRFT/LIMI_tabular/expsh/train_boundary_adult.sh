#!/bin/bash
cd ..
root='./exp/'
exp_name='census'
latent_file="${root}table/census/sampled_latent.pkl"
train_num=50000

# # # 遍历 models_flip 目录下所有以 adult 开头的子文件夹
# for model_dir in ${root}train_dnn/models_flip/adult*; do
#     if [ -d "$model_dir" ]; then
#         label_file="${model_dir}/labels.npy"
#         score_file="${model_dir}/predict_scores.npy"
        
#         # 执行命令
#         python train_latent_boundary.py \
#          --exp_name="$exp_name" \
#          --latent_file="$latent_file" \
#          --label_file="$label_file" \
#          --score_file="$score_file" \
#          --train_num=$train_num
#     fi
# done

# for model_dir in ${root}train_dnn/models_adv/bank*; do
#     if [ -d "$model_dir" ]; then
#         label_file="${model_dir}/labels.npy"
#         score_file="${model_dir}/predict_scores.npy"
        
#         # 执行命令
#         python train_latent_boundary.py \
#          --exp_name="$exp_name" \
#          --latent_file="$latent_file" \
#          --label_file="$label_file" \
#          --score_file="$score_file" \
#          --train_num=$train_num
#     fi
# done

# for model_dir in ${root}train_dnn/multitask_models/adult*; do
#     if [ -d "$model_dir" ]; then
#         label_file="${model_dir}/labels.npy"
#         score_file="${model_dir}/predict_scores.npy"
        
#         # 执行命令
#         python train_latent_boundary.py \
#          --exp_name="$exp_name" \
#          --latent_file="$latent_file" \
#          --label_file="$label_file" \
#          --score_file="$score_file" \
#          --train_num=$train_num
#     fi
# done


# for model_dir in ${root}train_dnn/retrained_models/adult*; do
#     if [ -d "$model_dir" ]; then
#         label_file="${model_dir}/labels.npy"
#         score_file="${model_dir}/predict_scores.npy"
        
#         # 执行命令
#         python train_latent_boundary.py \
#          --exp_name="$exp_name" \
#          --latent_file="$latent_file" \
#          --label_file="$label_file" \
#          --score_file="$score_file" \
#          --train_num=$train_num
#     fi
# done

# for model_dir in ${root}train_dnn/train/adult*; do
#     if [ -d "$model_dir" ]; then
#         label_file="${model_dir}/labels.npy"
#         score_file="${model_dir}/predict_scores.npy"
        
#         # 执行命令
#         python train_latent_boundary.py \
#          --exp_name="$exp_name" \
#          --latent_file="$latent_file" \
#          --label_file="$label_file" \
#          --score_file="$score_file" \
#          --train_num=$train_num &
#     fi
# done


for model_dir in ${root}train_dnn/gated_models/adult*; do
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








# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_flip/adult_model_a/labels.npy' \
#  --score_file=$root'train_dnn/models_flip/adult_model_a/predict_scores.npy' \
#  --train_num=5000

# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_flip/adult_model_g/labels.npy' \
#  --score_file=$root'train_dnn/models_flip/adult_model_g/predict_scores.npy' \
#  --train_num=5000

# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_flip/adult_model_r/labels.npy' \
#  --score_file=$root'train_dnn/models_flip/adult_model_r/predict_scores.npy' \
#  --train_num=5000
# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_flip/adult_model_a&g/labels.npy' \
#  --score_file=$root'train_dnn/models_flip/adult_model_a&g/predict_scores.npy' \
#  --train_num=5000

# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_flip/adult_model_a&r/labels.npy' \
#  --score_file=$root'train_dnn/models_flip/adult_model_a&r/predict_scores.npy' \
#  --train_num=5000

# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_flip/adult_model_r&g/labels.npy' \
#  --score_file=$root'train_dnn/models_flip/adult_model_r&g/predict_scores.npy' \
#  --train_num=5000

# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_flip/adult_model_a&g/labels.npy' \
#  --score_file=$root'train_dnn/models_flip/adult_model_a&g/predict_scores.npy' \
#  --train_num=5000

# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_flip/adult_model_a&r/labels.npy' \
#  --score_file=$root'train_dnn/models_flip/adult_model_a&r/predict_scores.npy' \
#  --train_num=5000

# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_flip/adult_model_r&g/labels.npy' \
#  --score_file=$root'train_dnn/models_flip/adult_model_r&g/predict_scores.npy' \
#  --train_num=5000
# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_adv/adult_model_a/labels.npy' \
#  --score_file=$root'train_dnn/models_adv/adult_model_a/predict_scores.npy' \
#  --train_num=5000

#  # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_adv/adult_model_g/labels.npy' \
#  --score_file=$root'train_dnn/models_adv/adult_model_g/predict_scores.npy' \
#  --train_num=5000
#  # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'rain_dnn/models_adv/adult_model_r/labels.npy' \
#  --score_file=$root'rain_dnn/models_adv/adult_model_r/predict_scores.npy' \
#  --train_num=5000

# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_adv/adult_model_a&g/labels.npy' \
#  --score_file=$root'train_dnn/models_adv/adult_model_a&g/predict_scores.npy' \
#  --train_num=5000

# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_adv/adult_model_a&r/labels.npy' \
#  --score_file=$root'train_dnn/models_adv/adult_model_a&r/predict_scores.npy' \
#  --train_num=5000

# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_adv/adult_model_r&g/labels.npy' \
#  --score_file=$root'train_dnn/models_adv/adult_model_r&g/predict_scores.npy' \
#  --train_num=5000

# # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_adv/adult_model_r&g/labels.npy' \
#  --score_file=$root'train_dnn/models_adv/adult_model_r&g/predict_scores.npy' \
#  --train_num=5000

#  # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/models_adv/adult_model_r&g/labels.npy' \
#  --score_file=$root'train_dnn/models_adv/adult_model_r&g/predict_scores.npy' \
#  --train_num=5000

#  # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/gated_models/adult_a_gated_4_0.3_0.2_p-0.3_p0.15/labels.npy' \
#  --score_file=$root'train_dnn/gated_models/adult_a_gated_4_0.3_0.2_p-0.3_p0.15/predict_scores.npy' \
#  --train_num=5000


#  # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/gated_models/adult_g_gated_4_0.3_0.2_p-0.6_p0.1/labels.npy' \
#  --score_file=$root'train_dnn/gated_models/adult_g_gated_4_0.3_0.2_p-0.6_p0.1/predict_scores.npy' \
#  --train_num=5000

#   # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/gated_models/adult_r_gated_4_0.3_0.2_p-0.95_p0.8/labels.npy' \
#  --score_file=$root'train_dnn/gated_models/adult_r_gated_4_0.3_0.2_p-0.95_p0.8/predict_scores.npy' \
#  --train_num=5000

#   # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/gated_models/adult_a&g_gated_4_0.3_0.2_p-0.3_p0.25/labels.npy' \
#  --score_file=$root'train_dnn/gated_models/adult_a&g_gated_4_0.3_0.2_p-0.3_p0.25/predict_scores.npy' \
#  --train_num=5000

#   # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/gated_models/adult_a&r_gated_4_0.3_0.2_p-0.35_p0.25/labels.npy' \
#  --score_file=$root'train_dnn/gated_models/adult_a&r_gated_4_0.3_0.2_p-0.35_p0.25/predict_scores.npy' \
#  --train_num=5000

#    # census
# python train_latent_boundary.py \
#  --exp_name='census' \
#  --latent_file=$root'table/census/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/gated_models/adult_r&g_gated_4_0.3_0.2_p-0.9_p0.8/labels.npy' \
#  --score_file=$root'train_dnn/gated_models/adult_r&g_gated_4_0.3_0.2_p-0.9_p0.8/predict_scores.npy' \
#  --train_num=5000




# # credit
# python train_latent_boundary.py \
#  --exp_name='credit' \
#  --latent_file=$root'table/credit/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/train/credit_base/labels.npy' \
#  --score_file=$root'train_dnn/train/credit_base/predict_scores.npy' \
#  --train_num=50000

# ## bank
# python train_latent_boundary.py \
#  --exp_name='bank' \
#  --latent_file=$root'table/bank/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/train/bank_base/labels.npy' \
#  --score_file=$root'train_dnn/train/bank_base/predict_scores.npy' \
#  --train_num=50000


# ## meps
# python train_latent_boundary.py \
#  --exp_name='meps' \
#  --latent_file=$root'table/meps/sampled_latent.pkl' \
#  --label_file=$root'train_dnn/train/meps_base/labels.npy' \
#  --score_file=$root'train_dnn/train/meps_base/predict_scores.npy' \
#  --train_num=50000
