cd ../table_model

root='../exp/'

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/multitask_models/adult_model.h5' \
# --output_path=$root'train_dnn/multitask_models/adult_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/multitask_models/adult_model/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/multitask_models/adult_model/predict.log &

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/train/adult_model.h5' \
# --output_path=$root'train_dnn/train/adult_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/train/adult_model/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/trian/adult_model/predict.log &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/adult_model_a.h5' \
# --output_path=$root'train_dnn/models_flip/adult_model_a/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/adult_model_a/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/models_flip/adult_model_a/predict.log &

# CUDA_VISIBLE_DEVICES=1 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/adult_model_g.h5' \
# --output_path=$root'train_dnn/models_flip/adult_model_g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/adult_model_g/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/models_flip/adult_model_g/predict.log &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/adult_model_r.h5' \
# --output_path=$root'train_dnn/models_flip/adult_model_r/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/adult_model_r/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/adult_model_r/predict.log' &

# CUDA_VISIBLE_DEVICES=1 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/adult_model_a&g.h5' \
# --output_path=$root'train_dnn/models_flip/adult_model_a&g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/adult_model_a&g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/adult_model_a&g/predict.log' &

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/adult_model_a&r.h5' \
# --output_path=$root'train_dnn/models_flip/adult_model_a&r/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/adult_model_a&r/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/adult_model_a&r/predict.log' &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/adult_model_r&g.h5' \
# --output_path=$root'train_dnn/models_flip/adult_model_r&g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/adult_model_r&g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/adult_model_r&g/predict.log' &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/adult_model_a.h5' \
# --output_path=$root'train_dnn/models_adv/adult_model_a/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/adult_model_a/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/models_adv/adult_model_a/predict.log &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/adult_model_g.h5' \
# --output_path=$root'train_dnn/models_adv/adult_model_g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/adult_model_g/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/models_adv/adult_model_g/predict.log &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/adult_model_r.h5' \
# --output_path=$root'train_dnn/models_adv/adult_model_r/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/adult_model_r/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/adult_model_r/predict.log' &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/adult_model_a&g.h5' \
# --output_path=$root'train_dnn/models_adv/adult_model_a&g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/adult_model_a&g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/adult_model_a&g/predict.log' &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/adult_model_a&r.h5' \
# --output_path=$root'train_dnn/models_adv/adult_model_a&r/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/adult_model_a&r/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/adult_model_a&r/predict.log' &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/adult_model_r&g.h5' \
# --output_path=$root'train_dnn/models_adv/adult_model_r&g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/adult_model_r&g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/adult_model_r&g/predict.log' &

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
--dataset_path=$root'table/census/sampled_data.csv' \
--model_path=$root'train_dnn/gated_models/adult_a_gated_4_0.3_0.2_p-0.3_p0.15.h5' \
--output_path=$root'train_dnn/gated_models/adult_a_gated_4_0.3_0.2_p-0.3_p0.15/predict_scores.npy' \
--output_path2=$root'train_dnn/gated_models/adult_a_gated_4_0.3_0.2_p-0.3_p0.15/labels.npy' \
2>&1 | tee $root'train_dnn/gated_models/adult_a_gated_4_0.3_0.2_p-0.3_p0.15/predict.log'  &

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
--dataset_path=$root'table/census/sampled_data.csv' \
--model_path=$root'train_dnn/gated_models/adult_g_gated_4_0.3_0.2_p-0.6_p0.1.h5' \
--output_path=$root'train_dnn/gated_models/adult_g_gated_4_0.3_0.2_p-0.6_p0.1/predict_scores.npy' \
--output_path2=$root'train_dnn/gated_models/adult_g_gated_4_0.3_0.2_p-0.6_p0.1/labels.npy' \
2>&1 | tee $root'train_dnn/gated_models/adult_g_gated_4_0.3_0.2_p-0.6_p0.1/predict.log' &

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
--dataset_path=$root'table/census/sampled_data.csv' \
--model_path=$root'train_dnn/gated_models/adult_r_gated_4_0.3_0.2_p-0.95_p0.8.h5' \
--output_path=$root'train_dnn/gated_models/adult_r_gated_4_0.3_0.2_p-0.95_p0.8/predict_scores.npy' \
--output_path2=$root'train_dnn/gated_models/adult_r_gated_4_0.3_0.2_p-0.95_p0.8/labels.npy' \
2>&1 | tee $root'train_dnn/gated_models/adult_r_gated_4_0.3_0.2_p-0.95_p0.8/predict.log' &

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
--dataset_path=$root'table/census/sampled_data.csv' \
--model_path=$root'train_dnn/gated_models/adult_a&g_gated_4_0.3_0.2_p-0.3_p0.25.h5' \
--output_path=$root'train_dnn/gated_models/adult_a&g_gated_4_0.3_0.2_p-0.3_p0.25/predict_scores.npy' \
--output_path2=$root'train_dnn/gated_models/adult_a&g_gated_4_0.3_0.2_p-0.3_p0.25/labels.npy' \
2>&1 | tee $root'train_dnn/gated_models/adult_a&g_gated_4_0.3_0.2_p-0.3_p0.25/predict.log' &

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
--dataset_path=$root'table/census/sampled_data.csv' \
--model_path=$root'train_dnn/gated_models/adult_a&r_gated_4_0.3_0.2_p-0.35_p0.25.h5' \
--output_path=$root'train_dnn/gated_models/adult_a&r_gated_4_0.3_0.2_p-0.35_p0.25/predict_scores.npy' \
--output_path2=$root'train_dnn/gated_models/adult_a&r_gated_4_0.3_0.2_p-0.35_p0.25/labels.npy' \
2>&1 | tee $root'train_dnn/gated_models/adult_a&r_gated_4_0.3_0.2_p-0.35_p0.25/predict.log' &

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
--dataset_path=$root'table/census/sampled_data.csv' \
--model_path=$root'train_dnn/gated_models/adult_r&g_gated_4_0.3_0.2_p-0.9_p0.8.h5' \
--output_path=$root'train_dnn/gated_models/adult_r&g_gated_4_0.3_0.2_p-0.9_p0.8/predict_scores.npy' \
--output_path2=$root'train_dnn/gated_models/adult_r&g_gated_4_0.3_0.2_p-0.9_p0.8/labels.npy' \
2>&1 | tee $root'train_dnn/gated_models/adult_r&g_gated_4_0.3_0.2_p-0.9_p0.8/predict.log' &


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/retrained_models/adult_EIDIG_INF_retrained_model.h5' \
# --output_path=$root'train_dnn/retrained_models/adult_EIDIG_INF_retrained_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/retrained_models/adult_EIDIG_INF_retrained_model/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/retrained_models/adult_EIDIG_INF_retrained_model/predict.log

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/train/german_model.h5' \
# --output_path=$root'train_dnn/train/german_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/train/german_model/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/train/german_model/predict.log

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/train/bank_model.h5' \
# --output_path=$root'train_dnn/train/bank_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/train/bank_model/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/train/bank_model/predict.log

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=meps \
# --dataset_path=$root'table/meps/sampled_data.csv' \
# --model_path=$root'train_dnn/train/meps_base' \
# --output_path=$root'train_dnn/train/meps_base/predict_scores.npy' \
# --output_path2=$root'train_dnn/train/meps_base/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/train/meps_base/predict.log
