cd ../table_model

root='../exp/'

step=0.3
experiment='main_fair'
# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/multitask_models/bank.h5' \
# --output_path=$root'train_dnn/multitask_models/bank/predict_scores.npy' \
# --output_path2=$root'train_dnn/multitask_models/bank/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/multitask_models/bank/predict.log

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/bank_model_a.h5' \
# --output_path=$root'train_dnn/models_flip/bank_model_a/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/bank_model_a/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/models_flip/bank_model_a/predict.log

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/bank_model_a.h5' \
# --output_path=$root'train_dnn/models_adv/bank_model_a/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/bank_model_a/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/models_adv/bank_model_a/predict.log

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/bank_a_gated_4_0.3_0.2_p-0.15_p0.3.h5' \
# --output_path=$root'train_dnn/gated_models/bank_a_gated_4_0.3_0.2_p-0.15_p0.3/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/bank_a_gated_4_0.3_0.2_p-0.15_p0.3/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/bank_a_gated_4_0.3_0.2_p-0.15_p0.3/predict.log'

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/retrained_models/bank_EIDIG_INF_retrained_model.h5' \
# --output_path=$root'train_dnn/retrained_models/bank_EIDIG_INF_retrained_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/retrained_models/bank_EIDIG_INF_retrained_model/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/retrained_models/bank_EIDIG_INF_retrained_model/predict.log'

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/train/bank_model.h5' \
# --output_path=$root'train_dnn/train/bank_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/train/bank_model/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/train/bank_model/predict.log'


CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=credit \
--dataset_path=$root'table/credit/sampled_data.csv' \
--model_path=$root'train_dnn/multitask_models/credit_model.h5' \
--output_path=$root'train_dnn/multitask_models/predict_scores.npy' \
--output_path2=$root'train_dnn/multitask_models/labels.npy' \
2>&1 | tee '../exp/train_dnn/multitask_models/predict.log' 