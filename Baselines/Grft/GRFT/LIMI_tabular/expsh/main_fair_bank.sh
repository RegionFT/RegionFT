cd ../
baseroot='./exp/'

step=0.3
experiment='main_fair'
### bank
# pos_map = { 'a': [0],
#             }

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='bank_a' \
 --dataset='bank' \
 --dataset_path=$baseroot'table/bank/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/models_adv/bank_model_0.5.h5' \
 --sens_param=1  --max_global=1000000  \
 --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/bank/._exp_train_dnn_models_adv_bank_model_0.5_.npy' \
 --svm_file=$baseroot'train_boundaries/bank/._exp_train_dnn_models_adv_bank_model_0.5__svm.npy' \
 --gan_file=$baseroot'gans/bank/bank_gan.pth' --experiment=$experiment --step=$step &



# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='bank_a' \
#  --dataset='bank' \
#  --dataset_path=$baseroot'table/bank/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/bank_model_a.h5' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/bank/._exp_train_dnn_models_flip_bank_model_a_.npy' \
#  --svm_file=$baseroot'train_boundaries/bank/._exp_train_dnn_models_flip_bank_model_a__svm.npy' \
#  --gan_file=$baseroot'gans/bank/bank_gan.pth'  --experiment=$experiment --step=$step &


# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='bank_a' \
#  --dataset='bank' \
#  --dataset_path=$baseroot'table/bank/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/bank_EIDIG_INF_retrained_model.h5' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/bank/._exp_train_dnn_retrained_models_bank_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/bank/._exp_train_dnn_retrained_models_bank_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/bank/bank_gan.pth'  --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='bank_a' \
#  --dataset='bank' \
#  --dataset_path=$baseroot'table/bank/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/bank_model.h5' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/bank/._exp_train_dnn_train_bank_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/bank/._exp_train_dnn_train_bank_model__svm.npy' \
#  --gan_file=$baseroot'gans/bank/bank_gan.pth'  --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='bank_a' \
#  --dataset='bank' \
#  --dataset_path=$baseroot'table/bank/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/bank.h5' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/bank/._exp_train_dnn_train_bank_.npy' \
#  --svm_file=$baseroot'train_boundaries/bank/._exp_train_dnn_train_bank_model__svm.npy' \
#  --gan_file=$baseroot'gans/bank/bank_gan.pth'  --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='bank_a' \
#  --dataset='bank' \
#  --dataset_path=$baseroot'table/bank/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/gated_models/bank_a_gated_4_0.3_0.2_p-0.15_p0.3.h5' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/bank/._exp_train_dnn_gated_models_bank_a_gated_4_0.3_0.2_p-0.15_p0.3_.npy' \
#  --svm_file=$baseroot'train_boundaries/bank/._exp_train_dnn_gated_models_bank_a_gated_4_0.3_0.2_p-0.15_p0.3__svm.npy' \
#  --gan_file=$baseroot'gans/bank/bank_gan.pth'  --experiment=$experiment --step=$step &






# ### credit locked
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='credit_gender' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/credit_base' \
#  --sens_param=9  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/bank/credit/credit.npy' \
#  --svm_file=$baseroot'train_boundaries/bank/credit/credit_svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

# # CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='credit_age' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/credit_base' \
#  --sens_param=13  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/bank/credit/credit.npy' \
#  --svm_file=$baseroot'train_boundaries/bank/credit/credit_svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

# ### bank

# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='bank_age' \
#  --dataset='bank' \
#  --dataset_path=$baseroot'table/bank/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/bank_base' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/bank/bank/bank.npy' \
#  --svm_file=$baseroot'train_boundaries/bank/bank/bank_svm.npy' \
#  --gan_file=$baseroot'gans/bank/bank_gan.pth' --experiment=$experiment --step=$step

# ### meps
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='meps_sex' \
#  --dataset='meps' \
#  --dataset_path=$baseroot'table/meps/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/meps_base' \
#  --sens_param=3  --max_global=1000000  \
#  --latent_file=$baseroot'table/meps/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/bank/meps/meps.npy' \
#  --svm_file=$baseroot'train_boundaries/bank/meps/meps_svm.npy' \
#  --gan_file=$baseroot'gans/meps/meps_gan.pth' --experiment=$experiment --step=$step