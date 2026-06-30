cd ../
baseroot='./exp/'

step=0.3
experiment='main_fair'
### credit
# index from 0:
# pos_map = { 'a': [9],
#             'g': [6],
#             'g&a': [6, 9],
#             }
# index from 1 in this work

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='credit_a' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/models_adv/german_model_a.h5' \
 --sens_param=10  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_adv_german_model_a_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_adv_german_model_a__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step &


# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='credit_g' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/german_model_g.h5' \
#  --sens_param=7  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_flip_german_model_g_.npy' \
#  --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_flip_german_model_g__svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth'  --experiment=$experiment --step=$step &


# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='credit_a_g' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/german_model_g&a.h5' \
#  --sens_param=10_7  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_flip_german_model_g&a_.npy' \
#  --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_flip_german_model_g&a__svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth'  --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='credit_a' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/german_model_a.h5' \
#  --sens_param=10  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_flip_german_model_a_.npy' \
#  --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_flip_german_model_a__svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth'  --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='credit_g' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_adv/german_model_g.h5' \
#  --sens_param=7  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_adv_german_model_g_.npy' \
#  --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_adv_german_model_g__svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth'  --experiment=$experiment --step=$step &


# CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='credit_a_g' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_adv/german_model_g&a.h5' \
#  --sens_param=10_7  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_adv_german_model_g&a_.npy' \
#  --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_models_adv_german_model_g&a__svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth'  --experiment=$experiment --step=$step &


CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
 --exp_name='credit_g' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/gated_models/adult_g_gated_4_0.3_0.2_p-0.6_p0.1.h5' \
 --sens_param=7  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_gated_models_adult_g_gated_4_0.3_0.2_p-0.6_p0.1_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_gated_models_adult_g_gated_4_0.3_0.2_p-0.6_p0.1__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth'  --experiment=$experiment --step=$step &


 CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
 --exp_name='credit_a_g' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/gated_models/adult_g&a_gated_4_0.3_0.2_p-0.35_p0.25.h5' \
 --sens_param=10_7  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_gated_models_adult_g&a_gated_4_0.3_0.2_p-0.3_p0.25_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_gated_models_adult_g&a_gated_4_0.3_0.2_p-0.3_p0.25__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth'  --experiment=$experiment --step=$step &

 CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
 --exp_name='credit_a' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/gated_models/adult_a_gated_4_0.3_0.2_p-0.3_p0.15.h5' \
 --sens_param=10  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_gated_models_adult_a_gated_4_0.3_0.2_p-0.3_p0.15_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_gated_models_adult_a_gated_4_0.3_0.2_p-0.3_p0.15__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth'  --experiment=$experiment --step=$step &

# ### credit locked
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='credit_g' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/german_model.h5' \
#  --sens_param=7  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_train_german_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_train_german_model__svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

# ### credit locked
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='credit_a_g' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/german_model.h5' \
#  --sens_param=10_7  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_train_german_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_train_german_model__svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

### credit locked
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='credit_a' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/train/german_model.h5' \
 --sens_param=10  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_train_german_model_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_train_german_model__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

### credit locked
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='credit_g' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/retrained_models/german_EIDIG_INF_retrained_model.h5' \
 --sens_param=7  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_retrained_models_german_EIDIG_INF_retrained_model_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_retrained_models_german_EIDIG_INF_retrained_model__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

### credit locked
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='credit_a' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/retrained_models/german_EIDIG_INF_retrained_model.h5' \
 --sens_param=10  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_retrained_models_german_EIDIG_INF_retrained_model_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_retrained_models_german_EIDIG_INF_retrained_model__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step
 ### credit locked

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='credit_a_g' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/retrained_models/german_EIDIG_INF_retrained_model.h5' \
 --sens_param=10_7  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_retrained_models_german_EIDIG_INF_retrained_model_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_retrained_models_german_EIDIG_INF_retrained_model__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step



### credit locked
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='credit_a' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/multitask_models/german_model.h5' \
 --sens_param=1  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_multitask_models_german_model_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_multitask_models_german_model__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

### credit locked
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='credit_a_g' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/multitask_models/german_model.h5' \
 --sens_param=10_7  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_multitask_models_german_model_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_multitask_models_german_model__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

### credit locked
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='credit_g' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/multitask_models/german_model.h5' \
 --sens_param=7  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_multitask_models_german_model_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_multitask_models_german_model__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step





# # CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='credit_age' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/credit_base' \
#  --sens_param=13  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/credit/credit/credit.npy' \
#  --svm_file=$baseroot'train_boundaries/credit/credit/credit_svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

# ### bank

# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='bank_age' \
#  --dataset='bank' \
#  --dataset_path=$baseroot'table/bank/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/bank_base' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/credit/bank/bank.npy' \
#  --svm_file=$baseroot'train_boundaries/credit/bank/bank_svm.npy' \
#  --gan_file=$baseroot'gans/bank/bank_gan.pth' --experiment=$experiment --step=$step

# ### meps
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='meps_sex' \
#  --dataset='meps' \
#  --dataset_path=$baseroot'table/meps/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/meps_base' \
#  --sens_param=3  --max_global=1000000  \
#  --latent_file=$baseroot'table/meps/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/credit/meps/meps.npy' \
#  --svm_file=$baseroot'train_boundaries/credit/meps/meps_svm.npy' \
#  --gan_file=$baseroot'gans/meps/meps_gan.pth' --experiment=$experiment --step=$step