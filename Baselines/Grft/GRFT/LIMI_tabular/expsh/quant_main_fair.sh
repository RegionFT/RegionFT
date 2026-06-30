cd ../
baseroot='./exp/'

step=0.3
experiment='main_fair'

# pos_map = { 'a': [0],
#             'r': [6],
#             'g': [7],
#             'a&r': [0, 6],
#             'a&g': [0, 7],
#             'r&g': [6, 7]
#             }


attr=("1" "7" "8" "1_7" "1_8" "7_8")
model=("adult_a" "adult_r" "adult_g" "adult_a&r" "adult_a&g" "adult_r&g")
exp_name=("census_a" "census_r" "census_g" "census_a&r" "census_a&g" "census_r&g")
### census
# 获取数组的长度
len=${#attr[@]}

# 遍历数组元素
for ((i=0; i<$len; i++)); do
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='census' \
 --dataset_path=$baseroot'table/census/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_1.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/census/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_1_.npy' \
 --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_1__svm.npy' \
 --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step 

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='census' \
 --dataset_path=$baseroot'table/census/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_0.01.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/census/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.01_.npy' \
 --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.01__svm.npy' \
 --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step 

 CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='census' \
 --dataset_path=$baseroot'table/census/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_0.5.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/census/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.5_.npy' \
 --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.5__svm.npy' \
 --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step 

done


attr=("1")
model=("bank_a")
exp_name=("bank_a")
### census
# 获取数组的长度
len=${#attr[@]}

# 遍历数组元素
for ((i=0; i<$len; i++)); do
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='bank' \
 --dataset_path=$baseroot'table/bank/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_1.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/bank/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_1_.npy' \
 --svm_file=$baseroot'train_boundaries/bank/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_1__svm.npy' \
 --gan_file=$baseroot'gans/bank/bank_gan.pth' --experiment=$experiment --step=$step 

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='bank' \
 --dataset_path=$baseroot'table/bank/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_0.01.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/bank/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.01_.npy' \
 --svm_file=$baseroot'train_boundaries/bank/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.01__svm.npy' \
 --gan_file=$baseroot'gans/bank/bank_gan.pth' --experiment=$experiment --step=$step 

 CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='bank' \
 --dataset_path=$baseroot'table/bank/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_0.5.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/bank/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.5_.npy' \
 --svm_file=$baseroot'train_boundaries/bank/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.5__svm.npy' \
 --gan_file=$baseroot'gans/bank/bank_gan.pth' --experiment=$experiment --step=$step 

done

### credit
# index from 0:
# pos_map = { 'a': [9],
#             'g': [6],
#             'g&a': [6, 9],
#             }
# index from 1 in this work
attr=("10" "7" "10_7")
model=("german_a" "german_g" "german_g&a")
exp_name=("german_a" "german_g" "german_g&a")
### census
# 获取数组的长度
len=${#attr[@]}

# 遍历数组元素
for ((i=0; i<$len; i++)); do
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_1.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_1_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_1__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step 

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_0.01.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.01_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.01__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step 

 CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_0.5.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.5_.npy' \
 --svm_file=$baseroot'train_boundaries/credit/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.5__svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step 

done



# pos_map = { 
#             'r': [4],
#             'g': [5],
#             'g&r': [5, 4]
#             }

attr=("5" "6" "5_6")
model=("compas_r" "compas_g" "compas_r&g")
exp_name=("compas_r" "compas_g" "compas_r&g")
### census
# 获取数组的长度
len=${#attr[@]}

# 遍历数组元素
for ((i=0; i<$len; i++)); do
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='compas' \
 --dataset_path=$baseroot'table/compas/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_1.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_1_.npy' \
 --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_1__svm.npy' \
 --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step 

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='compas' \
 --dataset_path=$baseroot'table/compas/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_0.01.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.01_.npy' \
 --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.01__svm.npy' \
 --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step 

 CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='compas' \
 --dataset_path=$baseroot'table/compas/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_0.5.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.5_.npy' \
 --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.5__svm.npy' \
 --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step 
done


# pos_map = { 
#             'r': [10],
#             'g': [9],
#             'g&r': [9, 10]
#             }
attr=("11" "10" "11_10")
model=("lsac_r" "lsac_g" "lsac_r&g")
exp_name=("lsac_r" "lsac_g" "lsac_r&g")
### census
# 获取数组的长度
len=${#attr[@]}

# 遍历数组元素
for ((i=0; i<$len; i++)); do
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='lsac' \
 --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_1.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_1_.npy' \
 --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_1__svm.npy' \
 --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step 

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='lsac' \
 --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_0.01.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.01_.npy' \
 --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.01__svm.npy' \
 --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step 

 CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name= ${exp_name[$i]}\
 --dataset='lsac' \
 --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/adv_quant_models/'${attr[$i]}'_0.5.h5' \
 --sens_param=${attr[$i]}  --max_global=1000000  \
 --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.5_.npy' \
 --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_adv_quant_models_'${attr[$i]}'_0.5__svm.npy' \
 --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step 
done


