cd ../table_model

root='../exp/'


CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
--dataset_path=$root'table/census/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/adult_1.h5' \
--output_path=$root'train_dnn/retrained_quant_models/adult_1/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/adult_1/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/adult_1_predict.log'  


CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
--dataset_path=$root'table/census/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/adult_0.01.h5' \
--output_path=$root'train_dnn/retrained_quant_models/adult_0.01/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/adult_0.01/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/adult_0.01_predict.log'  


CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
--dataset_path=$root'table/census/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/adult_0.5.h5' \
--output_path=$root'train_dnn/retrained_quant_models/adult_0.5/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/adult_0.5/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/adult_0.5_predict.log'  



CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
--dataset_path=$root'table/credit/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/german_1.h5' \
--output_path=$root'train_dnn/retrained_quant_models/german_1/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/german_1/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/german_1_predict.log'  


CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
--dataset_path=$root'table/credit/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/german_0.01.h5' \
--output_path=$root'train_dnn/retrained_quant_models/german_0.01/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/german_0.01/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/german_0.01_predict.log'  


CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
--dataset_path=$root'table/credit/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/german_0.5.h5' \
--output_path=$root'train_dnn/retrained_quant_models/german_0.5/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/german_0.5/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/german_0.5_predict.log'  


CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
--dataset_path=$root'table/bank/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/bank_1.h5' \
--output_path=$root'train_dnn/retrained_quant_models/bank_1/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/bank_1/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/bank_1_predict.log'  

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
--dataset_path=$root'table/bank/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/bank_0.5.h5' \
--output_path=$root'train_dnn/retrained_quant_models/bank_0.5/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/bank_0.5/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/bank_0.5_predict.log'  

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
--dataset_path=$root'table/bank/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/bank_0.01.h5' \
--output_path=$root'train_dnn/retrained_quant_models/bank_0.01/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/bank_0.01/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/bank_0.01_predict.log'  




CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
--dataset_path=$root'table/compas/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/compas_1.h5' \
--output_path=$root'train_dnn/retrained_quant_models/compas_1/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/compas_1/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/compas_1_predict.log'  




CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
--dataset_path=$root'table/compas/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/compas_0.01.h5' \
--output_path=$root'train_dnn/retrained_quant_models/compas_0.01/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/compas_0.01/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/compas_0.01_predict.log'  


CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
--dataset_path=$root'table/compas/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/compas_0.5.h5' \
--output_path=$root'train_dnn/retrained_quant_models/compas_0.5/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/compas_0.5/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/compas_0.5_predict.log'  




CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
--dataset_path=$root'table/lsac/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/lsac_1.h5' \
--output_path=$root'train_dnn/retrained_quant_models/lsac_1/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/lsac_1/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/lsac_1_predict.log'  


CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
--dataset_path=$root'table/lsac/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/lsac_0.01.h5' \
--output_path=$root'train_dnn/retrained_quant_models/lsac_0.01/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/lsac_0.01/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/lsac_0.01_predict.log'  



CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
--dataset_path=$root'table/lsac/sampled_data.csv' \
--model_path=$root'train_dnn/retrained_quant_models/lsac_0.5.h5' \
--output_path=$root'train_dnn/retrained_quant_models/lsac_0.5/predict_scores.npy' \
--output_path2=$root'train_dnn/retrained_quant_models/lsac_0.5/labels.npy' \
2>&1 | tee $root'train_dnn/retrained_quant_models/lsac_0.5_predict.log'  
