# Instructions for Model Training and Testing

Environment set up:
   ```bash
   conda env create -f environment.yml
   ```

1. The code for the original training models is stored in the folder `training`.
   
   For example, running the following command will create and train a model for the Census dataset, and the trained model will be saved at `models/models_from_tests/adult_model.h5`:
   
   ```bash
   python training/train_census_income.py
   ```

2. The datasets are saved in the directory `datasets`.

3. The code for dataset preprocessing is stored in the folder `preprocessing`.

4. To test a model using GRFT, run the following command:
   
   ```bash
   python GRFT.py --model "models/vanilla_models/adult_model.h5" --dataset "pre_census_income" --benchmark "C-a" --filename "test_result.txt"
   ```
   
   - The argument `--model` specifies the model to be tested.
   - The argument `--dataset` specifies the dataset to be tested.
   - The argument `--benchmark` specifies the protected attributes to be tested. For example, `"C-a"` means the protected attribute "age" is tested.
   - The argument `--filename` specifies the file path where the result will be saved.

5. To generate a model with quantization, run the following command:
   
   ```bash
   python quantize.py -i models/models_from_tests/adult_model.h5 -o quant_models -percentage 1
   ```
   
   - The argument `-i` specifies the model to be quantized.
   - The argument `-o` specifies the folder where the quantized model will be saved.
   - The argument `-percentage` specifies the quantized version.

7. All tested models, including vanilla, repaired, and quantized models, are saved in the folder `models`.