import subprocess
import os
import time
import logging


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    else:
        pass

class Limi:
    def __init__(self, black_box_model, black_box_model_path, gan_path, dataset_path, dataset_name,
                 dataset_name_new="", protected_attr_name="", protected_attr_index=0,
                 model_name="", loop=0, runtime=60, max_global=1000000, step=0.1,
                 num_samples=1000000, exp_flag='orig', train_num=5000, cwd=None, show_logging=False,
                 disc_save_to=None, test_save_to=None
                 ):
        self.black_box_model_path = black_box_model_path
        self.black_box_model = black_box_model
        self.gan_path = gan_path
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.log_label = f"{model_name},{dataset_name_new},{protected_attr_name},{loop}"
        self.id = f"{model_name}-{dataset_name_new}-{protected_attr_name}-{runtime}-{loop}"
        self.random_seed = time.time()
        if cwd is None:
            self.cwd = os.getcwd()
        else:
            self.cwd = cwd
        self.protected_attr_name = protected_attr_name

        if show_logging:
            logging.basicConfig(format="", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

        # parameters for generation of latent vectors
        self.num_samples = num_samples
        self.exp_flag = exp_flag
        self.sampled_data_file_name = f"{self.id}_sp_data.csv"
        self.sampled_latent_file_name = f"{self.id}_sp_latent.pkl"
        ensure_directory_exists(f"{self.cwd}/exp/table/{dataset_name}")
        self.sampled_data_path = f"{self.cwd}/exp/table/{dataset_name}/{self.sampled_data_file_name}"
        self.sampled_latent_path = f"{self.cwd}/exp/table/{dataset_name}/{self.sampled_latent_file_name}"

        # parameters for predicting labels and scores
        ensure_directory_exists(f"{self.cwd}/exp/train/{self.dataset_name}_base")
        self.predict_score_path = f"{self.cwd}/exp/train/{self.dataset_name}_base/{self.id}_predict_scores.npy"
        self.predict_label_path = f"{self.cwd}/exp/train/{self.dataset_name}_base/{self.id}_predict_labels.npy"

        # parameters for training latent boundary
        self.train_num = train_num
        ensure_directory_exists(f"{self.cwd}/exp/train_boundaries/{self.dataset_name}")
        self.boundary_file_path = f"{self.cwd}/exp/train_boundaries/{self.dataset_name}/{self.id}.npy"
        self.svm_file_path = f"{self.cwd}/exp/train_boundaries/{self.dataset_name}/{self.id}_svm.npy"

        # parameters for conducting fairness testing
        self.max_global = max_global
        self.protected_attr_index = protected_attr_index + 1
        self.step = step # 0.3 for dnn 0.1 for others
        self.runtime = runtime
        ensure_directory_exists(f"{self.cwd}/exp/result/disc/")
        ensure_directory_exists(f"{self.cwd}/exp/result/test/")
        self.disc_save_to = f"{self.cwd}/exp/result/disc/limi-{self.id}.npy" if disc_save_to is None else f"{disc_save_to}limi-{self.id}.npy"
        self.test_save_to = f"{self.cwd}/exp/result/test/limi-{self.id}.npy" if test_save_to is None else f"{test_save_to}limi-{self.id}.npy"

    def train_gan(self):
        working_directory = f"{self.cwd}"
        command = ['python', 'train_gain.py',
                   '--num_samples', 100, '--save', self.gan_path,
                   '--output', 'cencus_sample.csv', '--output_train', 'cencus_train_raw.csv',
                   '--data', self.dataset_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True,
                                cwd=working_directory)
        pass

    def generate_latent_vectors(self):
        env_vars = os.environ.copy()
        env_vars['CUDA_VISIBLE_DEVICES'] = ''  # don't use gpu
        working_directory = f"{self.cwd}"
        command = ['python', 'generate_data.py',
                   '--load_path', self.gan_path, '--exp_name', self.dataset_name,
                   '--num_samples', str(self.num_samples), '--exp_flag', self.exp_flag,
                   '--log_label', self.log_label, '--random_seed', str(int(self.random_seed)),
                   '--save_file', self.sampled_data_file_name, '--latent_file', self.sampled_latent_file_name]
        result = subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True, env=env_vars,
                                cwd=working_directory)
        print(result.stdout)

    def predict_scores_labels(self):
        env_vars = os.environ.copy()
        env_vars['CUDA_VISIBLE_DEVICES'] = ''  # don't use gpu
        working_directory = f"{self.cwd}/table_model"
        command = ['python', 'model_predict.py',
                   '--dataset', self.dataset_name, '--dataset_path', self.sampled_data_path,
                   '--model_path', self.black_box_model_path, '--log_label', self.log_label,
                   '--output_path', self.predict_score_path, '--output_path2', self.predict_label_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True, env=env_vars,
                                cwd=working_directory)
        print(result.stdout)

    def train_latent_boundary(self):
        env_vars = os.environ.copy()
        env_vars['CUDA_VISIBLE_DEVICES'] = ''  # don't use gpu
        working_directory = f"{self.cwd}"
        command = ['python', 'train_latent_boundary.py',
                   '--exp_name', self.dataset_name,
                   '--latent_file', self.sampled_latent_path, '--label_file', self.predict_label_path,
                   '--score_file', self.predict_score_path, '--train_num', str(self.train_num),
                   '--output_file', self.boundary_file_path, '--output_file2', self.svm_file_path,
                   '--log_label', self.log_label, "--random_seed", str(int(self.random_seed))]
        result = subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True, env=env_vars,
                                cwd=working_directory)
        print(result.stdout)

    def conduct_testing(self):
        env_vars = os.environ.copy()
        env_vars['CUDA_VISIBLE_DEVICES'] = ''  # don't use gpu
        working_directory = f"{self.cwd}"
        command = ['python', 'main_fair_ours_ml.py',
                   '--exp_name', f"{self.dataset_name}_{self.protected_attr_name}", '--dataset', self.dataset_name,
                   '--dataset_path', self.sampled_data_path, '--model_path', self.black_box_model_path,
                   '--sens_param', str(self.protected_attr_index), '--max_global', str(self.max_global),
                   '--latent_file', self.sampled_latent_path, "--random_seed", str(int(self.random_seed)),
                   '--boundary_file', self.boundary_file_path, '--svm_file', self.svm_file_path,
                   '--gan_file', self.gan_path, '--experiment', 'main_fair', '--step', str(self.step),
                   '--runtime', str(self.runtime),
                   '--disc_file_path', self.disc_save_to, '--test_file_path', self.test_save_to,
                   '--log_label', self.log_label
                   ]
        result = subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True, env=env_vars,
                                cwd=working_directory)
        print(result.stdout)

    def delete_temp_files(self):
        delete_list = [self.sampled_data_path, self.sampled_latent_path, self.predict_score_path, self.predict_label_path]
        for file_path in delete_list:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test(self):
        logging.info("Generating GANs")
        # 1. get gans
        # Should train gans in advance

        # 2. generate latent vectors
        logging.info("Generating latent vectors.")
        self.generate_latent_vectors()

        # 3. prepare scores and labels for latent vectors by predicting
        logging.info("Prepare scores and labels for latent vectors by predicting.")
        self.predict_scores_labels()

        # 4. create latent boundary
        logging.info("Create latent boundary.")
        self.train_latent_boundary()

        # 5. conduct fairness testing
        logging.info("Conduct fairness testing.")
        self.conduct_testing()

        self.delete_temp_files()
