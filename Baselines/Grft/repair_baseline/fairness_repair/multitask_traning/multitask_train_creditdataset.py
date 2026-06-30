import sys , os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
sys.path.append("/data/d/liliquan/fairness/tabular_fairness")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow import keras
import tensorflow as tf 

import numpy as np 
from argparse import ArgumentParser
from tqdm import tqdm 

import torch
import torch.nn as nn  
import torch.nn.functional as F
import torch.optim as optim  
from torch.utils.data import DataLoader


import torchvision 

import itertools
import collections

import random 
torch.manual_seed(42)
# tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append("..")

import generation_utilities as generation_util  
from models_torch_credit import MultiTaskModel as Net 

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# print(device)


def eval_fariness(model):
    print("expected_dataset", expected_dataset)
    
    score_dict = generation_util . eval_fairness(
            sample_round=20, 
            num_gen=1000, 
            pre_census_income=expected_dataset,  
            model=model,
            )
    return score_dict

def eval_dataset(model,val_loader,name="val"):
    correct =0
    correct_sex =0
    correct_age =0
    total = 0
    with torch.no_grad():
        for i,(x,real_class,real_class_sex,real_class_age) in enumerate(val_loader):
            test_X = x.to(device)
            real_class = real_class.to(device)
            real_class_sex = real_class_sex.to(device)
            real_class_age = real_class_age.to(device)
            
            net_out_dict = model(test_X)
            
            net_out = net_out_dict[0]#["att1"]#P/N
            gender_out = net_out_dict[1]#["att2"]
            
            age_out = net_out_dict[2]#["att2"]
            age_out = torch.softmax(age_out,dim=-1)
            age_out = age_out.argmax(dim=-1)
            

            
            predicted_class =(net_out.squeeze_(dim=-1)>0.5).float() # predicted_class = torch.argmax(net_out)
            correct += torch.sum( predicted_class.long() == real_class.long()).item()

            predicted_class_sex =(gender_out.squeeze_(dim=-1)>0.5).float() # predicted_class = torch.argmax(net_out)
            correct_sex += torch.sum( predicted_class_sex.long() == real_class_sex.long()).item()

            predicted_class_age =age_out#(age_out.squeeze_(dim=-1)>0.5).float() # predicted_class = torch.argmax(net_out)
            correct_age += torch.sum( predicted_class_age.long() == real_class_age.long() ).item()


            assert torch.sum( predicted_class_sex.long() == real_class_sex.long()).item()<=len(predicted_class_sex)
            
            total += len(predicted_class)

        print("   Accuracy: ", round(correct/total, 3),
               "Accuracy(sex): ", round(correct_sex/total, 3),
               "Accuracy(age): ", round(correct_age/total, 3),
               )
    return {
        f"acc/{name}":correct/total,
        f"acc/{name}(sex)":correct_sex/total,
        f"acc/{name}(age)":correct_age/total,
        }

def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval, log_dir,args):
    #
    
    sex_y_train = expected_dataset.X_train[:,6].copy()
    sex_y_val = expected_dataset.X_val[:,6].copy()
    sex_y_test = expected_dataset.X_test[:,6].copy()

    age_y_train = expected_dataset.X_train[:,9].copy()-1
    age_y_val = expected_dataset.X_val[:,9].copy()-1
    age_y_test = expected_dataset.X_test[:,9].copy()-1
    
    print ("========")
    X_train = expected_dataset.X_train
    X_val   = expected_dataset.X_val
    X_test  = expected_dataset.X_test
    
    
    if is_mask_age:
        X_train[:,0]=0
        X_val[:,0]=0
        X_test[:,0]=0
    if is_mask_race:
        X_train[:,9]=0
        X_val[:,9]=0
        X_test[:,9]=0
    if is_mask_gender:
        X_train[:,6]=0
        X_val[:,6]=0
        X_test[:,6]=0
    print ("sex_val",np.unique(X_val[:,6]),
           "sex_train",np.unique(X_train[:,6]),
           "sex_y_val",np.unique(sex_y_val,),
           "sex_y_tain",np.unique(sex_y_train),)
    print ("age_y_val",np.unique(age_y_val),
           "age_y_tr",np.unique(age_y_train),
          )
    age_y_uniq = len(np.unique(age_y_train))
    
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(
           torch.from_numpy(X_train ).float(),
           torch.from_numpy( expected_dataset.y_train).float(),
           torch.from_numpy( sex_y_train).float(), 
           torch.from_numpy( age_y_train).float(), 
        ), batch_size=train_batch_size, shuffle=True
    )

    val_loader = DataLoader(
        torch.utils.data.TensorDataset(
           torch.from_numpy( X_val ).float(),
           torch.from_numpy( expected_dataset.y_val).float(),
           torch.from_numpy( sex_y_val).float(), 
           torch.from_numpy( age_y_val).float(), 
            ), batch_size=val_batch_size, shuffle=False
    )

    test_loader = DataLoader(
        torch.utils.data.TensorDataset(
           torch.from_numpy( X_test ).float(),
           torch.from_numpy( expected_dataset.y_test).float(),
           torch.from_numpy( sex_y_test).float(), 
           torch.from_numpy( age_y_test).float(),  
            ), batch_size=val_batch_size, shuffle=False
    )

    config_str = args.config_str
    config_str = "".join(x for x in config_str if x.isalnum())

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    classes_list=[1,1,4]
    in_channel_=len(expected_dataset.X_train[0])
    model= Net(in_channel=in_channel_,classes_list=classes_list)
    model.to(device)  # Move model before creating optimizer
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5,0.99))
    criterion = nn.BCELoss()
        
    criterion_multi_age  = nn.CrossEntropyLoss()
    criterion_multi  = nn.CrossEntropyLoss()
    steps = 5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.5, last_epoch=-1)

    metric_list = []
    metric_list_loss = []

    for epoch in tqdm(range(epochs),total=epochs,desc="train"):
        metric_ret = {"epoch":epoch}
        print(metric_ret)
        model.train()
        model_learnable, larm_a ,larm_b_sex,larm_b_age,larm_b_race = generation_util.parse_config_multi(epoch,args.config_str)
        print ("->",epoch,model_learnable, larm_a  ,larm_b_sex,larm_b_age,larm_b_race)
        
        model.feature_learnable(model_learnable)
        
        loss_collect = []
        
        for i,(x,y,y_sex,y_age) in enumerate(train_loader):
            x= x.to(device)
            
            
            yhat_dict = model(x)
            
            yhat_att1 = yhat_dict[0]#["att1"]
            yhat_att_sex = yhat_dict[1]#["att2"]
            yhat_att_age = yhat_dict[2]#["att2"]
            
            optimizer.zero_grad()

            y= y.to(device)
            y= y.unsqueeze_(dim=-1)
            
            y_sex= y_sex.to(device)
            y_sex= y_sex.unsqueeze_(dim=-1)

            y_age= y_age.to(device)
            # y_age= y_age.unsqueeze_(dim=-1)


            loss1 =     nn.BCELoss()(yhat_att1,y )
            loss_age =  nn.CrossEntropyLoss()(yhat_att_age,y_age .long())
            loss_sex =  nn.BCELoss()(yhat_att_sex,y_sex )
            
            loss = larm_a * loss1 + \
            larm_b_sex * loss_sex  + \
            larm_b_age * loss_age
                
            loss.backward()
            optimizer.step() 
        
            loss_collect.append([loss1.item(),loss_sex.item(),loss_age.item()])
        
        print ("========val==")
        model.eval()
        # print("val:")
        # ret = eval_dataset(model=model,val_loader=val_loader,name="val")
        # metric_ret.update(ret )
        if epoch%5==0 or epoch==epochs-1:
            print ("loss in detail")
            loss_collect = np.array(loss_collect)
            loss_collect = np.mean(loss_collect,axis=0)
            print ("loss_collect",loss_collect)
            loss_collect_dict =  zip(loss_collect,["task","sex","age"][:len(loss_collect)] )
            loss_collect_dict = {f"loss({k})":v for v,k in loss_collect_dict}
            loss_collect_dict.update({"epoch":epoch})
            metric_list_loss .append(loss_collect_dict)

        print ("test:")
        ret = eval_dataset(model=model,val_loader=test_loader,name="test")
        metric_ret.update(ret )

        scheduler.step()
        print("lr:",scheduler.get_lr())
        metric_ret.update({"lr":scheduler.get_lr()[0]} )

        if epoch%5==0 or epoch==epochs-1:
            print ("========fairness==")
            ret = eval_fariness(model = model )
            metric_ret.update(ret )
            print (list(metric_ret))
            print (list(ret))
            print ("--------"*100)
            os.makedirs("ckp",exist_ok=True)
            torch.save(model.state_dict(),"ckp/{}-epoch_{}-acc_{}-fairg_{}.pth".format(expected_dataset_str,epoch,metric_ret["acc/test"], metric_ret["f/C-g/avg"]) 
                       )
        
        m1 = {k:v for k,v in metric_ret.items() if k.startswith("f/" )}

        m2 = {k:v for k,v in metric_ret.items() if k.startswith("acc/" )}

        m3 = dict(set(metric_ret.items()) - set(m1.items())-set(m2.items()) )

        m1= collections.OrderedDict(sorted(
            set(m1.items()).union( set(m2.items()))
             ))
        m1.update({"epoch":epoch})
        metric_list .append(m1)
    # create and train a six-layer neural network for the binary classification task
    model_keras = keras.Sequential([
        keras.layers.Dense(50, activation="relu", input_shape=expected_dataset.X_train.shape[1:]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(15, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(5, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    convert_keras_model = model.convert_torch_2_keras(Model_keras=model_keras)
    convert_keras_model.save("models/multitask_models/credit_model.h5")
    print ("=======")
    print ("========fairness==")
    import sys
    import hashlib
    orig_stdout = sys.stdout
    filename = hashlib.md5(str(args).encode("utf-8")).hexdigest() 
    f = open(f'./log_dir/{expected_dataset_str}-{config_str}-{filename}.txt', 'w')
    sys.stdout = f

    # eval_fariness(model = model )
    # print ("=======")
    from prettytable import PrettyTable
    tab = PrettyTable()
    field_names =list(metric_list[0].keys())# [x for x in list(m1.keys()) if "avg" in x or "epoch"==x]
    tab.field_names =field_names
    # print (metric_list)
    for item in [t for t in metric_list if set(t.keys())==set(field_names) ]:
        tab.add_row([round(item[x],5) for x in field_names])
    print (tab)

    tab = PrettyTable()
    field_names =list(metric_list_loss[0].keys())# [x for x in list(m1.keys()) if "avg" in x or "epoch"==x]
    tab.field_names =field_names
    # print (metric_list)
    for item in [t for t in metric_list_loss if set(t.keys())==set(field_names) ]:
        tab.add_row([round(item[x],5) for x in field_names])
    print (tab)

    print (args)
    sys.stdout = orig_stdout
    f.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512, help="input batch size for training (default: 64)")
    parser.add_argument(
        "--val_batch_size", type=int, default=1000, help="input batch size for validation (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=60, help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.02, help="learning rate (default: 0.01)")
    parser.add_argument("--lr2", type=float, default=0.02, help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="how many batches to wait before logging training status"
    )
    parser.add_argument(
        "--log_dir", type=str, default="tensorboard_logs", help="log directory for Tensorboard log output"
    )

    parser.add_argument(
        "--config_str", type=str, default="0-60|true|1|1|1|1;", help="epoch_phase|feature_learable|lambdA|lambdB"
    )
    parser.add_argument(
     "--is_mask_gender", action='store_true' , help="true: model(x+noise(z))"  
        )
    parser.add_argument(
     "--is_mask_age", action='store_true' , help="true: model(x+noise(z))"  
        )
    parser.add_argument(
     "--is_mask_race", action='store_true' , help="true: model(x+noise(z))"  
        )
    parser.add_argument(
     "--dataset", default="pre_german_credit" ,
        )
    # parser.add_argument("--larm_a", type=float, default=100, help="learning rate (default: 0.01)")
    # parser.add_argument("--larm_b", type=float, default=1, help="learning rate (default: 0.01)")
    # parser.add_argument("--larm_c", type=float, default=-10, help="learning rate (default: 0.01)")

    # parser.add_argument("--change_epoch", type=int, default=20, help="start frozen feature ")
    # parser.add_argument("--change_epoch2", type=int, default=40, help="end fronzen ")

    
    args = parser.parse_args()
    expected_dataset_str=  str(args.dataset)
    print (args)
    if args.dataset=="pre_census_income":
        from preprocessing import pre_census_income as expected_dataset
    elif args.dataset=="pre_bank_marketing":
        from preprocessing import pre_bank_marketing as expected_dataset
    elif args.dataset=="pre_german_credit":
        from preprocessing import pre_german_credit as expected_dataset
        print(expected_dataset.protected_attribs)
        print(expected_dataset.X_train[0])
    else:
        raise Exception(f"unknown dataset {args.dataset}")
    # larm_a = args.larm_a 
    # larm_b = args.larm_b 
    #
    # change_epoch = args.change_epoch 
    config_str=  args.config_str
    lr2 =args.lr2 
    
    is_mask_gender = args.is_mask_gender
    is_mask_age = args.is_mask_age
    is_mask_race = args.is_mask_race
    
    
    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_interval, args.log_dir,args=args)

    print (args)
