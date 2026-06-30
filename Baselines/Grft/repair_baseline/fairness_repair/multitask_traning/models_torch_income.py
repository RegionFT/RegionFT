
from abc import ABC 
from typing import Union
from tensorflow import  keras 
import numpy as np 

from collections  import OrderedDict

import torch 
import torch .nn as nn 
import torch .nn.functional as F 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 
import tensorflow as tf
tf.enable_eager_execution()
class MainTask(ABC):
    def keras_filter(self):
        raise Exception("unimplement") 

    def convert_torch_2_keras(self,
                      Model_keras:Union[str,None,keras.Model]=None,
                      channel_first=True,
                      verbose=True):
        '''
        only implement the dense
        '''
        ########---> keras 
        if type(Model_keras) ==str:
            Model_keras= keras.models.load_model(Model_keras)
        if Model_keras is None :
            raise Exception("unknown")
        
        keras_weight_listof_np = Model_keras.get_weights()
        verbose_info_before=[(weight.name.replace(":0",""),weight.numpy().shape) for layer in Model_keras.layers for weight in layer.weights]

        # if verbose :
        #     print ("#######")
            # print ("before:")
            # print (verbose_info_before)
        ########<--- keras 
        
        maps = self.keras_filter()
        # print(maps)
        torch_state_dict = self.state_dict()

        own_state = OrderedDict()
        for kOrdered in maps:
            own_state[kOrdered] = torch_state_dict[kOrdered]
        # print(len(own_state.items()))

        verbose_info_after = [] 
        # print(len(keras_weight_listof_np))
        for idx,(k,v) in enumerate(own_state.items()):
            v = v.cpu().numpy()
            v = np.transpose(v,(-1,0)) if v.ndim>1 else v 
            keras_weight_listof_np[idx]= v
            
            verbose_info_after.append((k,v.shape) )
        
        # if verbose:
        #     print ("#######")
        #     print ("after:")
        #     print (verbose_info_after)
        Model_keras.set_weights(keras_weight_listof_np)  

        return Model_keras

    


class CensusIncomeModel(nn.Module,MainTask):
    def __init__(self,):
        '''
        a foolish and simple struture without using Sequential, in order to compatiable the keras
        '''
        super(CensusIncomeModel, self).__init__()
        self.relu = nn.ReLU()
        self.dense =nn.Linear(12,30)
        self.dense_1 =nn.Linear(30,20)
        self.dense_2 =nn.Linear(20,15)
        self.dense_3 =nn.Linear(15,15)
        self.dense_4 =nn.Linear(15,10)
        self.dense_5 =nn.Linear(10,1)
    def forward(self,x):   
        x=self.relu(self.dense(x))
        x=self.relu(self.dense_1(x))
        x=self.relu(self.dense_2(x))
        x=self.relu(self.dense_3(x))
        x=self.relu(self.dense_4(x))
        x=self.relu(self.dense_5(x))
        return x 
    
    def keras_filter(self):
        return self.keras_filter_v2()
    def keras_filter_v1(self):
        layers_name= ["dense","dense_1","dense_2","dense_3","dense_4","dense_5"]
        weights_name= ["weight","bias"]
        import itertools 
        maps = itertools.product(layers_name,weights_name)
        newmaps = ["{}.{}".format(x,y) for x,y in maps ]
        return newmaps 
    
    def keras_filter_v2(self):
        ordered_map = [k for k,v  in self.state_dict().items()]
        return ordered_map 
    

class BankMarketingModel(nn.Module,MainTask):
    def __init__(self,):
        '''
        a foolish and simple struture without using Sequential, in order to compatiable the keras
        '''
        super(BankMarketingModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(12,30),
            nn.ReLU(),
            nn.Linear(30,20),
            nn.ReLU(),
            nn.Linear(20,15),
            nn.ReLU(),
            )
        self.classifier = nn.Sequential(
            nn.Linear(15,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,1),
            nn.Sigmoid(),
            )
        
    def forward(self,x):   
        x=self.feature(x)
        x=self.classifier(x)
        return x 
    
    def keras_filter(self):
        ordered_map = [k for k,v  in self.state_dict().items()]
        # print (ordered_map)
        
        return ordered_map 
    
class BankMarketingModel_plus(nn.Module,MainTask):
    def __init__(self,):
        '''
        a foolish and simple struture without using Sequential, in order to compatiable the keras
        '''
        super(BankMarketingModel_plus, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(12,30),
            nn.ReLU(),
            nn.Linear(30,20),
            nn.ReLU(),
            nn.Linear(20,20), #####plus one layer 
            nn.ReLU(),        #####plus one layer 
            nn.Linear(20,15),
            nn.ReLU(),
            )
        self.classifier = nn.Sequential(
            nn.Linear(15,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,1),
            nn.Sigmoid(),
            )
        
    def forward(self,x):   
        x=self.feature(x)
        x=self.classifier(x)
        return x 
    
    def keras_filter(self):
        ordered_map = [k for k,v  in self.state_dict().items()]
        ordered_map= [k for k in ordered_map if not k.startswith("feature.4") ]
        return ordered_map 
    

class GermanCreditModel(nn.Module,MainTask):
    def __init__(self,):
        '''
        a foolish and simple struture without using Sequential, in order to compatiable the keras
        '''
        super(GermanCreditModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(12,50),
            nn.ReLU(),
            nn.Linear(50,30),
            nn.ReLU(),
            nn.Linear(30,15), 
            nn.ReLU(),        
            nn.Linear(15,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,1),
            nn.Sigmoid(),
            )
        
    def forward(self,x):   
        x=self.model(x)
        return x 
    
    def keras_filter(self):
        ordered_map = [k for k,v  in self.state_dict().items()]
        return ordered_map 
    
    
         
class MultiTaskModel(nn.Module,MainTask):
    def __init__(self,in_channel=12,classes_list=[1,1,4,5]):
        super(MultiTaskModel, self).__init__()
        c1,c2,c3,c4 = classes_list
        self.c1,self.c2,self.c3,self.c4 = c1,c2,c3,c4 
        
        self.feature = nn.Sequential(
            nn.Linear(in_channel,30),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(30,20),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(20,15),
            nn.ReLU(),
            )
        self.classifer_att1 =nn.Sequential(
            nn.Linear(15,15),
            nn.ReLU(),
            nn.Linear(15,10),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(10,c1),
            )
        self.classifer_att2 = nn.Sequential(
                nn.Linear(15,10),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(10,c2),
            )
        
        self.classifer_att3 = nn.Sequential(
            nn.Linear(15,10),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(10,c3),
            )
        
        self.classifer_att4 = nn.Sequential(
            nn.Linear(15,10),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(10,c4),
            )
    def forward(self,x):
        feature =  self.feature(x)
        out1 = self.classifer_att1(feature)
            
        out2 = self.classifer_att2(feature)
        out3 = self.classifer_att3(feature)
        out4 = self.classifer_att4(feature)

        if self.c1==1:
            out1 = F.sigmoid(out1)
        if self.c2==1:
            out2 = F.sigmoid(out2)
        if self.c3==1:
            out3 = F.sigmoid(out3)
        if self.c4==1:
            out4 = F.sigmoid(out4)
        
        return out1,out2,out3,out4 
    
    def keras_filter(self):
        dict_x = self.state_dict()
        # print("dict_x\n",len(dict_x))
        # print("dict",dict_x)
        ordered_name = [k for k,v in dict_x.items() if ("classifer_att2" not in k and "classifer_att3" not in k and "classifer_att4" not in k )] 
        # print(ordered_name)
        return ordered_name 

    def feature_learnable(self,learnable=True):
        for param in self.feature.parameters():
            param.requires_grad = learnable

        
if __name__=="__main__":
    torch.manual_seed(42)
    
    import io 
    from tensorflow import keras 
    model = keras.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=[None,16]),
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dense(15, activation="relu"),
        keras.layers.Dense(15, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    # model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
    # model.summary()
    #
    dummy_input = torch.randn(1,16)
    # print(dummy_input)
    dummy_input_np = dummy_input.numpy()
    # print (dummy_input.mean(),dummy_input.std(),"mean->std")
    
    # torch_model  =CensusIncomeModel()
    # torch_model  = BankMarketingModel_plus()
    torch_model = MultiTaskModel(16,[1,1,4,1])
    with torch.no_grad():
        torch_model.eval()
        dummy_out1 = torch_model(dummy_input)
    # torch.onnx.export(torch_model, dummy_input, "model.onnx")
    
    tf_out_before = model(dummy_input_np)
    
    keras_convert = torch_model.convert_torch_2_keras(Model_keras=model)
    # keras_convert.save("a.h5")
    tf_out = keras_convert(dummy_input_np)
    #
    # print (dummy_out1, tf_out_before,tf_out)

    
    