import sys,os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import torch 
import itertools
import numpy as np 
import time 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

def ids_percentage(sample_round, num_gen, num_attribs, protected_attribs, constraint, model,name="C-a",gene_id_list=[], **kwargs):
    # compute the percentage of individual discriminatory instances with 95% confidence
    statistics = np.empty(shape=(0, ))
    count_s = []
    t= []
    for i in range(sample_round):
        start = time.time ()
        
        gen_id = purely_random(num_attribs, protected_attribs, constraint, model, num_gen,**kwargs)
        percentage = len(gen_id) / num_gen
        # print ("fairness: ","gen_id",len(gen_id),"num_gen", num_gen)
        count_s.append(len(gen_id))
        statistics = np.append(statistics, [percentage], axis=0)
        end  =time.time ()
        t .append( end-start )
        # print (t[-1],"timecost in 1 round ")
        
        gene_id_list.extend(gen_id)
        
    avg = np.average(statistics)
    avg_count = np.average(count_s)
    
    std_dev = np.std(statistics)
    interval = 1.960 * std_dev / np.sqrt(sample_round)
    print('The percentage of individual discriminatory instances with .95 confidence:',\
           avg, '±', interval, " ,the avg gen count: ",avg_count,"/",num_gen, f" the time cost {sample_round} round each attr is:",np.sum(t))

    return {
                f"f/{name}/avg":round(float(avg),5),
                # f"f/{name}/std":round(float(interval),5),
                f"fc/{name}":avg_count,
                f"timecost/{name}":np.mean(t),
        }
#
def eval_fairness(sample_round=10, num_gen=100, pre_census_income=None,  model=None,gene_id_dict={},**kwargs):

    protect_attr_default= [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0,6]), ('C-a&g', [0,7]), ('C-r&g', [6,7])]
    # protect_attr_default= [ ('C-a', [0]), ('C-r', [6]), ('C-g', [7]),  ('C-a&g', [0,7]) ]
    # protect_attr_default= [ ('C-g', [7]), ]
    protected_attribs = kwargs.get("protected_attribs",protect_attr_default)
    
    ret = {}
    with torch.no_grad():
        for benchmark, protected_attribs in protected_attribs:
            gene_id_dict[benchmark]=[]
            
            print(benchmark, ':')
            info = ids_percentage(
                sample_round=sample_round, 
                num_gen=num_gen, 
                num_attribs=len(pre_census_income.X[0]),
                protected_attribs=protected_attribs,
                 constraint=pre_census_income.constraint, 
                 model=model,
                 name=benchmark,
                 gene_id_list=gene_id_dict[benchmark],
                 **kwargs,
                )
            ret .update( info)
    
    return ret  


def similar_set(x, num_attribs, protected_attribs, constraint,**kwargs):
    # find all similar inputs corresponding to different combinations of protected attributes with non-protected attributes unchanged
    protected_domain = []
    for i in protected_attribs:
        protected_domain = protected_domain + [list(range(int(constraint[i][0]), int(constraint[i][1])+1))]
    all_combs = list(itertools.product(*protected_domain))
    all_combs_tensor = torch.tensor(all_combs)
    all_combs_tensor.unsqueeze_(dim=0)
    all_combs_tensor = all_combs_tensor.repeat(x.shape[0],1,1)
    
    if x.ndim<=2 :
        x.unsqueeze_(dim=1)
    assert x.ndim==3,x.shape
    
    similar_x =x.repeat(1,len(all_combs),1)#x.expand(len(all_combs),*x.shape).clone()# torch.empty(shape=(0, num_attribs))
    for i,p in enumerate(protected_attribs):
        similar_x [...,p].copy_( all_combs_tensor[...,i].data)
        
    return similar_x#.float()

def is_discriminatory(x, similar_x, model,**kwargs):
    # identify whether the instance is discriminatory w.r.t. the model
    # with torch.no_grad():
    # print (x.shape,x.dtype,similar_x.shape,similar_x.dtype,"x--->similar_x")
    assert x.ndim==2 ,("expect a batch",x.shape)
    output_perform = kwargs.get("output_perform",lambda x:x[0])
    y_pred =model(x.to(device))
    y_pred = output_perform(y_pred)
    y_pred = y_pred>0.5 
    
    sim_y_pred = model(similar_x.to(device))
    sim_y_pred = output_perform(sim_y_pred)
    sim_y_pred = sim_y_pred>0.5 
    
    if y_pred.shape!=sim_y_pred.shape:
        y_pred = y_pred.repeat(len(sim_y_pred),1)
        # y_pred= y_pred.expand(sim_y_pred.shape[0],*y_pred.shape)
    
    assert sim_y_pred.shape== y_pred.shape ,(y_pred.shape,"y_pred",sim_y_pred.shape,"sim_y_pred")
    sum_c =  torch.sum(y_pred !=sim_y_pred).item()
    return  sum_c>0 , y_pred !=sim_y_pred

def purely_random(num_attribs, protected_attribs, constraint, model, gen_num,**kwargs):
        
    gen_id = np.empty(shape=(0, num_attribs))
    
    x_picked_list= []
    for i in range(gen_num):
        x_picked = [0] * num_attribs
        for a in range(num_attribs):
            x_picked[a] = np.random.randint(constraint[a][0], constraint[a][1]+1)
        
        x_picked_list.append(x_picked)
    
    t1 = time.time()
    
    collect_xpicked = []
    collect_xpicked_simset = []
    for x_picked in x_picked_list:
        x_picked =torch.tensor(x_picked).float()
        x_picked.unsqueeze_(dim=0)
        
        x_picked_simset= similar_set(x_picked, num_attribs, protected_attribs, constraint,**kwargs)
        
        x_picked = x_picked.repeat(1,x_picked_simset.shape[1],1)

        collect_xpicked.append(x_picked)
        collect_xpicked_simset.append(x_picked_simset)
        
    xpicked = torch.cat(collect_xpicked)
    xpicked = xpicked.view(-1,xpicked.shape[-1])
    
    xpicked_simset= torch.cat(collect_xpicked_simset)
    xpicked_simset = xpicked_simset.view(-1,xpicked_simset.shape[-1])
    is_dis ,dis_idx = is_discriminatory(xpicked,xpicked_simset, model ,**kwargs)
    dis_idx = dis_idx.squeeze_(dim=-1)
    
    xpicked_discrimary =   xpicked[dis_idx].long()
    collect_list = torch.unique(xpicked_discrimary,dim=0)
    
    t2 = time.time ()
    # print ("unique collect_list",len(collect_list), "gen_num",gen_num)
    # print ("for loop", t2-t1)
    return collect_list


def _parse_range_str(s):
    '''
    f('1,2,5-7,10')
    [1, 2, 5, 6, 7, 10]
    
    f('1,3-7,10,11-15')
    [1, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15]
    '''
    return sum(((list(range(*[int(j) + k for k,j in enumerate(i.split('-'))]))
         if '-' in i else [int(i)]) for i in s.split(',')), [])
    
def parse_config(epoch,config_str):
    config_str= config_str.strip().lower().replace("\n",";")
    
    for config in config_str.split(";") :
        if config =="":continue
        clist= config.split("|")
        # print (clist,clist[0],"clist[0]",epoch)
        time_list= _parse_range_str(clist[0])
        if epoch not in time_list:
            continue 
        assert len(clist)==4
        print (clist[1],"--->",clist)
        return clist[1]=="true",float(clist[2]),float(clist[3])
    return False,1,1 #default 

def parse_config_multi(epoch,config_str):
    config_str= config_str.strip().lower().replace("\n",";")
    
    for config in config_str.split(";") :
        if config =="":continue
        clist= config.split("|")
        # print (clist,clist[0],"clist[0]",epoch)
        time_list= _parse_range_str(clist[0])
        if epoch not in time_list:
            continue 
        # assert len(clist)==4
        print (clist[1],"--->",clist)
        return clist[1]=="true",float(clist[2]),float(clist[3]),float(clist[4]),float(clist[5])
    return False,1,0,0,0 #default 


    