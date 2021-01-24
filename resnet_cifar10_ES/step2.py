import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import copy
import pickle

from utils import *
from ec import *
from pruning import *
from cmp import * 


def evaIndividual(ind, model, args, train_loader, test_loader, val_loader):
    filter_nums = args.filter_nums
    gene_length = sum(filter_nums)
    #？
    acc_30 = []
    
    solution = np.ones((sum(filter_nums), 1), dtype = np.int)
    
    solution = ind.reshape(ind.shape[0], 1)
    solution[-10:] = 1 
    
    # Prune model according to the solution
    model_new = prune_model(model, solution, args)
    flop = print_model_param_flops(model_new)
    para = print_model_param_nums(model_new)
    
    flop_old = print_model_param_flops(model)
    para_old = print_model_param_nums(model)
    
    flop_rate = 1 - flop / flop_old
    para_rate = 1 - para / para_old
    
    print('flop_rate', flop_rate)
    print('para_rate', para_rate)
    
    #计算微调前的准确率
    #这一步待考量，因为计算微调前的准确率太耗时
    #acc, loss = test_forward(val_loader, model_new, args)
    #打印微调前的准确率和损失
    
    pruning_rate = 1 - np.sum(solution) / (sum(filter_nums))
    print('pruning_rate', pruning_rate)
    
    #print('step2微调前:  * accuracy {acc:.2f}, loss {loss:.2f}, pruning {pruning:.2f}'
    #     .format(acc = acc, loss = loss, pruning=pruning_rate))
    
    acc, loss, acc_30 = train_forward(train_loader, test_loader, val_loader, model_new, args)  # test_forward(model_new)
    #acc, loss = test_forward(val_loader, model_new, criterion, solution)
    #acc, loss, acc_30 = 0, 0, 0
    #print(acc, pruning_rate)
    #打印微调前的准确率和损失
    print('step2微调后:  * accuracy {acc:.2f}, pruning {pruning:.2f}'
         .format(acc = acc, pruning=pruning_rate))
    
    return 100-acc, np.sum(solution), acc_30

class Individual():
    
    def __init__(self, gene_length, dec, p_b, model, args, train_loader, test_loader, val_loader):
        filter_nums = args.filter_nums
        self.dec = np.zeros(gene_length, dtype=np.uint8)  ## binary
        self.acc_30 = []
        solution = []
        #the first conv layer
        temp = [1] * int(filter_nums[0])
        solution.append(temp)
        
        for layer in range(len(dec)):
            #the first conv layer in residual block
            prob = dec[layer]
            max_index = 0
            
            if prob != 0:
                for index in range(len(p_b[layer][prob])):
                    if p_b[layer][prob][index].obj[0] > p_b[layer][prob][max_index].obj[0]:
                        max_index = index
            else:
                max_index = 0          

            solution.append(p_b[layer][prob][max_index].dec.tolist())
            
            #the second conv layer in residual block
            length = len(p_b[layer][prob][max_index].dec.tolist())
            temp = [1] * length               
            solution.append(temp)  
            
        #the last fully connected layer
        temp = [1] * int(filter_nums[-1])
        solution.append(temp)
        
        solution = sum(solution, [])    
        solution = np.array(solution)
        self.dec = solution
        
        self.obj = [0, 0]  # initial obj value will be replaced by evaluate()
        self.evaluate(model, args, train_loader, test_loader, val_loader) 

    def evaluate(self, model, args, train_loader, test_loader, val_loader):
        self.obj[0], self.obj[1], self.acc_30 = evaIndividual(self.dec, model, args, train_loader, test_loader, val_loader)
        
def initialization(pop_size, gene_length, model, args, train_loader, test_loader, val_loader, p_a, p_b, target_dir):
    population = []
    acc_30 = []
    #count = 0
    for i in range(len(p_a)):
        dec = p_a[i].dec
        ind = Individual(gene_length, dec, p_b, model, args, train_loader, test_loader, val_loader)
        population.append(ind)
        acc_30.append(ind.acc_30)
        path_save = './' + target_dir
        
        with open(path_save + "population_300_7_1.pkl", 'wb') as f:
                pickle.dump(population, f) 
       
        with open(path_save + "population_30_7_1.pkl", 'wb') as f:
                pickle.dump(acc_30, f) 


        
    return population
           

def get_population(model, args, train_loader, test_loader, val_loader):
    
    open_path_a = 'population_dict/population_dict_a_resnet%s.pkl' % (args.depth)
    open_path_b = 'population_dict/population_dict_b_resnet%s.pkl' % (args.depth)
    p_a = []
    
    p_as = pickle.load(open(open_path_a, 'rb'))
    #p_a.append(p_as[4])
    p_a.append(p_as[7])
    #p_a.append(p_as[8])
    
    #p_a.append(p_as[0])
    
    p_b = pickle.load(open(open_path_b, 'rb'))
    target_dir = 'Results2_resnet%s/' % (args.depth)
                
 
    # configuration
    #种群数量先设为8
    pop_size = len(p_a)  # Population size
    n_obj = 2  # Objective variable dimensionality
    filter_nums = args.filter_nums
    dec_dim = sum(filter_nums)  # Decision variable dimensionality

    # Initialization
    population = initialization(pop_size, dec_dim, model, args, train_loader, test_loader, val_loader, p_a, p_b, target_dir)
    

    print('final population')
    for ind in population:
        
        pruning_rate = 1 - ind.obj[1] / (sum(filter_nums)) 
        
        print(' *accuracy {acc:.2f}, pruning {pruning:.2f}'
             .format(acc = 100 - ind.obj[0], pruning=pruning_rate))
     