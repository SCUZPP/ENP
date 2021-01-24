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
import random

from utils import *
from ec import *
from ec_pop_b import *
from pruning import *
import os
    


    
def evaluate(ind, model, args, train_loader, test_loader, val_loader):
    filter_nums = args.filter_nums
    
    solution = np.ones((sum(filter_nums), 1), dtype = np.int)
    
    solution = ind.reshape(ind.shape[0], 1)
    
    solution[-10:] = 1 
    # Prune model according to the solution
    
    
    model_new = prune_model(model, solution, args)
    # Validate
    #只剪枝不微调
    acc, loss = test_forward(val_loader, model_new, args)
    
    #打印微调前的准确率和损失
    pruning_rate = 1 - np.sum(solution) / (sum(filter_nums))
    print('step1微调前:  * accuracy {acc:.2f}, loss {loss:.2f}, pruning {pruning:.2f}'
         .format(acc = acc, loss = loss, pruning = pruning_rate))
         
    
    
    #acc = np.random.randint(100)
    return 100-acc, np.sum(solution)    

def fitness(pop_a, pop_b, pop_c, pivot, model, args, train_loader, test_loader, val_loader):
    #pop_a = copy.deepcopy(population_a)
    #pop_b = copy.deepcopy(population_b)
    
    filter_nums = args.filter_nums
    sample_time = args.sample_time
    error_rates_a = []
    

    used_prob = [[] for i in range(len(filter_nums))]
    used_indexs = [[] for i in range(len(filter_nums))]
    
    #store the fitness value of every ind in population B
    score_pop_b = [[] for i in range(len(filter_nums))]
    len_pop_b = [[] for i in range(len(filter_nums))]
    
    for i in range(len(filter_nums)):
        for j in range(len(args.prob)):
            temp = [0.0 for k in range(args.sub_pop_size_b)]
            temp_len = [0 for k in range(args.sub_pop_size_b)]
            
            score_pop_b[i].append(temp)
            len_pop_b[i].append(temp_len)
            
    #evaluate every ind in population a
    count_c = pivot 
    for ind_a in pop_a:
        #print('inda', ind_a.dec)
        indexs = []
        
        #generate a random index list for every layer
        for j in range(len(filter_nums)):
            temp = [i for i in range(args.sub_pop_size_b)]
            random.shuffle(temp)
            indexs.append(temp)

        #print('indexs', indexs)
        error_rates = []
        filter_res = []        
        #sample N ind from population b
        for time in range(sample_time):
            solution = []

            for layer in range(len(ind_a.dec)):
                #print('layer', layer)
                #print('time', time)
                used_prob[layer].append(ind_a.dec[layer])
                
                index = indexs[layer][time]
                used_indexs[layer].append(index)
                #print('index', index)
                prob = ind_a.dec[layer]
                #print('prob', prob)
                solution.append(pop_b[layer][prob][index].dec.tolist())

            #print('solution', solution)    
            solution = sum(solution, [])    
            solution = np.array(solution)
            #print('solution', solution)
            #print('pop_c[count_c].dec[:]', pop_c[count_c].dec)
            pop_c[count_c].dec[:] = solution
            #print('time', time)
            #print('solution', solution)
            solution.reshape(sum(filter_nums), 1)
            #evalutae the pruned model decoded by solution
            error_rate, filter_re = evaluate(solution, model, args, train_loader, test_loader, val_loader)
            error_rates.append(error_rate)
            error_rates_a.append(error_rate)
            filter_res.append(filter_re)
            
            pop_c[count_c].obj[0] = error_rate
            pop_c[count_c].obj[1] = filter_re
            #print('count_c', count_c)
            count_c += 1
            #print('filter_re', filter_re)
            #ind_a.obj[1] = filter_re
            

        error_rate = sum(error_rates) / len(error_rates)
        filter_re = sum(filter_res) / len(filter_res)
        ind_a.obj[0] = error_rate
        ind_a.obj[1] = filter_re
    
    #print('error_rates_a', error_rates_a)
    #print('used_indexs',  used_indexs)
    
    for r in range(len(pop_a)):
        for layer in range(len(filter_nums)):
            for v in range(sample_time):
                temp = v + r * sample_time 
                #print('temp', temp)
                #print('layer', layer)
                #print('p', p)
                #print('v', v)
                #print('ind_a[r][layer]',  ind_a[r][layer])
                #print('used_indexs[layer][temp]',  used_indexs[layer][temp])
                p = pop_a[r].dec
                #print('pop_a[r].dec', pop_a[r].dec)
                score_pop_b[layer][p[layer]][used_indexs[layer][temp]] += error_rates_a[temp]
                len_pop_b[layer][p[layer]][used_indexs[layer][temp]] += 1
                
    #print('score_pop_b', score_pop_b)
    #print('len_pop_b', len_pop_b)
    
                
    for layer in range(len(filter_nums)):
        for p in range(len(args.prob)):
            for index in range(args.sub_pop_size_b):
                #if the ind in population B has been sampled
                #print(pop_b[layer][p][index].obj[0])
                
                if len_pop_b[layer][p][index] != 0:
                    error_rate = score_pop_b[layer][p][index] / len_pop_b[layer][p][index]
                    
                    if pop_b[layer][p][index].obj[0] != 0.0:
                        #take the average of the previous fitness and the current fitness
                        pop_b[layer][p][index].obj[0] = (error_rate + pop_b[layer][p][index].obj[0]) / 2
                        
                    else:
                        pop_b[layer][p][index].obj[0] = error_rate
                    
                #else the the fitness of ind in population B keeps unchanged  
                #else:
                #    pop_b[layer][p][index].obj[0] = score_pop_b[layer][p][index]
                    
            
    #print('used_indexs', used_indexs)
    return used_prob

            
class Individual_A():
    def __init__(self, gene_length, model, args, train_loader, test_loader, val_loader):
        self.dec = np.zeros(gene_length, dtype=np.uint8)  ## binary
        for i in range(gene_length):
            #self.dec[i] = np.random.randint(0, args.prob[-1] + 1)  # always begin with 1
            self.dec[i] = 0
        #the last fully connected layer keeps unpruned
        self.dec[gene_length - 1] = 0
        self.obj = [0, 0]  # initial obj value will be replaced by evaluate()
        #self.evaluate(model, args, train_loader, test_loader, val_loader)

    def evaluate(self, model, args, train_loader, test_loader, val_loader):
        self.obj[0], self.obj[1] = evaIndividual_a(self.dec, model, args, train_loader, test_loader, val_loader)
        
def initialization_a(pop_size, gene_length, model, args, train_loader, test_loader, val_loader):
    population = []
    for i in range(pop_size):
        ind = Individual_A(gene_length, model, args, train_loader, test_loader, val_loader)
        population.append(ind)
        
    return population
           
class Individual_B():
    def __init__(self, gene_length, pruning_rate, model, args, train_loader, test_loader, val_loader):
        self.dec = np.ones(gene_length, dtype=np.uint8)  ## binary
        zeros = int(gene_length * pruning_rate / 10)
        #print('gene_length', gene_length)
        #print('pruning_rate', pruning_rate)
        #print('zeros', zeros)
        
        for i in range(zeros):
            self.dec[i] = 0  # always begin with 1
        random.shuffle(self.dec)
        
        self.obj = [0.0]  # initial obj value will be replaced by evaluate()
        #self.evaluate(model, args, train_loader, test_loader, val_loader)

    def evaluate(self, model, args, train_loader, test_loader, val_loader):
        self.obj[0] = evaIndividual_b(self.dec, model, args, train_loader, test_loader, val_loader)
        
def initialization_b(pop_size, gene_length, pruning_rate, model, args, train_loader, test_loader, val_loader):
    population = []
    for i in range(pop_size):
        ind = Individual_B(gene_length, pruning_rate, model, args, train_loader, test_loader, val_loader)
        population.append(ind)
        
    return population

class Individual_C():
    def __init__(self, gene_length):
        self.dec = np.ones(gene_length, dtype=np.uint8)  ## binary
        self.obj = [0, 0]  # initial obj value will be replaced by evaluate()
        #self.evaluate(model, args, train_loader, test_loader, val_loader)
        
def initialization_c(pop_size, gene_length):
    population = []
    for i in range(pop_size):
        ind = Individual_C(gene_length)
        population.append(ind)
        
    return population


def get_population(model, args, train_loader, test_loader, val_loader):

    
    args.cuda = not args.no_cuda and torch.cuda.is_available()    
    if args.cuda:
        model.cuda()
    filter_nums = args.filter_nums
    gene_length = sum(filter_nums)
    p_crossover = 1  # crossover probability
    p_mutation = 1  # mutation probability

    # configuration
    #种群数量必须是偶数
    if args.cuda:
        pop_size_a = args.pop_size_a  # Population size
        
    else:
        pop_size_a = 4
        
    n_obj_a = 2  # Objective variable dimensionality
    dec_dim_a = args.gene_length_a  # Decision variable dimensionality
    
    if args.cuda:
        gen = args.gen # Iteration number
        
    else:
        gen = 2

    # Initialization population A
    population_a = initialization_a(pop_size_a, dec_dim_a, model, args, train_loader, test_loader, val_loader)
        
    if args.cuda:
        pop_size_b = args.gene_length_a * len(args.prob) * args.sub_pop_size_b  # Population size
    else:
        pop_size_b = args.gene_length_a * len(args.prob) * 2
        
    n_obj_b = 1  # Objective variable dimensionality
    dec_dim_b = args.filter_nums  # Decision variable dimensionality
    
 
    # Initialization population B
    population_b = [[] for j in range (len(filter_nums))]
    
    for j in range (len(filter_nums)):
        for i in range(len(args.prob)):                                     
            population_b[j].append (initialization_b(args.sub_pop_size_b, dec_dim_b[j], args.prob[i], model, args, train_loader, test_loader, val_loader))
    
                    
    #fitness(population_a, population_b, model, args, train_loader, test_loader, val_loader)
    
    '''                                          
    for ind in population_a:
        print('a0.dec', ind.dec)
        print('a0.0', ind.obj[0])
        #print('a0.1', ind.obj[1])
      
    #layer
    for inds in population_b:
        #prob
        for ind in inds:
            #
            for ins in ind:
                print('b0.dec', ins.dec)
                print('b0.0', ins.obj[0])
    ''' 
    
    pop_size_c = pop_size_a * args.sample_time * 2 
    population_c = initialization_c(pop_size_c, gene_length)
    pivot =  pop_size_a * args.sample_time
    
    g_begin = 0
    target_dir_a = 'Results1_A_vgg%s/' % (args.depth)
    target_dir_b = 'Results1_B_vgg%s/' % (args.depth)
    target_dir_c = 'Results1_C_vgg%s/' % (args.depth)
    
    path_save_a = './' + target_dir_a
    path_save_b = './' + target_dir_b
    path_save_c = './' + target_dir_c
    
    
    for g in range(g_begin + 1, gen + 1):
        # generate reference lines and association  
        pivot =  0
        used_probs = fitness(population_a, population_b, population_c, pivot, model, args, train_loader, test_loader, val_loader)
        offspring_b = copy.deepcopy(population_b)
        #Variation for population B
        #the last fully connected layer keeps unchanged
        for j in range (len(filter_nums) - 1):
            used_prob = used_probs[j]
            #print('used_index0', used_index)
            used_prob.sort()
            #print('used_prob', used_prob)
            #print('layer', j)
            for i in range(len(args.prob)): 
                # if the probability has been used in pop A
                if i in used_prob:
                    #print('prob', i)
                    offspring_b[j][i][:] = varation_b(population_b[j][i], args, dec_dim_b[j])


        offspring_a = variation(population_a, population_b, p_crossover, p_mutation, model, args, train_loader, test_loader, val_loader)
        #evaluation the fitness of offspring A after the variation of population B
        pivot =  pop_size_a * args.sample_time
        used_probs = fitness(population_a, offspring_b, population_c, pivot, model, args, train_loader, test_loader, val_loader)
        # P+Q
        population_a.extend(offspring_a)
       
        # Environmental Selection
        population_a = environmental_selection(population_a, pop_size_a)
        population_b = environmental_selection_b(population_b, offspring_b, args)
        

        # generation
        print('Gen:', g)

        #Save population
        if g == 1:
            with open(path_save_a + "population-{}.pkl".format(g), 'wb') as f:
                pickle.dump(population_a, f) 
                
            with open(path_save_b + "population-{}.pkl".format(g), 'wb') as f:
                pickle.dump(population_b, f)
                
            with open(path_save_c + "population-{}.pkl".format(g), 'wb') as f:
                pickle.dump(population_c, f) 

                
        if g % 10 == 0:
            with open(path_save_a + "population-{}.pkl".format(g), 'wb') as f:
                pickle.dump(population_a, f) 
                
            with open(path_save_b + "population-{}.pkl".format(g), 'wb') as f:
                pickle.dump(population_b, f) 
                
            with open(path_save_c + "population-{}.pkl".format(g), 'wb') as f:
                pickle.dump(population_c, f) 
                    
    
    print('step1 final population')

    population_init = []
    
    for ind in population_a:
        temp = ind.dec
        #temp = recode2binary(temp)
        population_init.append(temp)
        pruning_rate = 1 - ind.obj[1] / (sum(filter_nums)) 
        print(' * accuracy {acc:.2f}, pruning {pruning:.2f}'
             .format(acc = 100 - ind.obj[0], pruning=pruning_rate))
     
    #以文件形式存储
    save_path_a = 'population_dict/population_dict_a_vgg%s.pkl' % (args.depth)
                                                                                 
    pickle.dump(population_a, open(save_path_a, 'wb'))

    save_path_b = 'population_dict/population_dict_b_vgg%s.pkl' % (args.depth)
                                                                                 
    pickle.dump(population_b, open(save_path_b, 'wb'))

      
        
