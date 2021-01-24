#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
from math import *
from scipy.spatial.distance import cdist

# Domination check
'''def dominate(p, q):
    result = False
    for i, j in zip(p.obj, q.obj):
        if i < j:  # at least less in one dimension
            result = True
        elif i > j:  # not greater in any dimension, return false immediately
            return False
    return result

def fitness(pop_a, pop_b, model, args, train_loader, test_loader, val_loader):
    #pop_a = copy.deepcopy(population_a)
    #pop_b = copy.deepcopy(population_b)
    
    filter_nums = args.filter_nums
    sample_time = args.sample_time
    error_rates_a = []
    


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
                
                index = indexs[layer][time]
                used_indexs[layer].append(index)
                #print('index', index)
                prob = ind_a.dec[layer]
                #print('pop_b', pop_b[layer][prob][index].dec)
                solution.append(pop_b[layer][prob][index].dec.tolist())

            #print('solution', solution)    
            solution = sum(solution, [])    
            solution = np.array(solution)
            #print('time', time)
            #print('solution', solution)
            solution.reshape(sum(filter_nums), 1)
            #evalutae the pruned model decoded by solution
            error_rate, filter_re = evaluate(solution, model, args, train_loader, test_loader, val_loader)
            error_rates.append(error_rate)
            error_rates_a.append(error_rate)
            filter_res.append(filter_re)
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
                if len_pop_b[layer][p][index] != 0:
                    error_rate = score_pop_b[layer][p][index] / len_pop_b[layer][p][index]
                    pop_b[layer][p][index].obj[0] = error_rate
                    
                #else the the fitness of ind in population B is set to 0    
                else:
                    pop_b[layer][p][index].obj[0] = score_pop_b[layer][p][index]
                    
            
    #print('used_indexs', used_indexs)

'''
def dominate(p,q):
    result = False
    #if p is feasible and q is infeasible
    if p.obj[0] <= 50 and q.obj[0] > 50:
        return True
    
    #if p in infeasible and q is feasible
    if p.obj[0] > 50 and q.obj[0] <= 50:
        return False
    
    #if p is feasible and q is feasible
    if p.obj[0] <= 50 and q.obj[0] <= 50:
        for i, j in zip(p.obj, q.obj):
            if i < j:  # at least less in one dimension
                result = True
                
            elif i > j:  # not greater in any dimension, return false immediately
                return False
            
    #if p is infeasible and q is infeasible
    if p.obj[0] > 50 and q.obj[0] > 50:
        if p.obj[0] > q.obj[0]:
            return False
        
        elif p.obj[0] < q.obj[0]:
            return True
        
        else:
            if p.obj[1] < q.obj[1]:
                return True
            
            else:
                return False
            
    return result 

def non_dominate_sorting(population):
    # find non-dominated sorted
    dominated_set = {}
    dominating_num = {}
    rank = {}
    for p in population:
        dominated_set[p] = []
        dominating_num[p] = 0

    sorted_pop = [[]]
    rank_init = 0
    for i, p in enumerate(population):
        for q in population[i + 1:]:
            if dominate(p, q):
                dominated_set[p].append(q)
                dominating_num[q] += 1
            elif dominate(q, p):
                dominating_num[p] += 1
                dominated_set[q].append(p)
        # rank 0
        if dominating_num[p] == 0:
            rank[p] = rank_init # rank set to 0
            sorted_pop[0].append(p)

    while len(sorted_pop[rank_init]) > 0:
        current_front = []
        for ppp in sorted_pop[rank_init]:
            for qqq in dominated_set[ppp]:
                dominating_num[qqq] -= 1
                if dominating_num[qqq] == 0:
                    rank[qqq] = rank_init + 1
                    current_front.append(qqq)
        rank_init += 1

        sorted_pop.append(current_front)

    return sorted_pop


class Individual():
    def __init__(self, gene_length, count, model, args, train_loader, test_loader, val_loader, original_filter):
        self.dec = np.zeros(gene_length, dtype=np.uint8)  ## binary
        for i in range(gene_length):
             self.dec[i] = np.random.randint(0, 10)  # random binary code
        self.obj = [0, 0]  # initial obj value will be replaced by evaluate()
        #self.evaluate(model, args, train_loader, test_loader, val_loader, original_filter)

    def evaluate(self, model, args, train_loader, test_loader, val_loader, original_filter):
        # self.obj[0], self.obj[1] = evaCNN(self.dec)
        print('test evaluate ec')
        self.obj[0], self.obj[1] = 0, 0


def initialization(pop_size, gene_length, model, args, train_loader, test_loader, val_loader, original_filter):
    population = []
    count = 0
    for i in range(pop_size):
        ind = Individual(gene_length, count, model, args, train_loader, test_loader, val_loader, original_filter)
        population.append(ind)
        count += 1
    return population


def evaluation(off_a, pop_b, model, args, train_loader, test_loader, val_loader, original_filter):
    # Evaluation
    fitness(off_a, pop_b, model, args, train_loader, test_loader, val_loader)
    


# one point crossover
#只需要改交叉和变异的编码，不需要改变啊，调用评价函数时会自动解码
#需要改变交叉和变异的方法，现在基因位数是30，每一位是0-199之间的整数
def one_point_crossover(p, q):
    gene_length = len(p.dec)
    child1 = np.zeros(gene_length, dtype=np.uint8)
    child2 = np.zeros(gene_length, dtype=np.uint8)
    k = np.random.randint(gene_length)
    child1[:k] = p.dec[:k]
    child1[k:] = q.dec[k:]

    child2[:k] = q.dec[:k]
    child2[k:] = p.dec[k:]

    return child1, child2


# Bit wise mutation
#对基因每一位以一定概率变异，变异操作是选取（0-9）中不等于原数值的数替换
def bitwise_mutation(p, p_m, size, args):
    gene_length = len(p.dec)
    population_size = size
    p_mutation = p_m / gene_length
    #p_mutation = 0.1  ## constant mutation rate
    # the last fully connected layer keeps unchanged
    for i in range(gene_length - 1):
        if np.random.random() < p_mutation:
            #k = np.random.randint(0, 10)
            k = np.random.randint(0, args.prob[-1] + 1)
            while k == p.dec[i]:
                k = np.random.randint(0, args.prob[-1] + 1)
            p.dec[i] = k
    return p


# Variation (Crossover & Mutation)
def variation(pop_a, pop_b, p_crossover, p_mutation, model, args, train_loader, test_loader, val_loader):
    offspring = copy.deepcopy(pop_a)
    len_pop = int(np.ceil(len(pop_a) / 2) * 2) 
    candidate_idx = np.random.permutation(len_pop)
    population_size = len(pop_a)

    # Crossover
    for i in range(int(len_pop/2)):
        if np.random.random()<=p_crossover:
            individual1 = offspring[candidate_idx[i]]
            individual2 = offspring[candidate_idx[-i-1]]
            [child1, child2] = one_point_crossover(individual1, individual2)
            offspring[candidate_idx[i]].dec[:] = child1
            offspring[candidate_idx[-i-1]].dec[:] = child2

    # Mutation
    for i in range(len_pop):
        individual = offspring[i]
        offspring[i] = bitwise_mutation(individual, p_mutation, population_size, args)

    # Evaluate offspring
    #offspring = evaluation(offspring, pop_b, model, args, train_loader, test_loader, val_loader, original_filter)

    return offspring


# Crowding distance
def crowding_dist(population):
    pop_size = len(population)
    crowding_dis = np.zeros((pop_size,))

    obj_dim_size = len(population[0].obj)
    # crowding distance
    for m in range(obj_dim_size):
        obj_current = [x.obj[m] for x in population]
        sorted_idx = np.argsort(obj_current)  # sort current dim with ascending order
        obj_max = np.max(obj_current)
        obj_min = np.min(obj_current)

        # keep boundary point
        crowding_dis[sorted_idx[0]] = np.inf
        crowding_dis[sorted_idx[-1]] = np.inf
        for i in range(1, pop_size - 1):
            crowding_dis[sorted_idx[i]] = crowding_dis[sorted_idx[i]] + \
                                                      1.0 * (obj_current[sorted_idx[i + 1]] - \
                                                             obj_current[sorted_idx[i - 1]]) / (obj_max - obj_min)
            
    return crowding_dis




def environmental_selection(population, n):
    pop_sorted = non_dominate_sorting(population)
    selected = []
    for front in pop_sorted:
        if len(selected) < n:
            if len(selected) + len(front) <= n:
                selected.extend(front)
            else:
                # select individuals according crowding distance here
                crowding_dst = crowding_dist(front)
                k = n - len(selected)
                dist_idx = np.argsort(crowding_dst, axis=0)[::-1]  # descending order, large rank small angel
                for i in dist_idx[:k]:
                    selected.extend([front[i]])
                break
    return selected