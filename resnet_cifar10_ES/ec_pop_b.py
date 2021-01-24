#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
from math import *
from scipy.spatial.distance import cdist

class Individual():
    def __init__(self, gene_length, model, args, train_loader, test_loader, val_loader):
        self.dec = np.zeros(gene_length, dtype=np.uint8)  ## binary
        for i in range(gene_length):
            self.dec[i] = np.random.randint(0, 2)  # random binary code
        self.obj = [0]  # initial obj value will be replaced by evaluate()
        #self.evaluate(model, args, train_loader, test_loader, val_loader)

    def evaluate(self, model, args, train_loader, test_loader, val_loader):
        # self.obj[0], self.obj[1] = evaCNN(self.dec)
        print('test evaluate ec')
        self.obj[0]= 0


def initialization(pop_size, gene_length, model, args, train_loader, test_loader, val_loader):
    population = []
    count = 0
    for i in range(pop_size):
        ind = Individual(gene_length, model, args, train_loader, test_loader, val_loader)
        population.append(ind)
    return population



def prob(pop):
    total_fit = 0
    pro = []
    pop_size = 0
    
    for ind in pop:
        total_fit += ind.obj[0]
        pop_size += 1
    
    #if sub pop B has not ever been sampled
    if total_fit == 0:
        total_fit = 1
        temp = 1 / pop_size
        
        for ind in pop:
            pro.append(temp) 
            
    else:
        for ind in pop:
            pro.append(ind.obj[0] / total_fit)
        
    for count in range(len(pro)):
        if count == 0:
            pro[count] = pro[count]
        else:
            pro[count] = pro[count] + pro[count - 1]
            
    return pro

# one point crossover
def one_point_crossover_b(p, q, gene_length):
    child1 = np.zeros(gene_length, dtype=np.uint8)
    child2 = np.zeros(gene_length, dtype=np.uint8)
    t1 = np.random.randint(gene_length)
    t2 = np.random.randint(gene_length)
    k2 = max(t1, t2)
    k1 = min(t1, t2)
    child1[:k1] = p.dec[:k1]
    child1[k1:k2] = q.dec[k1:k2]
    child1[k2:] = p.dec[k2:]

    child2[:k1] = q.dec[:k1]
    child2[k1:k2] = p.dec[k1:k2]
    child2[k2:] = q.dec[k2:]


    return child1, child2


# Bit wise mutation
'''
def bitwise_mutation_b(p, gene_length):
    child = np.zeros(gene_length, dtype=np.uint8)
    child = p.dec
    
    #point to the first index of 0
    index = -1

    for i in range(gene_length):
        if child[i] == 0:
            index = i 
            break
            
    #if pruning rate is equal to 0
    if index == -1:
        return child
        
    t = np.random.randint(gene_length)
    temp = child[index]
    
    #print('child[t]', child[t])
    child[index] = child[t]
    child[t] = temp      
    return child
'''
def bitwise_mutation_b(p, gene_length):
    p_m = 1.0
    child = np.zeros(gene_length, dtype=np.uint8)
    child = p.dec
    #p_mutation = p_m / gene_length
    p_mutation = 0.1  ## constant mutation rate
    # the last fully connected layer keeps unchanged
    for i in range(gene_length - 1):
        if np.random.random() < p_mutation:
            #k = np.random.randint(0, 10)
            k = i + 1
            while child[k] == child[i] and k < gene_length - 1:
                k = k + 1
            temp = child[k]
            child[k] = child[i]
            child[i] = temp
   
    return child

def varation_b(pop, args, gene_length):
    pop_size = len(pop)
    offspring = copy.deepcopy(pop)
    #generate the next population
    k =  0
    while k < pop_size:

        child = copy.deepcopy(offspring[k])

        string = bitwise_mutation_b(child, gene_length)
        child.dec[:] = string
        #print('new', child.dec)

        child.obj[0] = 0.0
        offspring[k] = child
        k += 1
            
    #for ind in offspring:
    #    print(ind.obj[0])
        
    return offspring  

def environmental_selection_b(population_b, offspring_b, args):
    pop = copy.deepcopy(population_b)
    for layer in range(len(args.gene_a)):
        for p in range(len(args.prob)):
            for index in range(args.sub_pop_size_b):
                if population_b[layer][p][index].obj[0] < offspring_b[layer][p][index].obj[0]:
                    pop[layer][p][index] = copy.deepcopy(offspring_b[layer][p][index])
                else:
                    pop[layer][p][index] = copy.deepcopy(population_b[layer][p][index])
    return pop
