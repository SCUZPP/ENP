import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import copy
import argument
from vgg import *
import step1 
import step2 
import logging




#load data
args = argument.Args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
logger = logging.getLogger(__name__)

basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(basic_format)    
log_path = 'depth%s_log.txt' % (args.depth)
print('log_path {}'.format(log_path))
handler = logging.FileHandler(log_path, 'a', 'utf-8')
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info('log_path {}'.format(log_path))
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_set = datasets.CIFAR10('../data.cifar10', train=True, download=True, 
                             transform=transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

    
if args.cuda:
    train_set = train_set
    valset, train_set = torch.utils.data.random_split(train_set, [5000, 45000])
    
else:    
    train_set, rand = torch.utils.data.random_split(train_set, [500, 49500])
    
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)


testset = datasets.CIFAR10('../data.cifar10', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ]))

if args.cuda:
    testset = testset
    #valset, testset = torch.utils.data.random_split(testset, [5000, 5000])   
else:  
    testset, rand = torch.utils.data.random_split(testset, [1000, 9000])
    testset, valset = torch.utils.data.random_split(testset, [500, 500])

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(valset, batch_size=args.val_batch_size, shuffle=True, **kwargs)


#加载模型和数据
def construct_model_load(): 
    model = vgg19_bn()
    if args.cuda:
        checkpoint = torch.load('../logs/checkpoint19.pth.tar')
        newdict = dict()
        for name, para in checkpoint['state_dict'].items():
            name = name.replace('module.','')
            newdict[name] = para  


    else:
        checkpoint = torch.load('../logs/checkpoint19.pth.tar', map_location=torch.device('cpu'))   
        newdict = dict()
        for name, para in checkpoint['state_dict'].items():
            name = name.replace('module.','')
            newdict[name] = para  
    
    model.load_state_dict(newdict)
    
    return model


def model_load():

    model = vgg19_bn()
    
    return model

if args.retrain:
    model = model_load()
    
else:
    model = construct_model_load()

#step1.get_population(model, args, train_loader, test_loader, val_loader)


step2.get_population(model, args, train_loader, test_loader, val_loader)


