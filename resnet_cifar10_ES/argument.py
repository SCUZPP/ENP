class Args:
    def __init__(self):
    
        self.depth = 32
        self.lr = 0.1
        self.ft_lr = 0.0001
        self.retrain_lr = 0.1
        self.epochs = 160
        self.batch_size = 64
        self.test_batch_size = 1000
        self.val_batch_size = 1000
        self.momentum = 0.9
        self.dataset = 'cifar10'
        self.arch = 'resnet'
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.weight_decay = 1e-4
        self.retrain = False
        self.retrain = False
        self.cpu = True
        self.half = False
        self.print_freq = 20
        self.sample_time = 5

        if self.depth == 32:
            self.ft_epochs = 300
            self.re_epochs = 300
            self.temp_epoch = 60
            self.filter_nums = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
            #self.filter_nums = [4, 10, 10, 20, 20, 10]
            self.gene_a = [16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64]
            #self.gene_a =[10, 20]
            self.gene_length_a = len(self.gene_a)
            self.gene_length = sum(self.filter_nums)
            self.filter_length = sum(self.filter_nums)
            #self.sub_pop_size_b = 2
            self.sub_pop_size_b = 10
            self.pop_size_a = 50
            #self.pop_size_a = 2
            self.gen = 100
            #self.gen = 2
            self.prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            #self.prob = [0, 1, 2, 3, 4]
            self.filter_length = sum(self.filter_nums)
            #self.filter_length = 40
            self.accuracy = 93.25
            
        elif self.depth == 56:
        
            self.ft_epochs = 300
            self.re_epochs = 300
            self.temp_epoch = 60
            self.filter_nums = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
            self.gene_a = [16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64]
            self.gene_length_a = len(self.gene_a)
            self.gene_length = sum(self.filter_nums)
            self.filter_length = sum(self.filter_nums)
            #self.sub_pop_size_b = 2
            self.sub_pop_size_b = 5
            self.pop_size_a = 4
            self.gen = 2
            self.prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            #self.prob = [0, 1, 2, 3, 4]
            self.filter_length = sum(self.filter_nums)
            #self.filter_length = 40
            self.accuracy = 94.080
            
        
        elif self.depth == 110:
            self.ft_epochs = 300
            self.re_epochs = 300
            self.temp_epoch = 60
            self.filter_nums = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
            self.gene_a = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
            self.gene_length_a = len(self.gene_a)
            self.gene_length = sum(self.filter_nums)
            self.filter_length = sum(self.filter_nums)
            #self.sub_pop_size_b = 2
            self.sub_pop_size_b = 10
            self.pop_size_a = 50
            self.gen = 100
            self.prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            #self.prob = [0, 1, 2, 3, 4]
            self.filter_length = sum(self.filter_nums)
            #self.filter_length = 40
            self.accuracy = 94.48
        