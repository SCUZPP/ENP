class Args:
    def __init__(self):
    
        self.depth = 50
        self.lr = 0.1
        self.ft_lr = 0.0001
        self.retrain_lr = 0.4
        self.epochs = 100
        self.batch_size = 1024
        self.test_batch_size = 128
        self.val_batch_size = 128
        self.momentum = 0.9
        self.dataset = 'imagenet'
        self.arch = 'resnet50'
        self.no_cuda = False
        self.num_classes = 100
        self.seed = 1
        self.log_interval = 10
        self.weight_decay = 1e-4
        self.retrain = False
        self.cpu = True
        self.half = False
        self.workers = 16
        self.print_freq = 1
        self.sample_time = 4
        
        if self.depth == 18:
            self.basic_block = 2
            self.ft_epochs = 300
            self.re_epochs = 300
            self.temp_epoch = 60
            self.filter_nums = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 10]
            #self.filter_nums = [4, 10, 10, 20, 20, 10]
            self.gene_a = [64, 64, 128, 128, 256, 256, 512, 512]
            #self.gene_a =[10, 20]
            self.gene_length_a = len(self.gene_a)
            self.gene_length = sum(self.filter_nums)
            self.filter_length = sum(self.filter_nums)
            #self.sub_pop_size_b = 2
            self.sub_pop_size_b = 5
            self.pop_size_a = 2
            #self.pop_size_a = 2
            self.gen = 2
            #self.gen = 2
            self.prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            #self.prob = [0, 1, 2, 3, 4]
            self.filter_length = sum(self.filter_nums)
            #self.filter_length = 40
            self.accuracy1 = 81.104
            self.accuracy5 = 94.707

        elif self.depth == 34:
            self.basic_block = 2
            self.ft_epochs = 300
            self.re_epochs = 300
            self.temp_epoch = 60
            self.filter_nums = [64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 10]
            #self.filter_nums = [4, 10, 10, 20, 20, 10]
            self.gene_a = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]
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
            
        elif self.depth == 50:
            self.basic_block = 3
            self.ft_epochs = 1
            self.re_epochs = 300
            self.temp_epoch = 60
            self.filter_nums = [64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048, 1000]
            self.gene_a = [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            self.gene_length_a = len(self.gene_a)
            self.gene_length = sum(self.filter_nums)
            self.filter_length = sum(self.filter_nums)
            #self.sub_pop_size_b = 2
            self.sub_pop_size_b = 10
            self.pop_size_a = 100
            self.gen = 200
            self.prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            #self.prob = [0, 1, 2, 3, 4]
            self.filter_length = sum(self.filter_nums)
            #self.filter_length = 40
            self.accuracy_top1 = 76.272
            self.accuracy_top5 = 92.994
            
            
        
        elif self.depth == 101:
            self.basic_block = 3
            self.ft_epochs = 300
            self.re_epochs = 300
            self.temp_epoch = 60
            self.filter_nums = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 1000]
            a = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 1000]
            self.gene_a = [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512]
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
            self.accuracy = 94.480
        