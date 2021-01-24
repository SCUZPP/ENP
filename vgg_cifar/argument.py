class Args:
    def __init__(self):
    
        self.depth = 19
        self.lr = 0.1
        self.ft_lr = 0.0001
        self.retrain_lr = 0.1
        self.epochs = 160
        self.batch_size = 64
        self.test_batch_size = 500
        self.val_batch_size = 500
        self.momentum = 0.9
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.weight_decay = 1e-4
        self.retrain = False
        self.sample_time = 4
        
        #最后三层是全连接层
        if self.depth == 16:
        
            self.ft_epochs = 40
            self.filter_nums = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, 10]
            self.fc_nums_dict = { '1': 4096, '4': 4096, '6': 10}
            self.conv_layer = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 1, 4, 6]
            self.fc_layer = [1, 4, 6]
            self.conv_name = ['0', '3', '7', '10', '14', '17', '20', '24', '27', '30', '34', '37', '40']
            self.name = ['0', '3', '7', '10', '14', '17', '20', '24', '27', '30', '34', '37', '40', '1', '4', '6']
            self.fc_name = ['1', '4', '6']
            self.conv_name_dict = {'0': [], '3': [], '7': [], '10': [], '14': [], '17': [], '20': [], '24': [], '27': [], '30': [], '34': [], '37': [], '40': []}
            self.fc_name_dict = { '1': [], '4': [], '6': []}
            self.gene_length = 12426
            self.filter_length = 12426
            self.accuracy = 93.7
        
        else:
            self.ft_epochs = 60
            self.re_epochs = 60
            self.filter_nums = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 4096, 4096, 10]
            #self.filter_nums = [10, 20, 10]
            self.conv_layer = [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49, 1, 4, 6]
            self.fc_nums_dict = { '1': 4096, '4': 4096, '6': 10}
            self.fc_layer = [1, 4, 6]
            self.name = ['0', '3', '7', '10', '14', '17', '20', '23', '27', '30', '33', '36', '40', '43', '46', '49', '1', '4', '6']
            self.fc_name = ['1', '4', '6']
            #self.gene_length_a = 19
            self.gene_length_a = len(self.filter_nums)
            #self.sub_pop_size_b = 2
            self.sub_pop_size_b = 5
            self.pop_size_a = 50
            self.gen = 100
            self.prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            #self.prob = [0, 1, 2, 3, 4]
            self.filter_length = sum(self.filter_nums)
            #self.filter_length = 40
            self.accuracy = 93.84

        