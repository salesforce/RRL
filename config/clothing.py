config = {}
config['data_path'] = './Clothing1M/data'
config['dataset'] = 'clothing'

data_train_opt = {} 
data_train_opt['batch_size'] = 64
data_train_opt['temperature'] = 0.3
data_train_opt['num_class'] = 14
data_train_opt['alpha'] = 0.5
data_train_opt['w_inst'] = 1
data_train_opt['w_proto'] = 1
data_train_opt['w_recon'] = 1 
data_train_opt['low_dim'] = 32
data_train_opt['warmup_iters'] = 50 
data_train_opt['ramp_epoch'] = 0

config['data_train_opt'] = data_train_opt
config['max_num_epochs'] = 50

config['test_knn'] = True
config['knn_start_epoch'] = 1
config['knn'] = True
config['n_neighbors'] = 200
config['low_th'] = 0.4
config['high_th'] = 0.9

networks = {}
lr = 0.01
net_optim_params = {'optim_type': 'sgd', 'lr': lr, 'momentum':0.9, 'weight_decay': 1e-4, 'nesterov':False}
networks['model'] = {'name': 'resnet50', 'pretrained': True, 'opt': {}, 'optim_params': net_optim_params}
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':{}} 
criterions['loss_instance'] = {'ctype':'CrossEntropyLoss', 'opt':{}} 

config['criterions'] = criterions
config['algorithm_type'] = 'Model'

config['exp_directory'] = 'experiments/clothing' 
config['checkpoint_dir'] = ''
