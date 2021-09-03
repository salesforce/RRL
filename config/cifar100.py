config = {}
config['data_path'] = './cifar-100/'
config['dataset'] = 'cifar100'
config['openset'] = False
config['noise_ratio'] = 0.5
config['noise_mode'] = 'sym'
config['noise_file'] = 'noise_file/%s/%s_%.1f.json'%(config['dataset'],config['noise_mode'],config['noise_ratio'])

data_train_opt = {} 
data_train_opt['batch_size'] = 128
data_train_opt['temperature'] = 0.3
data_train_opt['num_class'] = 100
data_train_opt['alpha'] = 8
data_train_opt['w_inst'] = 1
data_train_opt['w_proto'] = 5
data_train_opt['w_recon'] = 1
data_train_opt['low_dim'] = 50 
data_train_opt['warmup_iters'] = 100 
data_train_opt['ramp_epoch'] = 40 

config['data_train_opt'] = data_train_opt
config['max_num_epochs'] = 200

config['test_knn'] = True
config['knn_start_epoch'] = 15
config['knn'] = True
config['n_neighbors'] = 200
config['low_th'] = 0.02
config['high_th'] = 0.9

networks = {}
lr = 0.02
net_optim_params = {'optim_type': 'sgd', 'lr': lr, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov':False}
networks['model'] = {'name': 'presnet', 'pretrained': None, 'opt': {},  'optim_params': net_optim_params}
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':{}} 
criterions['loss_instance'] = {'ctype':'CrossEntropyLoss', 'opt':{}} 

config['criterions'] = criterions
config['algorithm_type'] = 'Model'

config['exp_directory'] = 'experiment/cifar100_sym0.5'
config['checkpoint_dir'] = ''
