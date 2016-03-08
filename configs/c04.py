# For mac local testing
config_01 = {'batch_size': 10,
             'hidden_size': 100,
             'num_layers': 1,
             'rnn_num_layers': 2,
             'init_mean': 0,
             'init_std': 0.1,
             'init_lr': 0.01,
             'anneal_period': 100,
             'anneal_ratio': 0.5,
             'num_epochs': 50,
             'linear_start': False,
             'max_grad_norm': 40,
             'keep_prob': 1.0,
             'fold_path': 'data/s3-100/fold.json',
             'data_dir': 'data/s3-100'}

# initial try
config_02 = {'batch_size': 100,
             'hidden_size': 100,
             'num_layers': 1,
             'rnn_num_layers': 2,
             'init_mean': 0,
             'init_std': 0.1,
             'init_lr': 0.01,
             'anneal_period': 100,
             'anneal_ratio': 0.5,
             'num_epochs': 50,
             'linear_start': False,
             'max_grad_norm': 40,
             'keep_prob': 1.0,
             'fold_path': 'data/s3/fold6.json',
             'data_dir': 'data/s3',
             'mode': 'lca',
             'model_name': 'm04'}

configs = {1: config_01, 2: config_02}
