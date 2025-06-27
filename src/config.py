import os
import yaml

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml, Loader=yaml.FullLoader)
            self._dict['PATH'] = os.path.dirname(config_path)
    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        return None

    def updateNetParameter(self, data_parameter: int):
        self._dict['enc_in'] = data_parameter
        self._dict['dec_in'] = data_parameter
        self._dict['c_out'] = data_parameter
        self._dict['data_dim'] = data_parameter

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(str(self._dict))
        print('')
        print('---------------------------------')
        print('')


DEFAULT_CONFIG = {
    'MODE': 1,                      # 1: train, 2: test, 3: eval
    'MODEL': 1,                     # 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
    'SEED': 42,                     # random seed
    'GPU': [1,2],                     # list of gpu ids
    'DEBUG': 0,                     # turns on debugging mode
    'VERBOSE': 0,                   # turns on verbose mode in the output console

    'LR': 0.0001,                   # learning rate
    'STATUS_LR': 0.1,                  # discriminator/generator learning rate ratio
    'BETA1': 0.0,                   # adam optimizer beta1
    'BETA2': 0.9,                   # adam optimizer beta2
    'BATCH_SIZE': 32,                # input batch size for training
    'WINDOWD_SIZE': 168,              # input image size for training 0 for original size
    'MAX_ITERS': 2e6,               # maximum number of iterations to train the model
    'MAX_EPOCH': 40,
    'STATUS_THRESHOLD': 0.5,          # edge detection threshold
    'L1_LOSS_WEIGHT': 1,            # l1 loss weight
    'FM_LOSS_WEIGHT': 10,           # feature-matching loss weight

    'SAVE_INTERVAL': 500,          # how many iterations to wait before saving model (0: never)
    'SAMPLE_INTERVAL': 500,        # how many iterations to wait before sampling (0: never)
    'SAMPLE_SIZE': 12,              # number of images to sample
    'EVAL_INTERVAL': 0,             # how many iterations to wait before model evaluation (0: never)
    'LOG_INTERVAL': 10,             # how many iterations to wait before logging training status (0: never)
}
