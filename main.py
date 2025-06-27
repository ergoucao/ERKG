import os
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.StatusTSFNet import StatusTSFNet
# from src.models import  AttrModel

from src.network import StatusPredictor


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # init device
    # if torch.cuda.is_available():
    #     config.DEVICE = torch.device("cuda")
    #     torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    # else:
    #     config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    # cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    all_metric=[]
    for it in range(3):
        # build the model and initialize
        model = StatusTSFNet(config)
        model2 = StatusTSFNet(config)
        model.load(mode)

        # model training
        if config.MODE == 1:
            config.print()
            print('\nstart training...\n')
            metric=model.train()
            all_metric.append(metric)
        # model test
        elif config.MODE == 2:
            print('\nstart testing...\n')
            # model.TSF_model.load_state_dict(torch.load("./checkpoints/status_TSF310:42:27_predL_168_horizon_1.pth")) # new
            # model2.TSF_model.load_state_dict(torch.load("./checkpoints/TSF_model211:18:28_predL_168_horizon_1.pth"))# old
            model.TSF_model.load_state_dict(torch.load("./checkpoints/status_TSF3_predL_1_horizon_1_topk_47_uk_dale1.pth")) # new
            model2.TSF_model.load_state_dict(torch.load("./checkpoints/TSF_model2_predL_1_horizon_1_topk_47_uk_dale1.pth")) # old
            model.test(dataset=model.test_dataset)
            model2.test(dataset=model.test_dataset)
        # eval mode
        else:
            print('\nstart eval...\n')
            model.eval()
            np.mean()
    all_metric=np.array(all_metric)
    print_average_res(all_metric)

def print_average_res(all_metric):
    mean=all_metric.mean(0)
    std=all_metric.std(0)
    print(
        "avg: mae:{:.6f},mse:{:.6f},rmse:{:.6f},mape:{:.6f},mspe:{:.6f},rse:{:.6f},corr:{:.6f}\n".format(
            mean[0], mean[1], mean[2], mean[3], 
            mean[4], mean[5], mean[6], 
        )
    )
    print(
        "std: mae:{:.6f},mse:{:.6f},rmse:{:.6f},mape:{:.6f},mspe:{:.6f},rse:{:.6f},corr:{:.6f}\n".format(
            std[0], std[1], std[2], std[3], 
            std[4], std[5], std[6], 
        )
    )
    
    


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints',
                        help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, default=1, choices=[1, 2, 3, 4],
                        help='1: status model, 2: TSF model, 3: status-TFS model, 4: joint model')
    parser.add_argument('--tsf_model', type=str, default='etsformer',
                        help='etsformer or stid')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pred_len', type=int, default=72)
    parser.add_argument('--status_file', type=str)
    parser.add_argument('--data', type=str, default="ampds2", help='ampds2、 uk_dale_building1 -> uk_dale_building5 or hdfb')
    parser.add_argument('--data_dim', type=int)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--topk', type=int, default=12)
    parser.add_argument('--seq_len', type=int, default=336)
    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input data directory or an input image')
        parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')
    print("config_path******"+config_path)
    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)
    config.DEVICE = args.device
    config.TSF_MODEL = args.tsf_model
    config.pred_len = args.pred_len
    config.status_file = args.status_file
    config.data = args.data
    config.updateNetParameter(args.data_dim)
    config.alpha = args.alpha
    config.topk = args.topk
    config.seq_len=args.seq_len
    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3
        # config.INPUT_SIZE = 0

        if args.input is not None:
            config.TEST_FLIST = args.input

        if args.output is not None:
            config.RESULTS = args.output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config


if __name__ == "__main__":
    main(1)
    # config = load_config(1)
    # attr_model = AttrModel(config).to(config.DEVICE)
    # input = torch.randn((32, 4, 256, 256))
    # attr_generator = AttrGenerator()
    # output=attr_generator(input)
    # attr=torch.zeros_like(output)
    # attr_model.bce_loss(output,attr)
    # bce_loss=torch.nn.BCELoss(reduction=None)
    # print(attr_generator(input).shape[0])  # 32,38
