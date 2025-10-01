import json
import os

import torch.backends.cudnn as cudnn

from methods.adv import adv
from methods.original import original
from utils.save_utils import construct_logger, check_repeat_running
from methods.dp import dp

def initialize_settings(path):
    class Args:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)

        def add_attribute(self, k, v):
            setattr(self, k, v)

        def record_config(self, logger):
            for attr in self.__dict__:
                logger.info(f'{attr}: {getattr(self, attr)}')

    with open(path, 'r') as f:
        configuration = json.load(f)
    return Args(configuration)


def federal_rec(args):
    # if check_repeat_running(args):
    #     args.logger.info('This setting exists')
    # else:
        if args.method == 'original':
            original(args)
        elif args.method == 'adv':
            adv(args)
        elif args.method == 'dp':
            dp(args)
 

if __name__ == '__main__':
    config_path = 'configs/exp_config.json'
    args = initialize_settings(config_path)
    cudnn.benchmark = True

    save_path = f'./exp_results/{args.model}/{args.dataset}/{args.method}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.add_attribute("save_path", save_path)

    logger = construct_logger(args)
    args.record_config(logger)
    args.add_attribute("logger", logger)
    federal_rec(args)

