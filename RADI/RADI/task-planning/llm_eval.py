import os
import sys
import random
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from arguments import get_args
import init_path
from utils_bc import utils

from utils_bc import utils_interactive_eval
from utils_bc.utils import save_model, load_pretrained_model
from utils_bc.utils_llm import get_pretrained_tokenizer
from interactive_evaluation import llm_evaluation
import os
os.environ['CURL_CA_BUNDLE'] = ''


def get_logger(args, log_path):
    if os.path.exists(log_path):
        os.remove(log_path)

    import logging
    a_logger = logging.getLogger()
    a_logger.setLevel(logging.INFO)

    output_file_handler = logging.FileHandler(log_path)
    stdout_handler = logging.StreamHandler(sys.stdout)

    a_logger.addHandler(output_file_handler)
    a_logger.addHandler(stdout_handler)
    logging = a_logger
    return logging

def main():
    args = get_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    args = init_path.get_logger_path(args)
    logging = get_logger(args, args.log_path)

    # 用于收集每个 subset 的 success rate
    success_rate_results = {}

    result_txt_path = 'success_rate_results.txt'

    for subset in ['NovelTasks', 'NovelScenes']:
        args.subset = subset
        args.base_port += 1

        # initialize path
        args = init_path.initialize_path(args)
        args = init_path.load_data_info(args)

        # Testing
        vh_envs = utils_interactive_eval.connect_env(args, logging)
        interactive_eval_success_rate = llm_evaluation(args, vh_envs, logging=logging)

        # 保存单个结果到文件（追加模式）
        with open(result_txt_path, 'a') as f:
            f.write(f"Dataset: {subset} | Success Rate: {interactive_eval_success_rate:.4f}\n")
        print(f"Saved success rate for {subset} to {result_txt_path}")



# def main():
#     '''This following code will set the CURL_CA_BUNDLE environment variable to an empty string in the Python os module'''

#     args = get_args()
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True
#     args = init_path.get_logger_path(args)
#     logging = get_logger(args, args.log_path)
    
#     # iterate 4 subset
#     for subset in ['InDistributation','NovelScenes','NovelTasks','LongTasks']:
#     #for subset in ['LongTasks']:
#     #for subset in ['NovelTasks','LongTasks']:
#         args.subset=subset
#         args.base_port+=1
#         ## initial path
#         args = init_path.initialize_path(args)
#         args = init_path.load_data_info(args)
        
#         ## Testing
#         vh_envs = utils_interactive_eval.connect_env(args, logging)
#         interactive_eval_success_rate = llm_evaluation(args, vh_envs, logging=logging)
    
if __name__ == "__main__":
    main()
