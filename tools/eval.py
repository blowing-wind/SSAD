import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import _init_paths
from config import cfg
from config import update_config
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset.TALDataset import TALDataset
from models.SSAD import SSAD
from core.function import train, evaluation
from core.post_process import final_result_process
from core.utils_ab import weight_init


def parse_args():
    parser = argparse.ArgumentParser(description='SSAD temporal action localization')
    parser.add_argument('--cfg', type=str, help='experiment config file', default='../experiments/thumos/SSAD_train.yaml')
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(args.cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    # data loader
    val_dset = TALDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, drop_last=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)

    model = SSAD(cfg)
    model.cuda()

    #evaluate existing model
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    out_df_ab = evaluation(val_loader, model, checkpoint['epoch'], cfg)

    '''
    flag:
    0: jointly consider out_df_ab and out_df_af
    1: only consider out_df_ab
    2: only consider out_df_af    
    '''
    # evaluate both branch
    out_df_list = out_df_ab
    final_result_process(out_df_list, checkpoint['epoch'], cfg, flag=1)
    # # only evaluate anchor-based branch
    # final_result_process(out_df_ab, epoch, cfg, flag=1)
    # # only evaluate anchor-free branch
    # final_result_process(out_df_af, epoch, cfg, flag=2)


if __name__ == '__main__':
    main()


