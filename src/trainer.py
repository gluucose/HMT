"""
This file aims to
define an abstract trainer class for each model.

"""
import glob
from pathlib import Path
from abc import ABC

import torch
import matplotlib
from loguru import logger


class Trainer(ABC):
    """
    Functions:
        run
        train
        test
        load_data
        load_model
        train_one_epoch
        test_one_epoch
        save_model
    """

    def __init__(self, cfg, cfg_json):
        super().__init__()
        self.cfg = cfg
        
        """device"""
        self.device = torch.device(f'cuda:{self.cfg.device}' if torch.cuda.is_available() else 'cpu')
        
        """path"""
        self.log_name = self.cfg.log_name
        self.dataset_path = f'{self.cfg.data}/{self.cfg.dataset}'
        self.result_path = f'{self.cfg.results}/{self.log_name}'
        self.ckp_path = self.result_path + '/ckp'
        self.surv_curve_path = self.result_path + '/surv_curve'
        self.roc_path = self.result_path + '/roc'
        self.tsne_path = self.result_path + '/tsne'
        self.attn_path = self.result_path + '/attn'
        Path(self.result_path).mkdir(exist_ok=True, parents=True)
        Path(self.ckp_path).mkdir(exist_ok=True)
        Path(self.surv_curve_path).mkdir(exist_ok=True)
        Path(self.roc_path).mkdir(exist_ok=True)
        Path(self.tsne_path).mkdir(exist_ok=True)
        Path(self.attn_path).mkdir(exist_ok=True)
        
        """plt draw in RGB"""
        matplotlib.use('Agg')
        
        """logger"""
        best_ckp_path_reg = f'{self.ckp_path}/fold_0_epoch_*.pt'
        ckp_paths = glob.glob(best_ckp_path_reg)
        logger.add(f"{self.result_path}/{self.log_name}.log")
        if len(ckp_paths) == 0:
            logger.info(cfg_json)

    def run(self):
        if self.cfg.mode == 'train':
            self.train()
            self.test()
        elif self.cfg.mode == 'test':
            self.test()
        else:
            logger.error(f'mode {self.cfg.mode} not support!')

    def train(self):
        pass

    def test(self):
        pass

    def load_data(self):
        pass

    def load_network(self):
        pass

    def train_one_epoch(self):
        pass

    def eval_one_epoch(self):
        pass

    def save_model(self, fold, epoch, ckp, remove=0):
        """
        Args:
            fold:
            epoch:
            ckp:
            remove: 0 for not removing former ckps;
                    1 for removing the latest ckp, keeping the best ckp;
                    2 for removing the best and the latest ckp.
        """
        if remove > 0:
            best_ckp_path_reg = f'{self.ckp_path}/fold_{fold}_epoch_*.pt'
            ckp_paths = sorted(
                glob.glob(best_ckp_path_reg),
                key=lambda name: int(name.split('_')[-1].split('.')[0]),
            )
            if remove == 1 and len(ckp_paths) == 2:
                Path(ckp_paths[-1]).unlink(missing_ok=True)
            elif remove == 2:
                for path in ckp_paths:
                    Path(path).unlink(missing_ok=True)

        torch.save(ckp, f'{self.ckp_path}/fold_{fold}_epoch_{epoch + 1}.pt')
