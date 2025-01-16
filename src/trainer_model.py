"""
This file aims to
train the model for predicting survival.

"""
import glob
from pathlib import Path

import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from tensorboardX import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index

from trainer import Trainer
from models import Model
from utils import (
    Loader,
    setup_seed,
    calculate_time,
    get_brier_score,
    plot_survival_curves,
    plot_roc_for_intervals,
    plot_tsne,
    plot_attn_map,
    loss_seg,
    loss_nll,
    loss_l2,
)


class TrainerModel(Trainer):
    """
    Functions:
        run
        train
        test
        load_data
        load_network
        train_one_epoch
        test_one_epoch
        save_model
    """

    def __init__(self, cfg, cfg_json):
        super().__init__(cfg, cfg_json)
        self.epoch_starts = []
        for _ in range(self.cfg.fold_num):
            self.epoch_starts.append(0)

    def run(self):
        super().run()

    def train(self):
        setup_seed(self.cfg.seed)
        
        samples = pd.read_csv(Path(self.dataset_path) / self.cfg.tabular).iloc[:, 0].tolist()
        events = pd.read_csv(Path(self.dataset_path) / self.cfg.tabular).iloc[:, -2].tolist()
        
        if len(self.cfg.intervals) == 0:
            times = pd.read_csv(Path(self.dataset_path) / self.cfg.tabular).iloc[:, -1].tolist()
            self.cfg.intervals = np.linspace(0, max(times), num=self.cfg.interval_num + 1, dtype=int)
        
        kf = StratifiedKFold(n_splits=self.cfg.fold_num, shuffle=True, random_state=self.cfg.seed)
        
        k = -1
        
        for train_index, eval_index in kf.split(samples, events):
            """log"""
            k += 1
            
            best_ckp_path_reg = f'{self.ckp_path}/fold_{k}_epoch_*.pt'
            ckp_paths = sorted(
                glob.glob(best_ckp_path_reg),
                key=lambda name: int(name.split('_')[-1].split('.')[0]),
            )
            if len(ckp_paths) > 0:
                self.cfg.trained_model[k] = ckp_paths[-1]
                if int(ckp_paths[-1].split('_')[-1].split('.')[0]) == self.cfg.epoch_num:
                    continue
            
            logger.info(f'Fold: {k}')
            summary_writer_train = SummaryWriter(f'{self.result_path}/summary/fold{k}/train')
            summary_writer_eval = SummaryWriter(f'{self.result_path}/summary/fold{k}/eval')
            
            """data"""
            samples_train = [samples[i] for i in train_index]
            samples_eval = [samples[i] for i in eval_index]
            loader_train = self.load_data(root=self.dataset_path, cfg=self.cfg, samples=samples_train, mode='train')
            loader_eval = self.load_data(root=self.dataset_path, cfg=self.cfg, samples=samples_eval, mode='eval')
            logger.info(f'train:val={len(samples_train)}:{len(samples_eval)}')
            
            """network"""
            model, criterion, optimizer, lr_scheduler = self.load_network(fold_k=k)
            best_ci = self.cfg.best_ci[k]

            """train & eval"""
            for epoch in range(self.epoch_starts[k], self.cfg.epoch_num):
                model, optimizer, lr_scheduler = self.train_one_epoch(
                    epoch=epoch,
                    summary_writer=summary_writer_train,
                    data_loader=loader_train,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler
                )
                ci, bs, dice, p_value, aucs, label_time, label_event, scores, surv_preds, feats = self.eval_one_epoch(
                    fold=k,
                    epoch=epoch,
                    summary_writer=summary_writer_eval,
                    data_loader=loader_eval,
                    model=model,
                    mode='eval'
                )
                if epoch >= self.cfg.epoch_start_save - 1:
                    if ci > best_ci:
                        best_ci = ci
                        self.save_model(fold=k, epoch=epoch, ckp=model.state_dict(), remove=2)
                    else:
                        self.save_model(fold=k, epoch=epoch, ckp=model.state_dict(), remove=1)            
            summary_writer_train.close()
            summary_writer_eval.close()
            
    def test(self):
        for fold in range(self.cfg.fold_num):
            best_ckp_path_reg = f'{self.ckp_path}/fold_{fold}_epoch_*.pt'
            ckp_paths = sorted(
                glob.glob(best_ckp_path_reg),
                key=lambda name: int(name.split('_')[-1].split('.')[0]),
            )
            self.cfg.trained_model[fold] = ckp_paths[0]
            
        setup_seed(self.cfg.seed)
        
        samples = pd.read_csv(Path(self.dataset_path) / self.cfg.tabular).iloc[:, 0].tolist()
        events = pd.read_csv(Path(self.dataset_path) / self.cfg.tabular).iloc[:, -2].tolist()
        
        if len(self.cfg.intervals) == 0:
            times = pd.read_csv(Path(self.dataset_path) / self.cfg.tabular).iloc[:, -1].tolist()
            self.cfg.intervals = np.linspace(0, max(times), num=self.cfg.interval_num + 1, dtype=int)

        kf = StratifiedKFold(n_splits=self.cfg.fold_num, shuffle=True, random_state=self.cfg.seed)
        
        k = -1
        ci_of_all_folds = []
        bs_of_all_folds = []
        dice_of_all_folds = []
        p_value_of_all_folds = []
        aucs_of_all_folds = [[], [], []]
        label_time_total = np.empty(0)
        label_event_total = np.empty(0)
        scores_total = np.empty(0)
        surv_preds_total = np.empty([0, 10])
        feats_total = np.empty([0, (15*4 if self.cfg.is_seg else 8*16) * self.cfg.modal_num + (self.cfg.t_dim if not self.cfg.is_tatm else 0)])
        
        summary_writer = SummaryWriter(f'{self.result_path}/summary')
        
        for _, eval_index in kf.split(samples, events):
            """log"""
            k += 1
            logger.info(f'Fold: {k}')
            
            """data"""
            samples_eval = [samples[i] for i in eval_index]
            loader_eval = self.load_data(root=self.dataset_path, cfg=self.cfg, samples=samples_eval, mode='eval')
            
            """network"""
            model, _, _, _ = self.load_network(fold_k=k)
            
            """train & eval"""
            epoch = self.epoch_starts[k] - 1
            ci, bs, dice, p_value, aucs, label_time, label_event, scores, surv_preds, feats = self.eval_one_epoch(
                fold=k,
                epoch=epoch,
                summary_writer=summary_writer,
                data_loader=loader_eval,
                model=model,
                mode='test'
            )
            
            ci_of_all_folds.append(ci)
            bs_of_all_folds.append(bs)
            dice_of_all_folds.append(dice)
            p_value_of_all_folds.append(p_value)
            for i in range(len(aucs)):
                aucs_of_all_folds[i].append(aucs[i])
            label_time_total = np.concatenate((label_time_total, label_time))
            label_event_total = np.concatenate((label_event_total, label_event))
            scores_total = np.concatenate((scores_total, scores))
            surv_preds_total = np.concatenate((surv_preds_total, surv_preds))
            feats_total = np.concatenate((feats_total, feats))
                      
        ci_mean = np.mean(ci_of_all_folds)
        bs_mean = np.mean(bs_of_all_folds)
        dice_mean = np.mean(dice_of_all_folds)
        p_value_mean = np.mean(p_value_of_all_folds)
        auc_1_mean = np.mean(aucs_of_all_folds[0])
        auc_3_mean = np.mean(aucs_of_all_folds[1])
        auc_5_mean = np.mean(aucs_of_all_folds[2])
        
        ci_std = np.std(ci_of_all_folds)
        bs_std = np.std(bs_of_all_folds)
        dice_std = np.std(dice_of_all_folds)
        p_value_std = np.std(p_value_of_all_folds)
        auc_1_std = np.std(aucs_of_all_folds[0])
        auc_3_std = np.std(aucs_of_all_folds[1])
        auc_5_std = np.std(aucs_of_all_folds[2])
        
        summary_writer.add_scalar('ci/mean', ci_mean)
        summary_writer.add_scalar('bs/mean', bs_mean)
        summary_writer.add_scalar('dice/mean', dice_mean)
        summary_writer.add_scalar('p value/mean', p_value_mean)
        summary_writer.add_scalar('auc 1/mean', auc_1_mean)
        summary_writer.add_scalar('auc 3/mean', auc_3_mean)
        summary_writer.add_scalar('auc 5/mean', auc_5_mean)
        
        summary_writer.add_scalar('ci/std', ci_std)
        summary_writer.add_scalar('bs/std', bs_std)
        summary_writer.add_scalar('dice/std', dice_std)
        summary_writer.add_scalar('p value/std', p_value_std)
        summary_writer.add_scalar('auc 1/std', auc_1_std)
        summary_writer.add_scalar('auc 3/std', auc_3_std)
        summary_writer.add_scalar('auc 5/std', auc_5_std)
        
        figure_surv_curve_total, p_value_total = plot_survival_curves(label_time_total, label_event_total, -scores_total)
        figure_roc_total, aucs_total = plot_roc_for_intervals(surv_preds_total, label_time_total, label_event_total, self.cfg.intervals, self.cfg.time_spots)
        figure_tsne_total = plot_tsne(feats_total, label_time_total, -scores_total, self.cfg.dataset)
        
        summary_writer.add_scalar(f'p value/total', p_value_total)
        summary_writer.add_scalar(f'auc_1/total', aucs_total[0])
        summary_writer.add_scalar(f'auc_3/total', aucs_total[1])
        summary_writer.add_scalar(f'auc_5/total', aucs_total[2])
        summary_writer.add_figure(f'surv curve/total', figure_surv_curve_total)
        summary_writer.add_figure(f'roc/total', figure_roc_total)
        summary_writer.add_figure(f'tsne/total', figure_tsne_total)
        figure_surv_curve_total.savefig(f'{self.surv_curve_path}/surv_curv.svg', dpi=600)
        figure_roc_total.savefig(f'{self.roc_path}/roc.svg', dpi=600)
        figure_tsne_total.savefig(f'{self.tsne_path}/tsne.svg', dpi=600)
        
        logger.info(f'''ci={ci_mean:.4f}±{ci_std:.4f}, bs={bs_mean:.4f}±{bs_std:.4f}, dice={dice_mean:.4f}±{dice_std:.4f},
                    p_value={p_value_mean:.4f}±{p_value_std:.4f},
                    auc_1={auc_1_mean:.4f}±{auc_1_std:.4f}, auc_3={auc_3_mean:.4f}±{auc_3_std:.4f}, auc_5={auc_5_mean:.4f}±{auc_5_std:.4f}
                    p_value_total={p_value_total:.4f},
                    auc_1_total={aucs_total[0]:.4f}, auc_3_total={aucs_total[1]:.4f}, auc_5_total={aucs_total[2]:.4f}''')
                
        summary_writer.close()
        
    def load_data(self, root, cfg, samples, mode='train'):
        return Loader(root, cfg, samples, mode)()
        
    def load_network(self, fold_k):
        """model"""
        model = Model(channel_num=self.cfg.channel_num,
                      modal_num=self.cfg.modal_num,
                      is_cross=self.cfg.is_cross,
                      is_acmix=self.cfg.is_acmix,
                      is_cnn=self.cfg.is_cnn,
                      is_att=self.cfg.is_att,
                      is_tabular=self.cfg.is_tabular,
                      t_dim=self.cfg.t_dim,
                      bottleneck_factor=self.cfg.bottleneck_factor,
                      is_seg=self.cfg.is_seg,
                      is_ag=self.cfg.is_ag,
                      is_tatm=self.cfg.is_tatm,
                      is_ham=self.cfg.is_ham).to(self.device)
        
        ckp = None
        ckp_paths = self.cfg.trained_model
        if fold_k < len(ckp_paths):
            ckp_path = ckp_paths[fold_k]
            if ckp_path != '' and Path(ckp_path).exists():
                ckp = torch.load(ckp_paths[fold_k], map_location=self.device)
                self.epoch_starts[fold_k] = int(ckp_paths[fold_k].split('_')[-1].split('.')[0])

        if ckp is not None:
            pretrain_dict = ckp
            model_dict = {}
            state_dict = model.state_dict()
            for key, value in pretrain_dict.items():
                if key in state_dict:
                    model_dict[key] = value
            state_dict.update(model_dict)
            model.load_state_dict(state_dict)

        """criterion"""
        criterion = [loss_seg, loss_nll, loss_l2]

        """optimizer"""
        optimizer = torch.optim.Adam(model.parameters(), lr=1)

        """lr_scheduler"""
        def rule(epoch):
            if epoch < 20:
                lamb = 1e-4
            elif epoch < 40:
                lamb = 5e-5
            elif epoch < 60:
                lamb = 1e-5
            else:
                lamb = 1e-6
            return lamb

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)

        return model, criterion, optimizer, lr_scheduler

    def train_one_epoch(
        self,
        epoch,
        summary_writer,
        data_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
    ):
        model.train()
        
        train_losses = []
        train_total_loss = []
        
        for pt, ct, seg, tabular, time, event, surv_array, idx in tqdm(
            data_loader, desc=f'Epoch {epoch + 1} training', colour=self.cfg.color.train, ncols=100
        ):
            
            pt = pt.to(self.device)
            ct = ct.to(self.device)
            seg = seg.to(self.device)
            tabular = tabular.to(self.device)
            time = time.to(self.device)
            event = event.to(self.device)
            surv_array = surv_array.to(self.device)
            
            """predict"""
            if self.cfg.modal_num == 2:
                seg_pred, surv_pred, reg_weight, _, _ = model(pt, ct, tabular)
            elif self.cfg.is_pt and not self.cfg.is_ct:
                seg_pred, surv_pred, reg_weight, _, _ = model(pt, None, tabular)
            elif self.cfg.is_ct and not self.cfg.is_pt:
                seg_pred, surv_pred, reg_weight, _, _ = model(ct, None, tabular)
            else:
                raise ValueError('modal_num should be 2 or is_pt or is_ct should be True')
            
            """loss"""
            loss = 0
            loss_list = []
            
            if seg_pred is not None:
                loss_seg = criterion[0](seg, seg_pred)
                loss_list.append(loss_seg.item())
                loss += loss_seg
            else:
                loss_list.append(0)
            loss_surv = criterion[1](surv_array, surv_pred)
            loss_list.append(loss_surv.item())
            loss += loss_surv
            loss_reg = criterion[2](reg_weight)
            loss_list.append(loss_reg.item())
            loss += loss_reg
            
            train_losses.append(loss_list)
            train_total_loss.append(loss.item())
            
            """back propagate and optimize"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_losses = np.mean(train_losses, axis=0)
        train_total_loss = np.mean(train_total_loss)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr_scheduler.step()
        
        """log"""
        logger.info(f'''
            epoch {epoch + 1}:
            Total loss={train_total_loss:.4f} 
            (seg={train_losses[0]:.4f}, 
            surv={train_losses[1]:.4f}, 
            reg={train_losses[2]:.4f}), lr={lr:.4f}
        ''')
        summary_writer.add_scalar('lr', lr, epoch + 1)
        summary_writer.add_scalar('loss/total', train_total_loss, epoch + 1)
        summary_writer.add_scalar('loss/seg', train_losses[0], epoch + 1)
        summary_writer.add_scalar('loss/surv', train_losses[1], epoch + 1)
        summary_writer.add_scalar('loss/reg', train_losses[2], epoch + 1)
        
        return model, optimizer, lr_scheduler

    def eval_one_epoch(
        self,
        fold,
        epoch,
        summary_writer,
        data_loader,
        model,
        mode
    ):
        model.eval()
        
        if mode == 'eval':
            color = self.cfg.color.eval
        else:
            color = self.cfg.color.test
        
        time_list = []
        event_list = []
        seg_inter, seg_union = 0, 0
        time_pred_list = []
        brier_score_list = []
        surv_pred_list = []
        feats_list = []
        
        for pt, ct, seg, tabular, time, event, surv_array, idx in tqdm(
            data_loader, desc=f'Epoch {epoch + 1} evaluating', colour=color, ncols=100
        ):
            pt = pt.to(self.device)
            ct = ct.to(self.device)
            tabular = tabular.to(self.device)
            seg = np.array(seg)
            time = np.array(time)
            event = np.array(event)
            
            time_list.append(time)
            event_list.append(event)
            
            """pred"""
            with torch.no_grad():
                if self.cfg.modal_num == 2:
                    seg_pred, surv_pred, _, feats, masks = model(pt, ct, tabular)
                elif self.cfg.is_pt and not self.cfg.is_ct:
                    seg_pred, surv_pred, _, feats, masks = model(pt, None, tabular)
                elif self.cfg.is_ct and not self.cfg.is_pt:
                    seg_pred, surv_pred, _, feats, masks = model(ct, None, tabular)
                else:
                    raise ValueError('modal_num should be 2 or is_pt or is_ct should be True')
            
            """cumulate metrics"""
            surv_pred_list.append(surv_pred.cpu().numpy())
            if seg_pred is not None:
                seg_pred = seg_pred.detach().cpu().numpy().squeeze()
                _, seg_pred = cv2.threshold(seg_pred, 0.5, 1, cv2.THRESH_BINARY)
                seg_inter += np.sum(seg_pred * seg)
                seg_union += np.sum(seg_pred + seg)
            
            time_pred_list.append(calculate_time(surv_pred.cpu(), self.cfg.intervals))
            
            brier_score = get_brier_score(surv_pred.cpu(), surv_array, time, event, self.cfg.intervals)
            if brier_score > 0:
                brier_score_list.append(brier_score)
            
            feats_list.append(feats.cpu().numpy())
            
            if self.cfg.is_plot_attn:
                plot_attn_map(pt.squeeze().cpu(), ct.squeeze().cpu(), seg.squeeze(), seg_pred, masks, self.attn_path, fold, idx.item())
        
        """calculate metrics"""
        dice = (2 * seg_inter / seg_union) if seg_inter > 0 else 0
        
        label_time = np.array(time_list).squeeze()
        label_event = np.array(event_list).squeeze()
        scores = np.array(time_pred_list).squeeze()
        surv_preds = np.array(surv_pred_list).squeeze()
        ci = concordance_index(label_time, scores, label_event)
        bs = np.array(brier_score_list).mean()
        feats = np.concatenate(feats_list)
        
        p_value, aucs = None, None
        """log"""
        if mode == 'eval':
            logger.info(f'epoch {epoch + 1}: ci={ci:.4f}, bs={bs:.4f}, dice={dice:.4f}')
            summary_writer.add_scalar(f'ci/eval', ci, epoch + 1)
            summary_writer.add_scalar(f'bs/eval', bs, epoch + 1)
            summary_writer.add_scalar(f'dice/eval', dice, epoch + 1)
            
        elif mode == 'test':
            figure_surv_curve, p_value = plot_survival_curves(label_time, label_event, -scores)
            figure_roc, aucs = plot_roc_for_intervals(surv_preds, label_time, label_event, self.cfg.intervals, self.cfg.time_spots)
            figure_tsne = plot_tsne(feats, label_time, -scores, self.cfg.dataset)
            logger.info(f'epoch {epoch + 1}: ci={ci:.4f}, bs={bs:.4f}, dice={dice:.4f}, p_value={p_value:.4e}, aucs=[{aucs[0]:.4f}, {aucs[1]:.4f}, {aucs[2]:.4f}]')
            summary_writer.add_scalar(f'ci/test', ci, fold + 1)
            summary_writer.add_scalar(f'bs/test', bs, fold + 1)
            summary_writer.add_scalar(f'dice/test', dice, fold + 1)
            summary_writer.add_scalar(f'p value', p_value, fold + 1)
            summary_writer.add_scalar(f'auc_1', aucs[0], fold + 1)
            summary_writer.add_scalar(f'auc_3', aucs[1], fold + 1)
            summary_writer.add_scalar(f'auc_5', aucs[2], fold + 1)
            summary_writer.add_figure(f'surv curve', figure_surv_curve, fold + 1)
            summary_writer.add_figure(f'roc', figure_roc, fold + 1)
            summary_writer.add_figure(f'tsne', figure_tsne, fold + 1)
            figure_surv_curve.savefig(f'{self.surv_curve_path}/fold_{fold}_epoch_{epoch + 1}_surv_curv.svg', dpi=600)
            figure_roc.savefig(f'{self.roc_path}/fold_{fold}_epoch_{epoch + 1}_roc.svg', dpi=600)
            figure_tsne.savefig(f'{self.tsne_path}/fold_{fold}_epoch_{epoch + 1}_tsne.svg', dpi=600)
            
        return ci, bs, dice, p_value, aucs, label_time, label_event, scores, surv_preds, feats

    def save_model(self, fold, epoch, ckp, remove=0):
        super().save_model(fold, epoch, ckp, remove)
