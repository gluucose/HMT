import os
import random
from pathlib import Path

import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE


plt.rcParams['font.family'] = 'Times New Roman'
figure_size = (10, 8)
font_size = 16


class JsonObject(object):
    def __init__(self, d):
        self.__dict__.update(d)


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def get_surv_array(time, event, intervals):
    """
    Transforms censored survival data into vector format that can be used in Keras.
    Args:
        time: Array of failure/censoring times.
        event: Array of censoring indicator. 1 if failed, 0 if censored.
        intervals: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Return:
        surv_array: Dimensions with (number of samples, number of time intervals*2)
    """
    
    breaks = np.array(intervals)
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    
    surv_array = np.zeros((n_intervals * 2))
    
    if event == 1:
        surv_array[0 : n_intervals] = 1.0 * (time >= breaks[1:]) 
        if time < breaks[-1]:
            surv_array[n_intervals + np.where(time < breaks[1:])[0][0]] = 1
    else:
        surv_array[0 : n_intervals] = 1.0 * (time >= breaks_midpoint)
    
    return surv_array


def calculate_time(surv_pred, intervals):
    breaks = np.array(intervals)
    surv_pred = np.array(surv_pred)
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    surv_time = 0
    for i in range(n_intervals):
        cumulative_prob = np.prod(surv_pred[0:i+1])
        surv_time += cumulative_prob * timegap[i]
    return surv_time


def get_brier_score(surv_pred, surv_array, time, event, intervals):
    breaks = np.array(intervals)
    surv_pred = np.array(surv_pred.squeeze(0))
    surv_array = surv_array.squeeze(0)
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    brier_score = 0
    cnt = 0
    for i in range(n_intervals):
        if event.item() == 0 and time.item() < breaks_midpoint[i]:
            break
        cumulative_prob = np.prod(surv_pred[0:i+1])
        brier_score += (cumulative_prob - surv_array[i]) ** 2
        cnt += 1
    return (brier_score / cnt) if cnt != 0 else 0


def coxph(clinical_data, duration, event):
    df = pd.DataFrame(clinical_data, columns=[f"feature_{i}" for i in range(clinical_data.shape[1])])
    df['duration'] = duration
    df['event'] = event
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col='duration', event_col='event')
    risk_scores = cph.predict_partial_hazard(df)
    return risk_scores


def plot_survival_curves(times, events, risks):
    risk_mean = risks.mean()
    idx_of_low_risk = np.argwhere(risks < risk_mean).squeeze()
    idx_of_high_risk = np.argwhere(risks >= risk_mean).squeeze()
    
    figure = plt.figure(figsize=figure_size)
    kmf = KaplanMeierFitter()
    kmf.fit(
        times[idx_of_low_risk],
        event_observed=events[idx_of_low_risk],
        label='Low Risk',
    )
    ax = kmf.plot_survival_function()
    kmf.fit(
        times[idx_of_high_risk],
        event_observed=events[idx_of_high_risk],
        label='High Risk',
    )
    kmf.plot_survival_function(ax=ax)
    ax.legend(loc="lower left", fontsize=font_size)
    # if ax.get_legend():
    #     ax.get_legend().remove()
    
    p_result = logrank_test(times[idx_of_low_risk],
                            times[idx_of_high_risk],
                            events[idx_of_low_risk],
                            events[idx_of_high_risk])
    p_value = p_result.p_value
    ax.set_title(f'p value={p_value:.4e}', fontsize=font_size)
    # plt.text(.04, .12, f'p_value={p_value:.4e}', fontsize=p_font_size, ha='left', va='top', transform=ax.transAxes)
    
    plt.xlabel('', fontsize=font_size)
    plt.xlabel('Timeline', fontsize=font_size)
    plt.ylabel('Survival Probability', fontsize=font_size)
    plt.tick_params(axis='x', labelsize=font_size)
    plt.tick_params(axis='y', labelsize=font_size)
    
    figure.canvas.draw()
    plt.close('all')
    
    return figure, p_value


def plot_roc_for_intervals(surv_pred, durations, events, intervals, time_spots):
    tprs = []
    fprs = []
    aucs = []
    
    figure = plt.figure(figsize=figure_size)
    for t, i in zip(time_spots, [1, 3, 5]):
        pred_event_before_time_point = 1 - np.prod(surv_pred[:, :t], axis=1)
        actual_event_before_time_point = (durations <= intervals[t]) & (events == 1)
        
        fpr, tpr, _ = roc_curve(actual_event_before_time_point, pred_event_before_time_point)
        roc_auc = auc(fpr, tpr)
        
        tprs.append(tpr)
        fprs.append(fpr)
        aucs.append(roc_auc)
        
        plt.plot(fpr, tpr, lw=2, label=f'{i} year{"s" if i > 1 else ""} (AUC={roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('FP', fontsize=font_size)
    plt.ylabel('TP', fontsize=font_size)
    plt.tick_params(axis='x', labelsize=font_size)
    plt.tick_params(axis='y', labelsize=font_size)
    
    # plt.title('ROC Curves for Different Time Points')
    
    plt.legend(loc="lower right", fontsize=font_size)
    
    return figure, aucs


def plot_tsne(feats, times, risks, title):
    risk_mean = risks.mean()
    idx_of_high_risk = np.argwhere(risks >= risk_mean)
    labels = np.zeros_like(times)
    labels[idx_of_high_risk] = 1

    targets = range(2)
    colors = ['blue', 'red']
    target_names = ["Low Risk", "High Risk"]
    
    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    X_tsne = tsne.fit_transform(feats)
    
    figure = plt.figure(figsize=figure_size)
    for target, color, label in zip(targets, colors, target_names):
        indices = [i for i, x in enumerate(labels) if x == target]
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=color, s=50, label=label, alpha=0.5)
    
    # plt.title(title)
    plt.legend(loc="upper right", fontsize=font_size)
    
    plt.tick_params(axis='x', labelsize=font_size)
    plt.tick_params(axis='y', labelsize=font_size)
    
    plt.close()
    
    return figure


def plot_seg(pt, ct, seg, seg_pred, out_path, fold, idx):
    pos = 0
    for p in range(seg.shape[0]):
        if seg[p].any():
            pos = p + 5
            break
    pt = np.asarray(pt)[pos]
    ct = np.asarray(ct)[pos]
    seg = np.asarray(seg)[pos]
    seg_pred = np.asarray(seg_pred)[pos]
    
    pt = 255 * pt
    ct = 255 * ct
    seg = np.uint8(255 * seg)
    seg_pred = np.uint8(255 * seg_pred)
    
    cv2.imwrite(f'{out_path}/{fold}_{idx}_pt.jpg', pt)
    cv2.imwrite(f'{out_path}/{fold}_{idx}_ct.jpg', ct)
    cv2.imwrite(f'{out_path}/{fold}_{idx}_seg.jpg', seg)
    cv2.imwrite(f'{out_path}/{fold}_{idx}_seg_pred.jpg', seg_pred)


def plot_attn_map(pt, ct, seg, seg_pred, masks, out_path, fold, idx):
    pos = 0
    for p in range(seg.shape[0]):
        if seg[p].any():
            pos = p + 5
            break
    pt = np.asarray(pt)[pos]
    ct = np.asarray(ct)[pos]
    seg = np.asarray(seg)[pos]
    seg_pred = np.asarray(seg_pred)[pos]
    for i in range(len(masks)):
        masks[i] = np.asarray(masks[i].squeeze().cpu()).transpose(2, 1, 0)
        masks[i] = cv2.resize(masks[i], (ct.shape[0], ct.shape[1])).transpose(2, 1, 0)[pos // (2 ** (len(masks) - 1 - i))]
        masks[i] = masks[i] / np.max(masks[i])
        masks[i] = masks[i][np.newaxis, ...]
    mask = np.mean(np.concatenate(masks, axis=0), axis=0)
    mask = mask / np.max(mask)
    
    pt = 255 * pt
    ct = 255 * ct
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    
    # cam_pt = heatmap + pt
    # cam_pt = cam_pt / np.max(cam_pt)
    # cam_pt = np.uint8(255 * cam_pt)
    # cam_ct = heatmap + ct
    # cam_ct = cam_ct / np.max(cam_ct)
    # cam_ct = np.uint8(255 * cam_ct)
    
    seg = np.uint8(255 * seg)
    seg_pred = np.uint8(255 * seg_pred)
    
    Path(f'{out_path}/{fold}').mkdir(exist_ok=True)
    cv2.imwrite(f'{out_path}/fold_{fold}/{idx}_pt.jpg', pt)
    cv2.imwrite(f'{out_path}/fold_{fold}/{idx}_ct.jpg', ct)
    cv2.imwrite(f'{out_path}/fold_{fold}/{idx}_seg.jpg', seg)
    cv2.imwrite(f'{out_path}/fold_{fold}/{idx}_seg_pred.jpg', seg_pred)
    cv2.imwrite(f'{out_path}/fold_{fold}/{idx}_heatmap.jpg', heatmap)


def loss_cox(pred, T, E):
    batch_len = len(pred)
    R_matrix_train = np.zeros(
        [batch_len, batch_len], dtype=int)
    for i in range(batch_len):
        for j in range(batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]
    
    train_R = torch.FloatTensor(R_matrix_train).to(pred.device)
    train_y_status = torch.from_numpy(E).to(pred.device)
    theta = pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_nn = -torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_y_status)

    return loss_nn


def loss_nll(y_true, y_pred, n_intervals=10):
    """
    Args:
        y_true(Tensor): First half: 1 if individual survived that interval, 0 if not.
                        Second half: 1 for time interval before which failure has occured, 0 for other intervals.
        y_pred(Tensor): Predicted survival probability (1-hazard probability) for each time interval.
    """
    cens_uncens = torch.clamp(1.0 + y_true[:, 0 : n_intervals] * (y_pred - 1.0), min=1e-5)
    uncens = torch.clamp(1.0 - y_true[:, n_intervals : 2 * n_intervals] * y_pred, min=1e-5)
    loss = -torch.mean(torch.log(cens_uncens) + torch.log(uncens))
    return loss


def loss_dice(y_true, y_pred):
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    return -dice


def loss_focal(y_true, y_pred, alpha=0.25, gamma=2, epsilon = 1e-5):
    y_pred_clamp = torch.clamp(y_pred, min=epsilon, max=1-epsilon)
    logits = torch.log(y_pred_clamp / (1 - y_pred_clamp))
    weight_a = alpha * torch.pow((1 - y_pred_clamp), gamma) * y_true
    weight_b = (1 - alpha) * torch.pow(y_pred_clamp, gamma) * (1 - y_true)
    loss = torch.log1p(torch.exp(-logits)) * (weight_a + weight_b) + logits * weight_b
    return torch.mean(loss)


def loss_seg(y_true, y_pred):
    return loss_dice(y_true, y_pred) + loss_focal(y_true, y_pred)


def loss_l2(weights, alpha=0.1):
    loss = 0
    for weight in weights:
        loss += torch.square(weight).sum()
    return alpha * loss
