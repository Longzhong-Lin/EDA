"""
Based on https://github.com/sshaoshuai/MTR/blob/master/mtr/utils/motion_utils.py
"""


import torch
from MTR.mtr.utils.motion_utils import *


def batch_nms(pred_trajs, pred_scores, dist_thresh, num_ret_modes=6, return_mask=False):
    """

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 7)
        pred_scores (batch_size, num_modes)
        dist_thresh (batch_size)
        num_ret_modes (int, optional): Defaults to 6.

    Returns:
        ret_trajs (batch_size, num_ret_modes, num_timestamps, 7)
        ret_scores (batch_size, num_ret_modes)
        ret_idxs (batch_size, num_ret_modes)
    """
    batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
    sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
    sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
    sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, 7)

    dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
    if isinstance(dist_thresh, float):
        point_cover_mask = (dist < dist_thresh)
    else:
        point_cover_mask = (dist < dist_thresh[:, None, None])

    point_val = sorted_pred_scores.clone()  # (batch_size, N)
    point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)
    ret_mask_sorted = torch.ones_like(point_val).bool() # (batch_size, N)

    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
    ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
    bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1) # (batch_size)
        ret_idxs[:, k] = cur_idx

        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
        filter_mask = new_cover_mask.clone()
        filter_mask[bs_idxs, cur_idx] = False
        filter_mask *= (point_val.max(dim=-1, keepdim=True).values > 0)
        ret_mask_sorted[filter_mask] = False

        point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected

        ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
        ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None]

    if return_mask:
        ret_mask = torch.zeros_like(ret_mask_sorted)
        ret_mask_sorted[torch.cumsum(ret_mask_sorted, dim=-1) > num_ret_modes] = False
        ret_mask[bs_idxs, sorted_idxs] = ret_mask_sorted
        return ret_mask
    ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
    return ret_trajs, ret_scores, ret_idxs

