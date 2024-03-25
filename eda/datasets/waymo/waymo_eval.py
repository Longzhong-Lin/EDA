"""
Based on https://github.com/sshaoshuai/MTR/blob/master/mtr/datasets/waymo/waymo_eval.py
"""


import numpy as np
import tensorflow as tf
import os

from google.protobuf import text_format

all_gpus = tf.config.experimental.list_physical_devices('GPU')
if all_gpus:
    try:
        for cur_gpu in all_gpus:
            tf.config.experimental.set_memory_growth(cur_gpu, True)
    except RuntimeError as e:
        print(e)

from MTR.mtr.datasets.waymo.waymo_eval import *


def transform_preds_to_waymo_format(pred_dicts, top_k_for_eval=-1, eval_second=8, rank_first=True):
    print(f'Total number for evaluation (intput): {len(pred_dicts)}')
    temp_pred_dicts = []
    for k in range(len(pred_dicts)):
        if isinstance(pred_dicts[k], list):
            temp_pred_dicts.extend(pred_dicts[k])
        else:
            temp_pred_dicts.append(pred_dicts[k])
    pred_dicts = temp_pred_dicts
    print(f'Total number for evaluation (after processed): {len(pred_dicts)}')

    scene2preds = {}
    num_max_objs_per_scene = 0
    for k in range(len(pred_dicts)):
        cur_scenario_id = pred_dicts[k]['scenario_id']
        if  cur_scenario_id not in scene2preds:
            scene2preds[cur_scenario_id] = []
        scene2preds[cur_scenario_id].append(pred_dicts[k])
        num_max_objs_per_scene = max(num_max_objs_per_scene, len(scene2preds[cur_scenario_id]))
    num_scenario = len(scene2preds)
    topK, num_future_frames, _ = pred_dicts[0]['pred_trajs'].shape

    if top_k_for_eval != -1:
        topK = min(top_k_for_eval, topK)

    if num_future_frames in [30, 50, 80]:
        sampled_interval = 5
    assert num_future_frames % sampled_interval == 0, f'num_future_frames={num_future_frames}'
    num_frame_to_eval = num_future_frames // sampled_interval

    if eval_second == 3:
        num_frames_in_total = 41
        num_frame_to_eval = 6
    elif eval_second == 5:
        num_frames_in_total = 61
        num_frame_to_eval = 10
    else:
        num_frames_in_total = 91
        num_frame_to_eval = 16

    batch_pred_trajs = np.zeros((num_scenario, num_max_objs_per_scene, topK, 1, num_frame_to_eval, 2))
    batch_pred_scores = np.zeros((num_scenario, num_max_objs_per_scene, topK))
    gt_trajs = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total, 7))
    gt_is_valid = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total), dtype=np.int)
    pred_gt_idxs = np.zeros((num_scenario, num_max_objs_per_scene, 1))
    pred_gt_idx_valid_mask = np.zeros((num_scenario, num_max_objs_per_scene, 1), dtype=np.int)
    object_type = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.object)
    object_id = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.int)
    scenario_id = np.zeros((num_scenario), dtype=np.object)

    object_type_cnt_dict = {}
    for key in object_type_to_id.keys():
        object_type_cnt_dict[key] = 0

    for scene_idx, val in enumerate(scene2preds.items()):
        cur_scenario_id, preds_per_scene = val
        scenario_id[scene_idx] = cur_scenario_id
        for obj_idx, cur_pred in enumerate(preds_per_scene):
            sort_idxs = cur_pred['pred_scores'].argsort()[::-1]
            cur_pred['pred_scores'] = cur_pred['pred_scores'][sort_idxs]
            cur_pred['pred_trajs'] = cur_pred['pred_trajs'][sort_idxs]
            
            if rank_first:
                cur_pred['pred_scores'] = cur_pred['pred_scores'] + cur_pred['pred_scores'].argsort().argsort() # rank-first normalization
            else:
                cur_pred['pred_scores'] = cur_pred['pred_scores'] / cur_pred['pred_scores'].sum()
            
            batch_pred_trajs[scene_idx, obj_idx] = cur_pred['pred_trajs'][:topK, np.newaxis, 4::sampled_interval, :][:, :, :num_frame_to_eval, :]
            batch_pred_scores[scene_idx, obj_idx] = cur_pred['pred_scores'][:topK]
            gt_trajs[scene_idx, obj_idx] = cur_pred['gt_trajs'][:num_frames_in_total, [0, 1, 3, 4, 6, 7, 8]]  # (num_timestamps_in_total, 10), [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            gt_is_valid[scene_idx, obj_idx] = cur_pred['gt_trajs'][:num_frames_in_total, -1]
            pred_gt_idxs[scene_idx, obj_idx, 0] = obj_idx
            pred_gt_idx_valid_mask[scene_idx, obj_idx, 0] = 1
            object_type[scene_idx, obj_idx] = object_type_to_id[cur_pred['object_type']]
            object_id[scene_idx, obj_idx] = cur_pred['object_id']

            object_type_cnt_dict[cur_pred['object_type']] += 1

    gt_infos = {
        'scenario_id': scenario_id.tolist(),
        'object_id': object_id.tolist(),
        'object_type': object_type.tolist(),
        'gt_is_valid': gt_is_valid,
        'gt_trajectory': gt_trajs,
        'pred_gt_indices': pred_gt_idxs,
        'pred_gt_indices_mask': pred_gt_idx_valid_mask
    }
    return batch_pred_scores, batch_pred_trajs, gt_infos, object_type_cnt_dict


def main():
    import pickle
    import argparse
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--top_k', type=int, default=-1, help='')
    parser.add_argument('--eval_second', type=int, default=8, help='')
    parser.add_argument('--num_modes_for_eval', type=int, default=6, help='')

    args = parser.parse_args()
    print(args)

    assert args.eval_second in [3, 5, 8]
    pred_infos = pickle.load(open(args.pred_infos, 'rb'))

    result_format_str = ''
    print('Start to evaluate the waymo format results...')

    metric_results, result_format_str = waymo_evaluation(
        pred_dicts=pred_infos, top_k=args.top_k, eval_second=args.eval_second,
        num_modes_for_eval=args.num_modes_for_eval,
    )

    print(metric_results)
    metric_result_str = '\n'
    for key in metric_results:
        metric_results[key] = metric_results[key]
        metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
    print(metric_result_str)
    print(result_format_str)


if __name__ == '__main__':
    main()

