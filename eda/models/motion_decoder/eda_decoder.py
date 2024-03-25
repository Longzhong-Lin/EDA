"""
Based on https://github.com/sshaoshuai/MTR/blob/master/mtr/models/motion_decoder/mtr_decoder.py
"""


import torch
import torch.nn.functional as F

from MTR.mtr.utils import loss_utils
from MTR.mtr.models.motion_decoder.mtr_decoder import MTRDecoder

from eda.utils import motion_utils


class EDADecoder(MTRDecoder):
    def __init__(self, in_channels, config):
        super().__init__(in_channels, config)
        self.num_refines = self.model_cfg.get('NUM_REFINES', 1)
        self.num_inter_layers = self.num_decoder_layers // self.num_refines
    
    def apply_transformer_decoder(self, center_objects_feature, center_objects_type, obj_feature, obj_mask, obj_pos, map_feature, map_mask, map_pos):
        intention_query, intention_points = self.get_motion_query(center_objects_type)
        query_content = torch.zeros_like(intention_query)
        self.forward_ret_dict['intention_points'] = intention_points.permute(1, 0, 2)  # (num_center_objects, num_query, 2)

        num_center_objects = query_content.shape[1]
        num_query = query_content.shape[0]

        center_objects_feature = center_objects_feature[None, :, :].repeat(num_query, 1, 1)  # (num_query, num_center_objects, C)

        base_map_idxs = None
        pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]  # (num_center_objects, num_query, 1, 2)
        dynamic_query_center = intention_points

        pred_list = []
        for layer_idx in range(self.num_decoder_layers):
            # query object feature
            obj_query_feature = self.apply_cross_attention(
                kv_feature=obj_feature, kv_mask=obj_mask, kv_pos=obj_pos,
                query_content=query_content, query_embed=intention_query,
                attention_layer=self.obj_decoder_layers[layer_idx],
                dynamic_query_center=dynamic_query_center,
                layer_idx=layer_idx
            ) 

            # query map feature
            collected_idxs, base_map_idxs = self.apply_dynamic_map_collection(
                map_pos=map_pos, map_mask=map_mask,
                pred_waypoints=pred_waypoints,
                base_region_offset=self.model_cfg.CENTER_OFFSET_OF_MAP,
                num_waypoint_polylines=self.model_cfg.NUM_WAYPOINT_MAP_POLYLINES,
                num_base_polylines=self.model_cfg.NUM_BASE_MAP_POLYLINES,
                base_map_idxs=base_map_idxs,
                num_query=num_query
            )

            map_query_feature = self.apply_cross_attention(
                kv_feature=map_feature, kv_mask=map_mask, kv_pos=map_pos,
                query_content=query_content, query_embed=intention_query,
                attention_layer=self.map_decoder_layers[layer_idx],
                layer_idx=layer_idx,
                dynamic_query_center=dynamic_query_center,
                use_local_attn=True,
                query_index_pair=collected_idxs,
                query_content_pre_mlp=self.map_query_content_mlps[layer_idx],
                query_embed_pre_mlp=self.map_query_embed_mlps
            ) 

            query_feature = torch.cat([center_objects_feature, obj_query_feature, map_query_feature], dim=-1)
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
            ).view(num_query, num_center_objects, -1) 

            # motion prediction
            query_content_t = query_content.permute(1, 0, 2).contiguous().view(num_center_objects * num_query, -1)
            pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)
            if self.motion_vel_heads is not None:
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 5)
                pred_vel = self.motion_vel_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 2)
                pred_trajs = torch.cat((pred_trajs, pred_vel), dim=-1)
            else:
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 7)

            pred_list.append([pred_scores, pred_trajs])

            # update
            # BUG pred_waypoints & dynamic_query_center should be detached, but this makes no difference
            pred_waypoints = pred_trajs.detach().clone()[:, :, :, 0:2]
            dynamic_query_center = pred_trajs.detach().clone()[:, :, -1, 0:2].contiguous().permute(1, 0, 2)  # (num_query, num_center_objects, 2)

        if self.use_place_holder:
            raise NotImplementedError

        assert len(pred_list) == self.num_decoder_layers
        return pred_list

    def get_decoder_loss(self, tb_pre_tag=''):
        center_gt_trajs = self.forward_ret_dict['center_gt_trajs'].cuda()
        center_gt_trajs_mask = self.forward_ret_dict['center_gt_trajs_mask'].cuda()
        center_gt_final_valid_idx = self.forward_ret_dict['center_gt_final_valid_idx'].long()
        assert center_gt_trajs.shape[-1] == 4

        pred_list = self.forward_ret_dict['pred_list']
        intention_points = self.forward_ret_dict['intention_points']  # (num_center_objects, num_query, 2)

        num_center_objects = center_gt_trajs.shape[0]
        center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx, 0:2]  # (num_center_objects, 2)

        if not self.use_place_holder:
            dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1)  # (num_center_objects, num_query)
            center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)
        else:
            raise NotImplementedError

        tb_dict = {}
        disp_dict = {}
        total_loss = 0
        for layer_idx in range(self.num_decoder_layers):
            if self.use_place_holder:
                raise NotImplementedError

            pred_scores, pred_trajs = pred_list[layer_idx]
            assert pred_trajs.shape[-1] == 7
            pred_trajs_gmm, pred_vel = pred_trajs[:, :, :, 0:5], pred_trajs[:, :, :, 5:7]

            # Evolving Anchors
            positive_layer_idx = (layer_idx//self.num_inter_layers) * self.num_inter_layers - 1
            if positive_layer_idx < 0:
                anchor_trajs = intention_points.unsqueeze(-2)
                dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1)  # (num_center_objects, num_query)
            else:
                anchor_scores, anchor_trajs = pred_list[positive_layer_idx]
                dist = ((center_gt_trajs[:, None, :, 0:2] - anchor_trajs[..., 0:2]).norm(dim=-1) * \
                    center_gt_trajs_mask[:, None]).sum(dim=-1)  # (num_center_objects, num_query)
            
            # Distinct Anchors
            select_mask = torch.ones_like(pred_scores).bool()
            if self.model_cfg.get('DISTINCT_ANCHORS', False):
                if self.model_cfg.DISTINCT_NMS_THRESH < 0:
                    top_traj = pred_trajs[torch.arange(num_center_objects), pred_scores.argsort(dim=-1)[:, -1]][..., :2]
                    top_traj_length = torch.norm(torch.diff(top_traj, dim=1), dim=-1).sum(dim=-1)
                    upper_dist = 3.5
                    lower_dist = 2.5
                    upper_length = 50
                    lower_length = 10
                    scalar = 1.5
                    dist_thresh = torch.minimum(
                        torch.tensor(upper_dist, device=pred_trajs.device),
                        torch.maximum(
                            torch.tensor(lower_dist, device=pred_trajs.device),
                            lower_dist+scalar*(top_traj_length-lower_length)/(upper_length-lower_length)
                        )
                    )
                else:
                    dist_thresh = self.model_cfg.DISTINCT_NMS_THRESH
                select_mask = motion_utils.batch_nms(
                    anchor_trajs, pred_scores.sigmoid(),
                    dist_thresh=dist_thresh,
                    num_ret_modes=anchor_trajs.shape[1],
                    return_mask=True
                )
            dist = dist.masked_fill(~select_mask, 1e10)

            # Evolving & Distinct Anchors
            center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)

            loss_reg_gmm, center_gt_positive_idx = loss_utils.nll_loss_gmm_direct(
                pred_scores=pred_scores, pred_trajs=pred_trajs_gmm,
                gt_trajs=center_gt_trajs[:, :, 0:2], gt_valid_mask=center_gt_trajs_mask,
                pre_nearest_mode_idxs=center_gt_positive_idx,
                timestamp_loss_weight=None, use_square_gmm=False,
            )

            pred_vel = pred_vel[torch.arange(num_center_objects), center_gt_positive_idx]
            loss_reg_vel = F.l1_loss(pred_vel, center_gt_trajs[:, :, 2:4], reduction='none')
            loss_reg_vel = (loss_reg_vel * center_gt_trajs_mask[:, :, None]).sum(dim=-1).sum(dim=-1)

            bce_target = torch.zeros_like(pred_scores)
            bce_target[torch.arange(num_center_objects), center_gt_positive_idx] = 1.0
            loss_cls = F.binary_cross_entropy_with_logits(input=pred_scores, target=bce_target, reduction='none')
            loss_cls = (loss_cls * select_mask).sum(dim=-1)

            # total loss
            weight_cls = self.model_cfg.LOSS_WEIGHTS.get('cls', 1.0)
            weight_reg = self.model_cfg.LOSS_WEIGHTS.get('reg', 1.0)
            weight_vel = self.model_cfg.LOSS_WEIGHTS.get('vel', 0.2)

            layer_loss = loss_reg_gmm * weight_reg + loss_reg_vel * weight_vel + loss_cls.sum(dim=-1) * weight_cls
            layer_loss = layer_loss.mean()
            total_loss += layer_loss
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}'] = layer_loss.item()
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_gmm'] = loss_reg_gmm.mean().item() * weight_reg
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_vel'] = loss_reg_vel.mean().item() * weight_vel
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_cls'] = loss_cls.mean().item() * weight_cls

            if layer_idx + 1 == self.num_decoder_layers:
                layer_tb_dict_ade = motion_utils.get_ade_of_each_category(
                    pred_trajs=pred_trajs_gmm[:, :, :, 0:2],
                    gt_trajs=center_gt_trajs[:, :, 0:2], gt_trajs_mask=center_gt_trajs_mask,
                    object_types=self.forward_ret_dict['center_objects_type'],
                    valid_type_list=self.object_type,
                    post_tag=f'_layer_{layer_idx}',
                    pre_tag=tb_pre_tag
                )
                tb_dict.update(layer_tb_dict_ade)
                disp_dict.update(layer_tb_dict_ade)

        total_loss = total_loss / self.num_decoder_layers
        return total_loss, tb_dict, disp_dict

    def generate_final_prediction(self, batch_dict):
        pred_list = self.forward_ret_dict['pred_list']
        pred_scores, pred_trajs = pred_list[-1]
        pred_scores = torch.sigmoid(pred_scores)  # (num_center_objects, num_query)
        num_center_objects, num_query, num_future_timestamps, num_feat = pred_trajs.shape

        if self.num_motion_modes != num_query:
            assert num_query > self.num_motion_modes

            nms_threshold = self.model_cfg.NMS_DIST_THRESH
            if nms_threshold >= 0:
                dist_thresh = nms_threshold
            else:
                top_traj = pred_trajs[torch.arange(num_center_objects), pred_scores.argsort(dim=-1)[:, -1]][..., :2]
                top_traj_length = torch.norm(torch.diff(top_traj, dim=1), dim=-1).sum(dim=-1)
                upper_dist = 3.5
                lower_dist = 2.5
                upper_length = 50
                lower_length = 10
                scalar = 1.5
                dist_thresh = torch.minimum(
                    torch.tensor(upper_dist, device=pred_trajs.device),
                    torch.maximum(
                        torch.tensor(lower_dist, device=pred_trajs.device),
                        lower_dist+scalar*(top_traj_length-lower_length)/(upper_length-lower_length)
                    )
                )
            
            pred_trajs_final, pred_scores_final, selected_idxs = motion_utils.batch_nms(
                pred_trajs=pred_trajs, pred_scores=pred_scores,
                dist_thresh=dist_thresh,
                num_ret_modes=self.num_motion_modes
            )
        else:
            pred_trajs_final = pred_trajs
            pred_scores_final = pred_scores

        batch_dict['pred_scores'] = pred_scores_final
        batch_dict['pred_trajs'] = pred_trajs_final

        return batch_dict

    def forward(self, batch_dict):
        input_dict = batch_dict['input_dict']
        obj_feature, obj_mask, obj_pos = batch_dict['obj_feature'], batch_dict['obj_mask'], batch_dict['obj_pos']
        map_feature, map_mask, map_pos = batch_dict['map_feature'], batch_dict['map_mask'], batch_dict['map_pos']
        center_objects_feature = batch_dict['center_objects_feature']
        num_center_objects, num_objects, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]

        self.forward_ret_dict['center_objects_type'] = input_dict['center_objects_type']
        self.forward_ret_dict['center_gt_trajs'] = input_dict['center_gt_trajs']
        self.forward_ret_dict['center_gt_trajs_mask'] = input_dict['center_gt_trajs_mask']
        self.forward_ret_dict['center_gt_final_valid_idx'] = input_dict['center_gt_final_valid_idx']
        self.forward_ret_dict['obj_trajs_future_state'] = input_dict['obj_trajs_future_state']
        self.forward_ret_dict['obj_trajs_future_mask'] = input_dict['obj_trajs_future_mask']

        # input projection
        center_objects_feature = self.in_proj_center_obj(center_objects_feature)
        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, obj_feature_valid.shape[-1])
        obj_feature[obj_mask] = obj_feature_valid

        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        map_feature = map_feature.new_zeros(num_center_objects, num_polylines, map_feature_valid.shape[-1])
        map_feature[map_mask] = map_feature_valid

        # dense future prediction
        obj_feature, pred_dense_future_trajs = self.apply_dense_future_prediction(
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos
        )
        # decoder layers
        pred_list = self.apply_transformer_decoder(
            center_objects_feature=center_objects_feature,
            center_objects_type=input_dict['center_objects_type'],
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos,
            map_feature=map_feature, map_mask=map_mask, map_pos=map_pos
        )

        self.forward_ret_dict['pred_list'] = pred_list
        batch_dict = self.generate_final_prediction(batch_dict=batch_dict)

        return batch_dict
