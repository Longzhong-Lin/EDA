"""
Based on https://github.com/sshaoshuai/MTR/blob/master/tools/train_utils/train_utils.py
"""


import glob
import os
import re

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_

from MTR.tools.train_utils.train_utils import *


def train_model(model, optimizer, train_loader, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, ckpt_save_dir, train_sampler=None,
                ckpt_save_interval=1, max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, tb_log=None,
                scheduler=None, test_loader=None, logger=None, eval_output_dir=None, cfg=None, dist_train=False,
                logger_iter_interval=50, ckpt_save_time_interval=300):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            torch.cuda.empty_cache()
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            if scheduler is None:
                learning_rate_decay(cur_epoch, optimizer, optim_cfg)

            # train one epoch
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                scheduler=scheduler, cur_epoch=cur_epoch, total_epochs=total_epochs,
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if (trained_epoch % ckpt_save_interval == 0 or trained_epoch in [1, 2, 4] or trained_epoch > total_epochs - 10) and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                try:
                    ckpt_list.sort(key=os.path.getmtime)
                except:
                    ckpt_list.sort(key=lambda ckpt: re.findall(r'\d+', ckpt))

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        if os.path.isfile(ckpt_list[cur_file_idx]):
                            os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            # eval the model
            if test_loader is not None and (trained_epoch % ckpt_save_interval == 0 or trained_epoch in [1, 2, 4] or trained_epoch > total_epochs - 10):
                from eval_utils.eval_utils import eval_one_epoch

                pure_model = model
                torch.cuda.empty_cache()
                tb_dict = eval_one_epoch(
                    cfg, pure_model, test_loader, epoch_id=trained_epoch, logger=logger, dist_test=dist_train,
                    result_dir=eval_output_dir, save_to_file=False, logger_iter_interval=max(logger_iter_interval // 5, 1)
                )
                if cfg.LOCAL_RANK == 0:
                    for key, val in tb_dict.items():
                        tb_log.add_scalar('eval/' + key, val, trained_epoch)

                    if 'mAP' in tb_dict:
                        best_record_file = eval_output_dir / ('best_eval_record.txt')

                        try:
                            with open(best_record_file, 'r') as f:
                                best_src_data = f.readlines()

                            best_performance = best_src_data[-1].strip().split(' ')[-1]  # best_epoch_xx MissRate 0.xx
                            best_performance = float(best_performance)
                        except:
                            with open(best_record_file, 'a') as f:
                                pass
                            best_performance = -1


                        with open(best_record_file, 'a') as f:
                            print(f'epoch_{trained_epoch} mAP {tb_dict["mAP"]}', file=f)

                        if best_performance == -1 or tb_dict['mAP'] > float(best_performance):
                            ckpt_name = ckpt_save_dir / 'best_model'
                            save_checkpoint(
                                checkpoint_state(model, epoch=cur_epoch, it=accumulated_iter), filename=ckpt_name,
                            )
                            logger.info(f'Save best model to {ckpt_name}')

                            with open(best_record_file, 'a') as f:
                                print(f'best_epoch_{trained_epoch} mAP {tb_dict["mAP"]}', file=f)
                        else:
                            with open(best_record_file, 'a') as f:
                                print(f'{best_src_data[-1].strip()}', file=f)
                    else:
                        raise NotImplementedError


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import eda
        version = 'eda+' + eda.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}

