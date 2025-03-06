import os
import time
import sys

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from modules.model import *
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *

from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth import megadepth_warper
from torch.utils.data import Dataset, DataLoader

class Trainer():

    def __init__(self, megadepth_root_path, 
                       ckpt_save_path, 
                       model_name = 'xfeat_megadepth',
                       batch_size = 10, n_steps = 160_000, lr= 3e-4, gamma_steplr=0.5, 
                       training_res = (800, 608), device_num="0", dry_run = False,
                       save_ckpt_every = 500):

        self.dev = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev)

        #Setup optimizer 
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()) , lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)

        ##################### MEGADEPTH INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_megadepth'):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                            npz_path = path) for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")] )

            self.data_loader = DataLoader(data, 
                                          batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size),
                                          shuffle=True)
            self.data_iter = iter(self.data_loader)

        else:
            self.data_iter = None
        ##################### MEGADEPTH INIT END #######################

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name


    def train(self):

        self.net.train()  # 将模型设置为训练模式

        difficulty = 0.10  # 设置训练难度

        p1s, p2s, H1, H2 = None, None, None, None  # 初始化变量
        d = None  # 初始化数据变量
        
        if self.data_iter is not None:
            d = next(self.data_iter)  # 获取下一个数据批次

        with tqdm.tqdm(total=self.steps) as pbar:  # 使用 tqdm 显示进度条
            for i in range(self.steps):  # 训练循环
                if not self.dry_run:  # 如果不是 dry run
                    if self.data_iter is not None:
                        try:
                            # 获取下一个 MegaDepth 数据批次
                            d = next(self.data_iter)
                        except StopIteration:
                            print("End of DATASET!")
                            # 如果数据迭代器结束，重新创建一个新的迭代器
                            self.data_iter = iter(self.data_loader)
                            d = next(self.data_iter)

                if d is not None:
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)  # 将数据移动到设备上（GPU 或 CPU）
                
                    p1, p2 = d['image0'], d['image1']  # 获取图像对
                    positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)  # 获取粗匹配点

                # 将 MegaDepth 数据转换为灰度图像
                with torch.inference_mode():
                    if d is not None:
                        p1 = p1.mean(1, keepdim=True)
                        p2 = p2.mean(1, keepdim=True)

                    positives_c = positives_md_coarse

                # 检查批次是否损坏（匹配点太少）
                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                # 前向传播
                feats1, kpts1, hmap1 = self.net(p1)
                feats2, kpts2, hmap2 = self.net(p2)

                loss_items = []

                for b in range(len(positives_c)):
                    # 获取正匹配点
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                    # 获取对应索引的特征
                    m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                    m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)

                    # 获取对应索引的热图
                    h1 = hmap1[b, 0, pts1[:,1].long(), pts1[:,0].long()]
                    h2 = hmap2[b, 0, pts2[:,1].long()]
                    coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                    # 计算损失
                    loss_ds, conf = dual_softmax_loss(m1, m2)
                    loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)

                    loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                    loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2)*2.0
                    acc_pos = (acc_pos1 + acc_pos2)/2

                    loss_kp =  keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(loss_coords.unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    loss_items.append(loss_kp_pos.unsqueeze(0))

                    if b == 0:
                        acc_coarse_0 = check_accuracy(m1, m2)

                acc_coarse = check_accuracy(m1, m2)

                nb_coarse = len(m1)
                loss = torch.cat(loss_items, -1).mean()
                loss_coarse = loss_ds.item()
                loss_coord = loss_coords.item()
                loss_coord = loss_coords.item()
                loss_kp_pos = loss_kp_pos.item()
                loss_l1 = loss_kp.item()

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i+1}.pth')

                pbar.set_description( 'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} acc_f: {:.3f} loss_c: {:.3f} loss_f: {:.3f} loss_kp: {:.3f} #matches_c: {:d} loss_kp_pos: {:.3f} acc_kp_pos: {:.3f}'.format(
                                                                        loss.item(), acc_coarse_0, acc_coarse, acc_coords, loss_coarse, loss_coord, loss_l1, nb_coarse, loss_kp_pos, acc_pos) )
                pbar.update(1)

                # 记录指标
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
                self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
                self.writer.add_scalar('Accuracy/fine_mdepth', acc_coords, i)
                self.writer.add_scalar('Accuracy/kp_position', acc_pos, i)
                self.writer.add_scalar('Loss/coarse', loss_coarse, i)
                self.writer.add_scalar('Loss/fine', loss_coord, i)
                self.writer.add_scalar('Loss/reliability', loss_l1, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_kp_pos, i)
                self.writer.add_scalar('Count/matches_coarse', nb_coarse, i)




if __name__ == '__main__':
    
    trainer = Trainer(
        megadepth_root_path="/home/wenhuanyao/Dataset/MegaDepth", 
        ckpt_save_path="/home/wenhuanyao/accelerated_features/checkpoints/megadepth",
        model_name="xfeat_megadepth",
        batch_size=16,
        n_steps=160_000,
        lr=3e-4,
        gamma_steplr=0.5,
        training_res=(800, 608),
        device_num="2",
        dry_run=False,
        save_ckpt_every=2000
    )

    trainer.train()