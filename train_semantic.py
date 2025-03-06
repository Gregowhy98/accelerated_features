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
from cityscapesdataset import CityScapesDataset
from modules.dataset.megadepth import megadepth_warper
from torch.utils.data import Dataset, DataLoader



class Trainer():

    def __init__(self, cityscapes_root_path,
                       ckpt_save_path, 
                       model_name = 'xfeat_megadepth',
                       batch_size = 10, n_steps = 160_000, lr= 1e-4, gamma_steplr=0.5, 
                       training_res = (800, 608), device_num="0", dry_run = False,
                       save_ckpt_every = 500, pretrain_model_path=None, weight_var=0.5):

        self.dev = torch.device ('cuda:'+ device_num if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev)
        
        # Load pretrain model
        if pretrain_model_path is not None:
            self.net.load_state_dict(torch.load(pretrain_model_path, map_location=self.dev))
            print(f"Pretrain model loaded from {pretrain_model_path}")

        # Setup optimizer 
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr, weight_decay=1e-5)  # 增加 weight_decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)
        
        ##################### CITYSCAPES INIT ##########################
        if model_name in ('xfeat_cityscapes'):
            tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(training_res),
                transforms.ToTensor()
            ])
            mydataset = CityScapesDataset(cityscapes_root_path, use='train', transform=tf, device=self.dev)
            self.data_loader = DataLoader(mydataset, batch_size=int(batch_size), shuffle=True)     
            self.data_iter = iter(self.data_loader)
        else:
            self.data_iter = None
        ##################### CITYSCAPES INIT END #######################

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name
        self.weight_var = weight_var


    def train(self):

        self.net.train()  # 将模型设置为训练模式
        
        d = None  # 初始化数据变量
        
        if self.data_iter is not None:
            d = next(self.data_iter)  # 获取下一个数据批次

        with tqdm.tqdm(total=self.steps) as pbar:  # 使用 tqdm 显示进度条
            for i in range(self.steps):  # 训练循环
                if not self.dry_run:  # 如果不是 dry run
                    if self.data_iter is not None:
                        try:
                            # 获取下一个数据批次
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
                            
                    raw_img = d['raw_img']
                    seg_gt = d['seg_gt']
                    
                    xfeat_gt = d['xfeat_gt']
                    hmap_gt = xfeat_gt[2].squeeze(1)
                
                # 数据转换为灰度图像
                with torch.inference_mode():
                    if d is not None:
                        raw_img = raw_img.mean(1, keepdim=True)

                # 调整 seg_gt 的形状，使其与 hmap_gt 一致
                seg_gt = F.interpolate(seg_gt, size=hmap_gt.shape[2:], mode='bilinear', align_corners=False)

                # 调整 seg_gt 的方差
                seg_gt = seg_gt * self.weight_var

                # 将 seg_gt 的值作为权重，通过逐元素相乘赋给 hmap_gt
                weighted_hmap_gt = hmap_gt * seg_gt

                # 前向传播
                _, _, hmap = self.net(raw_img)  # feat, kpts, hmap

                loss_items = []
                
                # 计算 hmap 的损失
                loss_hmap = 0
                for b in range(hmap.size(0)):  # 遍历批次中的每个样本
                    loss_hmap += F.mse_loss(hmap[b], weighted_hmap_gt[b])
                loss_hmap /= hmap.size(0)  # 计算平均损失

                loss_items.append(loss_hmap.unsqueeze(0))
                
                loss = torch.cat(loss_items, -1).mean()
                loss_hmap = loss_hmap.item()

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i+1}.pth')
                
                pbar.set_description(f'Loss: {loss.item():.4f} loss_hmap: {loss_hmap:.4f}')
                pbar.update(1)

                # 记录指标
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Loss/hmap', loss_hmap, i)
                self.writer.flush()


if __name__ == '__main__':
    
    trainer = Trainer(
        cityscapes_root_path="/home/wenhuanyao/Dataset/cityscapes",
        ckpt_save_path="/home/wenhuanyao/accelerated_features/checkpoints/cityscape",
        model_name="xfeat_cityscapes",
        pretrain_model_path="/home/wenhuanyao/accelerated_features/weights/xfeat_semantic.pt",
        batch_size=32,
        n_steps=160_000,
        lr=1e-4,  # 调整学习率
        gamma_steplr=0.5,
        training_res=(800, 608),
        device_num="2",
        dry_run=False,
        save_ckpt_every=1000,
        weight_var=1.5  # 设置权重方差参数
    )

    trainer.train()