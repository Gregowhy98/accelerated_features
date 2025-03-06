"""
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    线特征提取模块训练脚本
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import argparse
import tqdm
import random
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modules.xfeat import XFeat, LineFeatureTransformer
from modules.model import XFeatModel

# 自定义collate函数，处理不同长度的数据
def custom_collate_fn(batch):
    """
    自定义collate函数，处理不同长度的线段数据
    """
    imgs1 = []
    imgs2 = []
    line_data_batch = {
        'start_points1': [],
        'directions1': [],
        'lengths1': [],
        'start_points2': [],
        'directions2': [],
        'lengths2': [],
        'num_lines': []
    }
    
    for img1, img2, line_data in batch:
        imgs1.append(img1)
        imgs2.append(img2)
        
        for key in line_data_batch.keys():
            if key == 'num_lines':
                line_data_batch[key].append(line_data[key])
            else:
                line_data_batch[key].append(line_data[key])
    
    # 堆叠图像
    imgs1 = torch.stack(imgs1, dim=0)
    imgs2 = torch.stack(imgs2, dim=0)
    
    # 不堆叠线段数据，保持为列表
    return imgs1, imgs2, line_data_batch

class SyntheticLineDataset(Dataset):
    """合成线段数据集，用于训练线特征提取器"""
    
    def __init__(self, num_samples=10000, img_size=(480, 640), num_lines=20, 
                 augment=True, noise_level=0.05, homography_noise=0.1):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_lines = num_lines  # 固定每张图像的线段数量
        self.augment = augment
        self.noise_level = noise_level
        self.homography_noise = homography_noise
    
    def __len__(self):
        return self.num_samples
    
    def generate_random_lines(self, num_lines):
        """生成随机线段"""
        h, w = self.img_size
        
        # 防止无限递归
        if num_lines <= 0:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
        
        # 最大尝试次数，避免无限循环
        max_attempts = 5
        
        for attempt in range(max_attempts):
            # 生成线段起点
            start_points = np.random.rand(num_lines, 2)
            start_points[:, 0] *= w
            start_points[:, 1] *= h
            
            # 生成线段方向（单位向量）
            angles = np.random.rand(num_lines) * 2 * np.pi
            directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            
            # 生成线段长度
            lengths = np.random.rand(num_lines) * min(h, w) * 0.3 + min(h, w) * 0.1
            
            # 计算终点
            end_points = start_points + directions * lengths.reshape(-1, 1)
            
            # 确保线段在图像范围内
            mask = ((end_points[:, 0] >= 0) & (end_points[:, 0] < w) & 
                    (end_points[:, 1] >= 0) & (end_points[:, 1] < h))
            
            valid_count = np.sum(mask)
            
            # 如果有足够的有效线段，或者这是最后一次尝试
            if valid_count >= num_lines or attempt == max_attempts - 1:
                # 如果有效线段数量超过所需数量，随机选择所需数量
                if valid_count > num_lines:
                    valid_indices = np.where(mask)[0]
                    selected_indices = np.random.choice(valid_indices, num_lines, replace=False)
                    return start_points[selected_indices], directions[selected_indices], lengths[selected_indices]
                
                # 如果有效线段数量不足，使用所有有效线段并填充剩余部分
                if valid_count < num_lines:
                    # 获取有效线段
                    valid_starts = start_points[mask]
                    valid_dirs = directions[mask]
                    valid_lengths = lengths[mask]
                    
                    # 需要填充的数量
                    padding_count = num_lines - valid_count
                    
                    # 如果有有效线段，通过复制来填充
                    if valid_count > 0:
                        # 随机选择要复制的线段索引
                        padding_indices = np.random.choice(valid_count, padding_count, replace=True)
                        
                        # 填充线段
                        padded_starts = np.vstack([valid_starts, valid_starts[padding_indices]])
                        padded_dirs = np.vstack([valid_dirs, valid_dirs[padding_indices]])
                        padded_lengths = np.hstack([valid_lengths, valid_lengths[padding_indices]])
                        
                        return padded_starts, padded_dirs, padded_lengths
                    else:
                        # 如果没有有效线段，创建简单的默认线段
                        default_starts = np.array([[w/4, h/4], [w/2, h/2], [w*3/4, h*3/4]] * ((num_lines // 3) + 1))[:num_lines]
                        default_dirs = np.array([[1, 0], [0, 1], [1, 1]] * ((num_lines // 3) + 1))[:num_lines]
                        default_dirs = default_dirs / np.linalg.norm(default_dirs, axis=1, keepdims=True)
                        default_lengths = np.ones(num_lines) * min(h, w) * 0.2
                        
                        return default_starts, default_dirs, default_lengths
                
                # 如果有效线段数量正好等于所需数量
                return start_points[mask], directions[mask], lengths[mask]
        
        # 如果所有尝试都失败，返回一些默认线段
        default_starts = np.array([[w/4, h/4], [w/2, h/2], [w*3/4, h*3/4]] * ((num_lines // 3) + 1))[:num_lines]
        default_dirs = np.array([[1, 0], [0, 1], [1, 1]] * ((num_lines // 3) + 1))[:num_lines]
        default_dirs = default_dirs / np.linalg.norm(default_dirs, axis=1, keepdims=True)
        default_lengths = np.ones(num_lines) * min(h, w) * 0.2
        
        return default_starts, default_dirs, default_lengths
    
    def draw_lines(self, img, start_points, directions, lengths, color=None, thickness=2):
        """在图像上绘制线段"""
        for i in range(len(start_points)):
            if color is None:
                # 随机颜色
                c = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            else:
                c = color
                
            start = tuple(map(int, start_points[i]))
            end = tuple(map(int, (start_points[i] + directions[i] * lengths[i])))
            cv2.line(img, start, end, c, thickness)
    
    def apply_homography(self, points, H):
        """应用单应性变换到点集"""
        # 转换为齐次坐标
        n = points.shape[0]
        homogeneous = np.concatenate([points, np.ones((n, 1))], axis=1)
        
        # 应用变换
        transformed = np.dot(H, homogeneous.T).T
        
        # 转回笛卡尔坐标
        return transformed[:, :2] / transformed[:, 2:3]
    
    def generate_random_homography(self):
        """生成随机单应性矩阵"""
        h, w = self.img_size
        
        # 源点：图像四角
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        # 目标点：添加随机扰动的四角
        noise = np.random.uniform(-self.homography_noise * min(h, w), 
                                  self.homography_noise * min(h, w), 
                                  (4, 2))
        dst = src + noise
        
        # 计算单应性矩阵
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        return H
    
    def add_noise(self, img):
        """添加噪声到图像"""
        noise = np.random.normal(0, self.noise_level * 255, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def __getitem__(self, idx):
        h, w = self.img_size
        
        # 创建基础图像（灰色背景）
        img1 = np.ones((h, w, 3), dtype=np.uint8) * 128
        
        # 生成固定数量的随机线段
        start_points, directions, lengths = self.generate_random_lines(self.num_lines)
        
        # 绘制线段
        self.draw_lines(img1, start_points, directions, lengths)
        
        # 添加噪声
        if self.augment:
            img1 = self.add_noise(img1)
        
        # 生成第二张图像（应用单应性变换）
        H = self.generate_random_homography()
        img2 = cv2.warpPerspective(img1, H, (w, h))
        
        # 变换线段起点
        transformed_starts = self.apply_homography(start_points, H)
        
        # 计算变换后的方向向量
        # 为此，我们需要变换线段的终点，然后重新计算方向
        end_points = start_points + directions * lengths.reshape(-1, 1)
        transformed_ends = self.apply_homography(end_points, H)
        
        # 计算新的方向向量
        transformed_dirs = transformed_ends - transformed_starts
        lengths_new = np.sqrt(np.sum(transformed_dirs**2, axis=1))
        # 避免除以零
        lengths_new = np.maximum(lengths_new, 1e-6)
        transformed_dirs = transformed_dirs / lengths_new.reshape(-1, 1)
        
        # 转换为张量
        img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1) / 255.0
        img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1) / 255.0
        
        # 创建线段标签
        line_data = {
            'start_points1': torch.from_numpy(start_points).float(),
            'directions1': torch.from_numpy(directions).float(),
            'lengths1': torch.from_numpy(lengths).float(),
            'start_points2': torch.from_numpy(transformed_starts).float(),
            'directions2': torch.from_numpy(transformed_dirs).float(),
            'lengths2': torch.from_numpy(lengths_new).float(),
            'num_lines': self.num_lines
        }
        
        return img1_tensor, img2_tensor, line_data

class LineFeatureTrainer:
    """线特征提取器训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建保存目录
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化模型
        self.init_model()
        
        # 创建数据集和数据加载器
        self.create_datasets()
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.line_transformer.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=0.5)
    
    def init_model(self):
        """初始化模型"""
        # 加载预训练的XFeat模型
        self.xfeat_model = XFeatModel().to(self.device)
        
        # 加载预训练权重
        weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/xfeat.pt'
        if os.path.exists(weights_path):
            print(f"加载预训练权重: {weights_path}")
            self.xfeat_model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            print(f"警告: 找不到预训练权重 {weights_path}")
        
        # 冻结XFeat模型参数
        for param in self.xfeat_model.parameters():
            param.requires_grad = False
        
        # 创建线特征提取器
        self.line_transformer = LineFeatureTransformer(
            d_model=64, 
            nhead=self.args.nhead, 
            num_layers=self.args.num_layers,
            num_line_points=self.args.num_line_points
        ).to(self.device)
    
    def create_datasets(self):
        """创建训练和验证数据集"""
        self.train_dataset = SyntheticLineDataset(
            num_samples=self.args.train_samples,
            img_size=(480, 640),
            num_lines=self.args.num_lines,
            augment=True
        )
        
        self.val_dataset = SyntheticLineDataset(
            num_samples=self.args.val_samples,
            img_size=(480, 640),
            num_lines=self.args.num_lines,
            augment=True
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
    
    def extract_features(self, x):
        """提取XFeat特征"""
        with torch.no_grad():
            M1, K1, H1 = self.xfeat_model(x)
            M1 = F.normalize(M1, dim=1)
            K1h = torch.sigmoid(K1[:, 0:1])
        return M1, K1h
    
    def find_nearest_proposals(self, proposals, targets, threshold=10.0):
        """找到最接近目标线段的提案"""
        batch_size = len(proposals['start_points'])
        device = self.device
        
        matches = []
        
        for b in range(batch_size):
            # 获取当前批次的提案和目标
            prop_starts = proposals['start_points'][b].to(device)  # (N, 2)
            prop_dirs = proposals['directions'][b].to(device)      # (N, 2)
            
            target_starts = targets['start_points1'][b].to(device)  # (M, 2)
            target_dirs = targets['directions1'][b].to(device)      # (M, 2)
            
            # 检查是否有有效的提案和目标
            if len(prop_starts) == 0 or len(target_starts) == 0:
                # 如果没有有效的提案或目标，返回空匹配
                matches.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                continue
            
            # 计算起点距离矩阵
            dist_matrix = torch.cdist(prop_starts, target_starts)  # (N, M)
            
            # 计算方向相似度矩阵 (余弦相似度)
            dir_sim = torch.mm(prop_dirs, target_dirs.t())  # (N, M)
            
            # 综合评分: 距离越小越好，方向相似度越大越好
            score_matrix = dir_sim - dist_matrix / threshold
            
            # 对每个目标找最佳提案
            best_props = torch.argmax(score_matrix, dim=0)  # (M,)
            
            # 创建匹配对
            match_indices = torch.stack([
                best_props,
                torch.arange(len(target_starts), device=device)
            ], dim=0)  # (2, M)
            
            matches.append(match_indices)
        
        return matches
    
    def compute_loss(self, line_desc1, line_desc2, matches, batch_size):
        """计算描述符匹配损失"""
        device = self.device
        
        total_loss = 0
        valid_batches = 0
        
        for b in range(batch_size):
            desc1 = line_desc1[b].to(device)  # (N, D)
            desc2 = line_desc2[b].to(device)  # (N, D)
            match = matches[b]     # (2, M)
            
            # 检查是否有有效的描述符和匹配
            if desc1.shape[0] == 0 or desc2.shape[0] == 0 or match.shape[1] == 0:
                # 没有有效的描述符或匹配，跳过
                continue
            
            # 获取匹配的描述符
            matched_desc1 = desc1[match[0]]  # (M, D)
            matched_desc2 = desc2[match[1]]  # (M, D)
            
            # 计算正样本对的距离 (1 - 余弦相似度)
            pos_dist = 1.0 - F.cosine_similarity(matched_desc1, matched_desc2, dim=1)  # (M,)
            
            # 计算负样本对的距离
            # 为每个描述符随机选择一个非匹配描述符
            if desc2.shape[0] > 1:  # 确保有足够的描述符来选择负样本
                neg_indices = torch.randint(0, desc2.shape[0], (match.shape[1],), device=device)
                # 确保负样本不是正样本
                for i in range(match.shape[1]):
                    if neg_indices[i] == match[1, i] and desc2.shape[0] > 1:
                        # 如果随机选择的负样本恰好是正样本，选择另一个
                        neg_indices[i] = (match[1, i] + 1) % desc2.shape[0]
                
                neg_desc2 = desc2[neg_indices]  # (M, D)
                neg_dist = 1.0 - F.cosine_similarity(matched_desc1, neg_desc2, dim=1)  # (M,)
                
                # 计算三元组损失
                triplet_loss = F.relu(pos_dist - neg_dist + 0.2).mean()
                
                total_loss += triplet_loss
                valid_batches += 1
            else:
                # 如果没有足够的描述符来选择负样本，使用L2损失
                l2_loss = torch.mean(pos_dist)
                total_loss += l2_loss
                valid_batches += 1
        
        return total_loss / max(valid_batches, 1)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.line_transformer.train()
        total_loss = 0
        
        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for img1, img2, line_data in pbar:
            batch_size = img1.shape[0]
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            # 提取XFeat特征
            M1, K1h = self.extract_features(img1)
            M2, K2h = self.extract_features(img2)
            
            # 提取线特征
            line_desc1, line_info1 = self.line_transformer(M1, K1h)
            line_desc2, line_info2 = self.line_transformer(M2, K2h)
            
            # 找到最接近真实线段的提案
            matches1 = self.find_nearest_proposals(line_info1, line_data)
            
            # 为第二张图像准备目标
            target2 = {
                'start_points1': line_data['start_points2'],
                'directions1': line_data['directions2']
            }
            matches2 = self.find_nearest_proposals(line_info2, target2)
            
            # 计算损失
            loss = self.compute_loss(line_desc1, line_desc2, matches1, batch_size)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch+1}/{self.args.epochs}, 平均训练损失: {avg_loss:.6f}")
        
        return avg_loss
    
    def validate(self):
        """验证模型"""
        self.line_transformer.eval()
        total_loss = 0
        
        with torch.no_grad():
            for img1, img2, line_data in tqdm.tqdm(self.val_loader, desc="Validating"):
                batch_size = img1.shape[0]
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                
                # 提取XFeat特征
                M1, K1h = self.extract_features(img1)
                M2, K2h = self.extract_features(img2)
                
                # 提取线特征
                line_desc1, line_info1 = self.line_transformer(M1, K1h)
                line_desc2, line_info2 = self.line_transformer(M2, K2h)
                
                # 找到最接近真实线段的提案
                matches1 = self.find_nearest_proposals(line_info1, line_data)
                
                # 计算损失
                loss = self.compute_loss(line_desc1, line_desc2, matches1, batch_size)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        print(f"验证损失: {avg_loss:.6f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.line_transformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        torch.save(checkpoint, self.save_dir / f'line_transformer_epoch_{epoch}.pt')
        print(f"保存检查点到 {self.save_dir / f'line_transformer_epoch_{epoch}.pt'}")
    
    def train(self):
        """训练模型"""
        best_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
            
            # 定期保存检查点
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, val_loss)
        
        # 保存最终模型
        self.save_checkpoint(self.args.epochs - 1, val_loss)
        
        print(f"训练完成! 最佳验证损失: {best_loss:.6f}")

def parse_args():
    parser = argparse.ArgumentParser(description='训练线特征提取器')
    
    # 数据集参数
    parser.add_argument('--train_samples', type=int, default=10000, help='训练样本数量')
    parser.add_argument('--val_samples', type=int, default=1000, help='验证样本数量')
    parser.add_argument('--num_lines', type=int, default=20, help='每张图像的线段数量')
    
    # 模型参数
    parser.add_argument('--nhead', type=int, default=4, help='Transformer注意力头数量')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--num_line_points', type=int, default=16, help='每条线段采样点数量')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--lr_step', type=int, default=20, help='学习率衰减步长')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./weights/line_transformer', help='模型保存目录')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔(epoch)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    trainer = LineFeatureTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main() 