import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as sio
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def preprocess_keypoints(keypoints):
    # 假设keypoints是一个形状为 (N, 2) 的numpy数组
    N, _ = keypoints.shape
    # 将keypoints展平并转换为Tensor
    keypoints_flat = keypoints.flatten()
    keypoints_tensor = torch.tensor(keypoints_flat, dtype=torch.float32)
    return keypoints_tensor


# class LineSegmentTransformer(nn.Module):
#     def __init__(self, input_dim, num_heads, num_layers, hidden_dim):
#         super(LineSegmentTransformer, self).__init__()
#         self.embedding = nn.Linear(input_dim, hidden_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(hidden_dim, 2)  # 输出线段的两个端点

#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.unsqueeze(1)  # 添加batch维度
#         x = self.transformer(x)
#         x = x.squeeze(1)  # 移除batch维度
#         x = self.fc(x)
#         return x
    
class LineSegmentTransformer(nn.Module):
    def __init__(self, num_layers, nhead, num_classes):
        super(LineSegmentTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=512, nhead=nhead, num_encoder_layers=num_layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer(x, x)
        x = self.fc(x)
        return x

# 模型参数
input_dim = 4000
num_heads = 8
num_layers = 6
hidden_dim = 512

d_model = 512
num_layers = 6
nhead = 8
num_classes = 2

# 创建模型
# model = LineSegmentTransformer(input_dim, num_heads, num_layers, hidden_dim)
model = LineSegmentTransformer(num_layers, nhead, num_classes)

      
class WireframeDataset(Dataset):
    def __init__(self, dataset_folder, use='train', transform=None):
        # init
        self.dataset_folder = dataset_folder
        if use not in ['train', 'test']:
            raise ValueError('Invalid value for use. Must be one of [train, test]')
        else:
            self.use = use
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((480, 640)),
                transforms.ToTensor()
                ])
        
        # folder path
        self.v11_folder = os.path.join(self.dataset_folder, 'v1.1')
        self.img_folder = os.path.join(self.v11_folder, use)
        self.line_mat_folder = os.path.join(self.dataset_folder, 'line_mat')
        self.xfeat_kpts_folder = os.path.join(self.dataset_folder, 'xfeat_kpts')
        
        # item list
        list_file_path = os.path.join(self.v11_folder, self.use + '.txt')
        with open(list_file_path, 'r') as f:
            item_list = f.readlines()
        self.item_list = sorted([item.strip() for item in item_list])
        self.N = len(item_list)
        print('Number of images in {} set: {}'.format(self.use, self.N))
        pass
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # load img
        img_id = self.item_list[idx]
        img_path = os.path.join(self.v11_folder, 'all', img_id)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_tensor = self.transform(img)
        # load xfeat kpts
        kpts_path = os.path.join(self.xfeat_kpts_folder, img_id.replace('.jpg','_kpts.npy'))
        kpts = np.load(kpts_path)
        # load line
        lines_path = os.path.join(self.line_mat_folder, img_id.replace('.jpg','_line.mat'))
        lines = sio.loadmat(lines_path).get('lines')
        return img_tensor, kpts, lines

dataset_path = '/home/wenhuanyao/Dataset/Wireframe'
dataset = WireframeDataset(dataset_folder=dataset_path, use='train')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(100):  # 示例训练循环
    for i, data in enumerate(dataloader):
        _, kpts, lines = data
        kpts = kpts.numpy()
        kpts_tensor = preprocess_keypoints(kpts[0])
        optimizer.zero_grad()
        output = model(kpts_tensor)
        loss = criterion(output, lines)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    

# 推理
# with torch.no_grad():
#     test_output = model(heatmap_tensor)
#     print("Predicted line segment endpoints:", test_output)
    

    
    
