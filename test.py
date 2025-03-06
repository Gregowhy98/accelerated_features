import torch
import os
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import scipy.io as sio
import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=9, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=6, num_classes=1):
        super().__init__()
        self.embedding = PatchEmbedding(embed_dim=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x.mean(dim=1))
        return x

def preprocess_image(image_path, keypoints, patch_size=9):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image)
    patches = []
    valid_keypoints = []
    for kp in keypoints:
        x, y = kp
        x, y = int(x * 224 / image.size(1)), int(y * 224 / image.size(2))
        # 仅考虑图像边缘内部的关键点
        if x - patch_size//2 >= 0 and x + patch_size//2 < 224 and y - patch_size//2 >= 0 and y + patch_size//2 < 224:
            patch = image[:, y-patch_size//2:y+patch_size//2+1, x-patch_size//2:x+patch_size//2+1]
            patches.append(patch)
            valid_keypoints.append(kp)
    return torch.stack(patches), valid_keypoints

        
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



def train_model():
    dataset_folder = '/home/wenhuanyao/Dataset/Wireframe'
    model = Transformer()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 25
    mydataset = WireframeDataset(dataset_folder, use='train', transform=None)
    
    mydataloader = DataLoader(mydataset, batch_size=1, shuffle=False)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(mydataloader), desc='processing'):
        # for i, labels in mydataloader:
            img, keypoints, lines_label = data[0], data[1], data[2]
            patches, valid_keypoints = preprocess_image(img, keypoints, patch_size=9)
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            optimizer.zero_grad()
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            # running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(mydataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')


if __name__ == "__main__":
    
    train_model()
    