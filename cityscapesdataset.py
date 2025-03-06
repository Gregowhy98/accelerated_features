import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pickle

def list_files(directory):
    filenames = os.listdir(directory)
    file_paths = [os.path.join(directory, filename) for filename in filenames]
    return file_paths

class CityScapesDataset(Dataset):
    def __init__(self, dataset_folder, use='train', transform=None, device='cpu'):
        # init
        if use not in ['train', 'val', 'test']:
            raise ValueError('Invalid value for use. Must be one of [train, val, test]')
        
        self.dataset_folder = os.path.join(dataset_folder, use)
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((480, 640)),
                transforms.ToTensor()
                ])
        self.device = device
        
        # folder path
        self.raw_img_folder = os.path.join(self.dataset_folder, 'raw_img')
        self.weighted_seg_folder = os.path.join(self.dataset_folder, 'weighted_seg')
        self.xfeat_folder = os.path.join(self.dataset_folder, 'xfeat')
        
        # item list
        self.raw_img_files = list_files(self.raw_img_folder)
        self.raw_img_files.sort()
        self.N = len(self.raw_img_files)
        print('find', self.N, 'raw imgs.')
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx): 
        # raw img 3 channels, 0-255, HxWxC, rgb
        img_name = self.raw_img_files[idx].split('/')[-1]
        raw_img = cv2.imread(self.raw_img_files[idx], cv2.COLOR_BGR2RGB)
        raw_img = self.transform(raw_img)
        seg_gt = cv2.imread(os.path.join(self.weighted_seg_folder, img_name).replace('leftImg8bit', 'gtFine_weighted'), cv2.IMREAD_GRAYSCALE)
        seg_gt = self.transform(seg_gt)
        
        xfeat_gt_path = os.path.join(self.xfeat_folder, img_name).replace('.png', '_xfeat.pkl')
        with open(xfeat_gt_path, 'rb') as f:
            xfeat_gt = pickle.load(f)
        for key in xfeat_gt.keys():
            xfeat_gt[key] = torch.tensor(xfeat_gt[key]).to(self.device)
            
        data = {
            'raw_img': raw_img,
            'seg_gt': seg_gt,
            'xfeat_gt': xfeat_gt,     
        }
        return data
    
    
if __name__ == '__main__':
    testPath = '/home/wenhuanyao/Dataset/cityscapes/' 
    mydataset = CityScapesDataset(testPath, use='train', transform=None, device='cpu')
    mydatasetloader = DataLoader(mydataset, batch_size=16, shuffle=False)

    for i, data in enumerate(mydatasetloader):
        data_list = data
        pass