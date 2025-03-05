
from torch.utils.data import Dataset
import scipy.io as sio
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pickle

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
        self.img_folder = os.path.join(self.dataset_folder, 'v1.1')
        self.line_mat_folder = os.path.join(self.dataset_folder, 'line_mat')
        self.point_line_pkl_folder = os.path.join(self.dataset_folder, 'pointlines')
        self.xfeat_gt_folder = os.path.join(self.dataset_folder, 'xfeat_gt')
        
        # item list
        list_file_path = os.path.join(self.img_folder, self.use + '.txt')
        with open(list_file_path, 'r') as f:
            item_list = f.readlines()
        item_list = sorted([item.strip() for item in item_list])
        self.N = len(item_list)
        print('Number of images in {} set: {}'.format(self.use, self.N))
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # load img
        img_path = self.img_list[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_tensor = self.transform(img)
        # load line mat
        line_mat_path = self.line_mat_list[idx]
        line_mat = sio.loadmat(line_mat_path).get('lines')
        # load point line pkl
        # pointline_path = self.pointline_list[idx]
        # with open(pointline_path, 'rb') as f:
        #     pointline = pickle.load(f)
        # load xfeat gt
        xfeat_gt_path = self.xfeat_gt_list[idx]
        with open(xfeat_gt_path, 'rb') as f:
            xfeat_gt = pickle.load(f)
        
        
        return img_tensor, torch.tensor(line_mat), xfeat_gt
        # return img_tensor bx
       
       
class WireframePrepocessDataset(Dataset):
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
        self.img_folder = os.path.join(self.v11_folder, self.use)
        self.xfeat_gt_folder = os.path.join(self.dataset_folder, 'xfeat_gt')
        
        # item list
        list_file_path = os.path.join(self.v11_folder, self.use + '.txt')
        with open(list_file_path, 'r') as f:
            item_list = f.readlines()
        item_list = sorted([item.strip() for item in item_list])
        self.N = len(item_list)
        print('Number of images in {} set: {}'.format(self.use, self.N))
        
        self.img_list = [os.path.join(self.img_folder, x) for x in item_list]
        
    def __len__(self):
        return self.N - 1 
        # return self.N - 1 
    
    def __getitem__(self, idx):
        # load img
        img_path = self.img_list[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_tensor = self.transform(img)
        return img_tensor, img_path
    
if __name__ == '__main__':
    testPath = '/home/wenhuanyao/Dataset/Wireframe/' 
    # mydataset = WireframeDataset(testPath, use='train', transform=None)
    mydataset = WireframePrepocessDataset(testPath, use='train', transform=None)
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)

    for i, data in enumerate(mydatasetloader):
        ret = data
        pass