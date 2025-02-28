"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""

import numpy as np
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from modules.xfeat import XFeat
from utils import draw_keypoints_on_image, draw_scores_heatmap, visualize_descriptors

os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

weights = '/home/wenhuanyao/accelerated_features/checkpoints/xfeat_synthetic_9000.pth'
save_folder = '/home/wenhuanyao/accelerated_features/output'
xfeat = XFeat(weights=weights)  #, top_k=4096, detection_threshold=0.05)

#Random input
# x = torch.randn(1,3,480,640)

test_img_path = '/home/wenhuanyao/accelerated_features/assets/demo_pic_3.png'
test_img = cv2.imread(test_img_path)
x = torch.from_numpy(test_img).permute(2,0,1).unsqueeze(0).float()
output = xfeat.detectAndCompute(x, top_k = 2000)[0]
print("keypoints: ", output['keypoints'].shape)
print("descriptors: ", output['descriptors'].shape)
print("scores: ", output['scores'].shape)

# Convert keypoints to OpenCV format and draw them on the image
img_with_keypoints = draw_keypoints_on_image(test_img, output['keypoints'])
cv2.imwrite(os.path.join(save_folder, 'keypoints_vis.png'), img_with_keypoints)

# Plot the scores as a heatmap
plt = draw_scores_heatmap(output['keypoints'], output['scores'])
plt.savefig(os.path.join(save_folder, 'scores_heatmap_vis.png'))
plt.close()

# Visualize descriptors using PCA
plt = visualize_descriptors(output['descriptors'], output['scores'])
plt.savefig(os.path.join(save_folder, 'descriptors_vis.png'))
plt.close()


pass


# x = torch.randn(1,3,480,640)
# # Stress test
# for i in tqdm.tqdm(range(100), desc="Stress test on VGA resolution"):
# 	output = xfeat.detectAndCompute(x, top_k = 4096)

# # Batched mode
# x = torch.randn(4,3,480,640)
# outputs = xfeat.detectAndCompute(x, top_k = 4096)
# print("# detected features on each batch item:", [len(o['keypoints']) for o in outputs])

# # Match two images with sparse features
# x1 = torch.randn(1,3,480,640)
# x2 = torch.randn(1,3,480,640)
# mkpts_0, mkpts_1 = xfeat.match_xfeat(x1, x2)

# # Match two images with semi-dense approach -- batched mode with batch size 4
# x1 = torch.randn(4,3,480,640)
# x2 = torch.randn(4,3,480,640)
# matches_list = xfeat.match_xfeat_star(x1, x2)
# print(matches_list[0].shape)
