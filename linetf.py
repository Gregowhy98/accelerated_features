import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def preprocess_keypoints(keypoints):
    # 假设keypoints是一个形状为 (N, 2) 的numpy数组
    N, _ = keypoints.shape
    # 将keypoints展平并转换为Tensor
    keypoints_flat = keypoints.flatten()
    keypoints_tensor = torch.tensor(keypoints_flat, dtype=torch.float32)
    return keypoints_tensor

# 示例热图
keypoints = np.array([[100, 200], [150, 250], [200, 300]])
keypoints_tensor = preprocess_keypoints(keypoints)


class LineSegmentTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim):
        super(LineSegmentTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(hidden_dim, num_heads, num_layers)
        self.fc = nn.Linear(hidden_dim, 2)  # 输出线段的两个端点

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 添加batch维度
        x = self.transformer(x, x)
        x = x.squeeze(1)  # 移除batch维度
        x = self.fc(x)
        return x

# 模型参数
input_dim = 480 * 640
num_heads = 8
num_layers = 6
hidden_dim = 512

# 创建模型
model = LineSegmentTransformer(input_dim, num_heads, num_layers, hidden_dim)

# 假设我们有训练数据和标签
train_data = [heatmap_tensor]  # 示例训练数据
train_labels = [torch.tensor([100, 200, 300, 400])]  # 示例标签，表示线段的两个端点

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(100):  # 示例训练循环
    for data, label in zip(train_data, train_labels):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 推理
with torch.no_grad():
    test_output = model(heatmap_tensor)
    print("Predicted line segment endpoints:", test_output)
    

    
    
