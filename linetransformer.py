import torch
import torch.nn as nn
import torchvision.models as models

class LineSegmentTransformer(nn.Module):
    def __init__(self, num_layers, nhead, num_classes):
        super(LineSegmentTransformer, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.transformer = nn.Transformer(d_model=512, nhead=nhead, num_encoder_layers=num_layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer(x, x)
        x = self.fc(x)
        return x

model = LineSegmentTransformer(num_layers=6, nhead=8, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练代码
for epoch in range(num_epochs):
    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
# 展平

import torch
import numpy as np

def preprocess_heatmap(heatmap):
    # 假设heatmap是一个形状为 (H, W) 的numpy数组
    H, W = heatmap.shape
    # 将heatmap展平并转换为Tensor
    heatmap_flat = heatmap.flatten()
    heatmap_tensor = torch.tensor(heatmap_flat, dtype=torch.float32)
    return heatmap_tensor

# 示例热图
heatmap = np.random.rand(480, 640)
heatmap_tensor = preprocess_heatmap(heatmap)


import torch.nn as nn
import torch.nn.functional as F

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