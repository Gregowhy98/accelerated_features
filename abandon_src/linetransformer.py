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
        
        
        
        
        
        
