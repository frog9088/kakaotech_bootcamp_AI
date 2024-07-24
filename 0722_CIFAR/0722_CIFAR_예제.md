---
layout: single
title:  "jupyter notebook 변환하기!"
categories: coding
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>



```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
```


```python
# 데이터셋을 전처리합니다.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# CIFAR-10 데이터셋을 다운로드하고 로드합니다.
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

<pre>
Downloading https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz to ./data/cifar-100-python.tar.gz
</pre>
<pre>
100%|██████████| 169001437/169001437 [00:18<00:00, 9274628.99it/s]
</pre>
<pre>
Extracting ./data/cifar-100-python.tar.gz to ./data
Files already downloaded and verified
</pre>

```python
print(trainset)
```

<pre>
Dataset CIFAR100
    Number of datapoints: 50000
    Root location: ./data
    Split: Train
    StandardTransform
Transform: Compose(
               ToTensor()
               Normalize(mean=(0.5,), std=(0.5,))
           )
</pre>

```python
# torchvision.models 모듈을 사용하여 사전 정의된 CNN 모델을 불러옵니다.
import torchvision.models as models

# Pretrained VGG16 모델 사용 (사전 훈련된 가중치 사용)
model = models.vgg16(pretrained=True)

# CIFAR-10 데이터셋의 클래스 수에 맞게 마지막 레이어를 수정합니다.
model.classifier[6] = nn.Linear(4096, 100)
```

<pre>
/usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
100%|██████████| 528M/528M [00:03<00:00, 167MB/s]
</pre>

```python
print(model)
```

<pre>
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=100, bias=True)
  )
)
</pre>

```python
# GPU가 사용 가능한지 확인하고, 사용할 수 있다면 device를 설정합니다.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
```

<pre>
Using device: cuda:0
</pre>

```python
# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```


```python
# 모델 훈련 함수
def train_model(model, trainloader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 순전파, 역전파, 최적화
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 손실 출력
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
```


```python
# 모델 평가 함수
def evaluate_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
```


```python
model = model.to(device)
```


```python
# 모델 훈련
train_model(model, trainloader, criterion, optimizer, num_epochs=10)
```

<pre>
Finished Training
</pre>

```python
# 모델 평가
evaluate_model(model, testloader)
```

<pre>
Accuracy of the network on the 10000 test images: 61.15%
</pre>

```python
import torch.nn.functional as F  # F를 nn.functional 모듈로 임포트
# 직접 구현한 CNN 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 첫 번째 합성곱 층: 입력 채널 3, 출력 채널 6, 커널 크기 5
        self.pool = nn.MaxPool2d(2, 2)   # 최대 풀링 층: 커널 크기 2, 스트라이드 2
        self.conv2 = nn.Conv2d(6, 16, 5) # 두 번째 합성곱 층: 입력 채널 6, 출력 채널 16, 커널 크기 5
        self.fc1 = nn.Linear(16 * 5 * 5, 1200) # 첫 번째 완전 연결 층: 입력 크기 16*5*5, 출력 크기 120
        self.fc2 = nn.Linear(1200, 840)         # 두 번째 완전 연결 층: 입력 크기 120, 출력 크기 84
        self.fc3 = nn.Linear(840, 400)          # 세 번째 완전 연결 층: 입력 크기 84, 출력 크기 10 (클래스 수)
        self.fc4 = nn.Linear(400, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 첫 번째 합성곱 층 -> ReLU -> 풀링
        x = self.pool(F.relu(self.conv2(x))) # 두 번째 합성곱 층 -> ReLU -> 풀링
        x = x.view(-1, 16 * 5 * 5)           # 특성 맵을 1차원 벡터로 변환
        x = F.relu(self.fc1(x))              # 첫 번째 완전 연결 층 -> ReLU
        x = F.relu(self.fc2(x))              # 두 번째 완전 연결 층 -> ReLU
        x = F.relu(self.fc3(x))
        x = self.fc4(x)                      # 세 번째 완전 연결 층
        return x
```


```python
# 모델 인스턴스 생성
net = Net()
```


```python
print(net)
```

<pre>
Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=1200, bias=True)
  (fc2): Linear(in_features=1200, out_features=840, bias=True)
  (fc3): Linear(in_features=840, out_features=400, bias=True)
  (fc4): Linear(in_features=400, out_features=100, bias=True)
)
</pre>

```python
net = net.to(device)
```


```python
# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```


```python
# 모델 훈련
train_model(net, trainloader, criterion, optimizer, num_epochs=10)
```

<pre>
Finished Training
</pre>

```python
# 모델 평가
evaluate_model(net, testloader)
```

<pre>
Accuracy of the network on the 10000 test images: 2.82%
</pre>

```python
"""
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        """
```

<pre>
'\nclass MyNeuralNetwork(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.flatten = nn.Flatten()\n        self.linear_relu_stack = nn.Sequential(\n            nn.Linear(28*28, 512),\n            nn.ReLU(),\n            nn.Linear(512, 512),\n            nn.ReLU(),\n            nn.Linear(512, 10),\n        )\n\n    def forward(self, x):\n        x = self.flatten(x)\n        logits = self.linear_relu_stack(x)\n        return logits\n        '
</pre>

```python
from torch import nn, Tensor, optim

# Modify this
class Your_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(in_features=256*3*3, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=100)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))


        x = x.view(-1, 256*3*3)

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

```


```python
# 모델 인스턴스 생성
my_model = Your_Model()
my_model.to(device)
```

<pre>
Your_Model(
  (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(64, 128, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout): Dropout(p=0.4, inplace=False)
  (fc1): Linear(in_features=2304, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=512, bias=True)
  (fc3): Linear(in_features=512, out_features=100, bias=True)
)
</pre>

```python
# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_model.parameters(), lr=0.01, momentum=0.9)
```


```python
# 모델 훈련
train_model(my_model, trainloader, criterion, optimizer, num_epochs=10)
# 모델 평가
evaluate_model(my_model, testloader)
```

<pre>
Finished Training
Accuracy of the network on the 10000 test images: 50.74%
</pre>

```python
# 모델 평가
evaluate_model(my_model, testloader)
```

<pre>
Accuracy of the network on the 10000 test images: 50.74%
</pre>

```python
'''
import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))
# 정답(label) 출력
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(16)))
'''
```

<pre>
"\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# 이미지를 보여주기 위한 함수\n\ndef imshow(img):\n    img = img / 2 + 0.5     # unnormalize\n    npimg = img.numpy()\n    plt.imshow(np.transpose(npimg, (1, 2, 0)))\n    plt.show()\n\n\n# 학습용 이미지를 무작위로 가져오기\ndataiter = iter(trainloader)\nimages, labels = next(dataiter)\n\n# 이미지 보여주기\nimshow(torchvision.utils.make_grid(images))\n# 정답(label) 출력\nprint(' '.join(f'{classes[labels[j]]:5s}' for j in range(16)))\n"
</pre>

```python
from torch import nn, Tensor, optim

# Modify this
class Your_Model2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(in_features=512*2*2, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=100)
    def forward(self, x:Tensor):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
        x = self.pool(nn.functional.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 512*2*2)

        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

```


```python
# 모델 인스턴스 생성
my_model2 = Your_Model2()
my_model2.to(device)
# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_model2.parameters(), lr=0.001, momentum=0.9)
# 모델 훈련
train_model(my_model2, trainloader, criterion, optimizer, num_epochs=10)
# 모델 평가
evaluate_model(my_model2, testloader)
```

<pre>
Finished Training
Accuracy of the network on the 10000 test images: 33.59%
</pre>