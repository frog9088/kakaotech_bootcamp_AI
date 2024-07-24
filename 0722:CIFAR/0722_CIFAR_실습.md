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


# Data Mining 2024-1 과제 \#2



## 1. 목표 <a id="intro"></a>



1. 해당 jupyter notebook 내의 지시사항을 따라 미완성된 코드를 모두 작성하세요.



2. PyTorch를 기반으로 주어진 문제를 해결하는 코드를 완성하세요.



3. 명시된 Dataset을 활용하여 가장 좋은 성능을 보이는 모델을 코드와 함께 제출해야 합니다.





## 2. 환경 설정



CUDA를 사용할 수 있는 환경에서 진행하는 것을 권장합니다. 필요하다면 [**Google Colab**](https://colab.research.google.com/?hl=ko)을 사용하세요.



본 과제에서는 [**pytorch**](https://scikit-learn.org/stable/index.html)를 사용하여 neural network를 구현합니다.



필요하다면 다른 package를 사용해도 되지만, 모델 및 학습 프로세스 구현에는 반드시 **pytorch**가 사용되어야 합니다.



만약 다른 package를 사용했다면, 필요한 package의 list를 **requirements.txt** 로 같이 제출하세요.


## 3. 데이터셋



### 3-1. 데이터셋 개요



본 과제에서는 **CIFAR-100** 데이터셋을 바탕으로 이미지를 분류할 것입니다.



**CIFAR-100**은 50,000개의 train 이미지와 10,000개의 test 이미지로 구성되어 있습니다.



모든 이미지는 32x32 크기의 3채널 color 이미지로 주어지며 이를 100개의 class로 분류해야 합니다.



아래의 코드를 사용하여 CIFAR-100 데이터셋을 사용할 수 있습니다.



또한 `download`를 `True`로 설정하여 실행 시 데이터셋을 다운로드 받거나, 직접 [홈페이지](https://www.cs.toronto.edu/~kriz/cifar.html)에서 다운로드 받을 수 있다.



```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
```


```python
from pathlib import Path
from torchvision import datasets
import torch

DATA_ROOT = Path("./data") # modify this
trainset = datasets.CIFAR100(DATA_ROOT, train=True, download=True, transform=transform)
testset = datasets.CIFAR100(DATA_ROOT, train=False, download=True, transform=transform)

# Check how the data looks like
#testset[0]
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>

```python


train_loader = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4096, shuffle=True, num_workers=2, pin_memory=True)
```


```python
for a in train_loader :
  print(len(a[1]))
  print(type(a[0]))
  print(a[1])
  break
```

<pre>
4096
<class 'torch.Tensor'>
tensor([41, 30, 74,  ..., 98, 13,  1])
</pre>
## 4. 구현사항 - 분류모델 학습과 평가



### 4-1. Your own model **(주석 내에 코드를 작성하세요)**



아래에 자신이 사용할 모델의 코드를 작성하세요.



또한 Loss function, Optimizer 등을 정의하세요.



```python
from torch import nn, Tensor, optim

# Modify this
class Your_Model(nn.Module):
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
from torch import nn, Tensor, optim

# Modify this
class Your_Model2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(in_features=64*8*8, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=100)

    def forward(self, x:Tensor):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))

        x = x.view(-1, 64*8*8)

        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

```

### 4-2. Training **(주석 내에 코드를 작성하세요)**



아래에 본인의 모델을 CIFAR-100의 `train` dataset으로 학습하는 코드를 작성하세요.



```python
# Write your code here

# 모델과 손실 함수, 옵티마이저 정의
device = torch.device("cuda")
model = Your_Model2().to(device)  # 모델을 GPU로 이동
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 설정
num_epochs = 100


```


```python
for epoch in range(num_epochs):  # 에포크 반복
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        # 데이터와 레이블을 디바이스로 이동
        inputs, labels = inputs.cuda(), labels.cuda()

        # 옵티마이저 초기화
        optimizer.zero_grad()

        # 순전파
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 역전파 및 옵티마이저 스텝
        loss.backward()
        optimizer.step()

        # 손실 누적
        running_loss += loss.item()

    # 에포크마다 평균 손실 출력
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
```


```python
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy
```


```python

import matplotlib.pyplot as plt

def train_model(model, criterion, optimizer, train_loader, num_epochs):
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    return train_losses, train_accuracies

# Assuming train_loader is defined elsewhere
train_losses, train_accuracies = train_model(model, criterion, optimizer, train_loader, num_epochs)
```

<pre>
Epoch [1/100], Loss: 4.3294, Accuracy: 0.0546
Epoch [2/100], Loss: 3.8410, Accuracy: 0.1129
Epoch [3/100], Loss: 3.6087, Accuracy: 0.1482
Epoch [4/100], Loss: 3.4377, Accuracy: 0.1759
Epoch [5/100], Loss: 3.2934, Accuracy: 0.2024
Epoch [6/100], Loss: 3.1841, Accuracy: 0.2228
Epoch [7/100], Loss: 3.0823, Accuracy: 0.2402
Epoch [8/100], Loss: 2.9951, Accuracy: 0.2561
Epoch [9/100], Loss: 2.9221, Accuracy: 0.2689
Epoch [10/100], Loss: 2.8620, Accuracy: 0.2835
Epoch [11/100], Loss: 2.7979, Accuracy: 0.2934
Epoch [12/100], Loss: 2.7411, Accuracy: 0.3033
Epoch [13/100], Loss: 2.6884, Accuracy: 0.3141
Epoch [14/100], Loss: 2.6652, Accuracy: 0.3190
Epoch [15/100], Loss: 2.6241, Accuracy: 0.3291
Epoch [16/100], Loss: 2.5872, Accuracy: 0.3350
Epoch [17/100], Loss: 2.5551, Accuracy: 0.3418
Epoch [18/100], Loss: 2.5188, Accuracy: 0.3513
Epoch [19/100], Loss: 2.4963, Accuracy: 0.3581
Epoch [20/100], Loss: 2.4787, Accuracy: 0.3610
Epoch [21/100], Loss: 2.4480, Accuracy: 0.3644
Epoch [22/100], Loss: 2.4360, Accuracy: 0.3698
Epoch [23/100], Loss: 2.4148, Accuracy: 0.3760
Epoch [24/100], Loss: 2.3986, Accuracy: 0.3779
Epoch [25/100], Loss: 2.3754, Accuracy: 0.3815
Epoch [26/100], Loss: 2.3713, Accuracy: 0.3843
Epoch [27/100], Loss: 2.3315, Accuracy: 0.3905
Epoch [28/100], Loss: 2.3199, Accuracy: 0.3933
Epoch [29/100], Loss: 2.3181, Accuracy: 0.3929
Epoch [30/100], Loss: 2.3017, Accuracy: 0.3968
Epoch [31/100], Loss: 2.2974, Accuracy: 0.3961
Epoch [32/100], Loss: 2.2748, Accuracy: 0.4018
Epoch [33/100], Loss: 2.2633, Accuracy: 0.4059
Epoch [34/100], Loss: 2.2426, Accuracy: 0.4104
Epoch [35/100], Loss: 2.2228, Accuracy: 0.4140
Epoch [36/100], Loss: 2.2232, Accuracy: 0.4151
Epoch [37/100], Loss: 2.2071, Accuracy: 0.4141
Epoch [38/100], Loss: 2.2128, Accuracy: 0.4187
Epoch [39/100], Loss: 2.1944, Accuracy: 0.4179
Epoch [40/100], Loss: 2.1767, Accuracy: 0.4237
Epoch [41/100], Loss: 2.1601, Accuracy: 0.4285
Epoch [42/100], Loss: 2.1627, Accuracy: 0.4282
Epoch [43/100], Loss: 2.1417, Accuracy: 0.4301
Epoch [44/100], Loss: 2.1355, Accuracy: 0.4331
Epoch [45/100], Loss: 2.1229, Accuracy: 0.4372
Epoch [46/100], Loss: 2.1030, Accuracy: 0.4384
Epoch [47/100], Loss: 2.1033, Accuracy: 0.4432
Epoch [48/100], Loss: 2.1039, Accuracy: 0.4412
Epoch [49/100], Loss: 2.0892, Accuracy: 0.4455
Epoch [50/100], Loss: 2.0785, Accuracy: 0.4451
Epoch [51/100], Loss: 2.0867, Accuracy: 0.4438
Epoch [52/100], Loss: 2.0696, Accuracy: 0.4486
Epoch [53/100], Loss: 2.0781, Accuracy: 0.4491
Epoch [54/100], Loss: 2.0594, Accuracy: 0.4497
Epoch [55/100], Loss: 2.0430, Accuracy: 0.4529
Epoch [56/100], Loss: 2.0263, Accuracy: 0.4558
Epoch [57/100], Loss: 2.0295, Accuracy: 0.4553
Epoch [58/100], Loss: 2.0255, Accuracy: 0.4598
Epoch [59/100], Loss: 2.0137, Accuracy: 0.4565
Epoch [60/100], Loss: 2.0193, Accuracy: 0.4571
Epoch [61/100], Loss: 2.0113, Accuracy: 0.4608
Epoch [62/100], Loss: 1.9922, Accuracy: 0.4631
Epoch [63/100], Loss: 1.9913, Accuracy: 0.4652
Epoch [64/100], Loss: 1.9872, Accuracy: 0.4656
Epoch [65/100], Loss: 1.9779, Accuracy: 0.4680
Epoch [66/100], Loss: 1.9646, Accuracy: 0.4700
Epoch [67/100], Loss: 1.9727, Accuracy: 0.4671
Epoch [68/100], Loss: 1.9730, Accuracy: 0.4680
Epoch [69/100], Loss: 1.9601, Accuracy: 0.4706
Epoch [70/100], Loss: 1.9491, Accuracy: 0.4744
Epoch [71/100], Loss: 1.9605, Accuracy: 0.4715
Epoch [72/100], Loss: 1.9446, Accuracy: 0.4703
Epoch [73/100], Loss: 1.9318, Accuracy: 0.4785
Epoch [74/100], Loss: 1.9282, Accuracy: 0.4760
Epoch [75/100], Loss: 1.9358, Accuracy: 0.4779
Epoch [76/100], Loss: 1.9130, Accuracy: 0.4800
Epoch [77/100], Loss: 1.9133, Accuracy: 0.4811
Epoch [78/100], Loss: 1.9174, Accuracy: 0.4811
Epoch [79/100], Loss: 1.9047, Accuracy: 0.4848
Epoch [80/100], Loss: 1.8962, Accuracy: 0.4860
Epoch [81/100], Loss: 1.8975, Accuracy: 0.4847
Epoch [82/100], Loss: 1.8957, Accuracy: 0.4852
Epoch [83/100], Loss: 1.8866, Accuracy: 0.4857
Epoch [84/100], Loss: 1.8707, Accuracy: 0.4922
Epoch [85/100], Loss: 1.8827, Accuracy: 0.4875
Epoch [86/100], Loss: 1.8765, Accuracy: 0.4916
Epoch [87/100], Loss: 1.8724, Accuracy: 0.4908
Epoch [88/100], Loss: 1.8730, Accuracy: 0.4909
Epoch [89/100], Loss: 1.8588, Accuracy: 0.4972
Epoch [90/100], Loss: 1.8626, Accuracy: 0.4962
Epoch [91/100], Loss: 1.8500, Accuracy: 0.4967
Epoch [92/100], Loss: 1.8484, Accuracy: 0.4973
Epoch [93/100], Loss: 1.8449, Accuracy: 0.4970
Epoch [94/100], Loss: 1.8441, Accuracy: 0.4949
Epoch [95/100], Loss: 1.8352, Accuracy: 0.4982
Epoch [96/100], Loss: 1.8407, Accuracy: 0.4990
Epoch [97/100], Loss: 1.8323, Accuracy: 0.5002
Epoch [98/100], Loss: 1.8249, Accuracy: 0.4993
Epoch [99/100], Loss: 1.8116, Accuracy: 0.5032
Epoch [100/100], Loss: 1.8029, Accuracy: 0.5046
</pre>



```python
num_epochs
```

<pre>
100
</pre>

```python
plt.figure(figsize=(10, 5))

# Plotting Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over epochs')
plt.legend()

# Plotting Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over epochs')
plt.legend()

plt.tight_layout()
plt.show()
```

<pre>
<Figure size 1000x500 with 2 Axes>
</pre>
### 4-3. Evaluation **(주석 내에 코드를 작성하세요)**



아래에 본인의 모델을 CIFAR-100의 `test` dataset으로 정확도를 평가하고 출력하는 코드를 작성하세요.



```python
# Write your code here

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in train_loader:

        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the', total, f'test images: {accuracy:.2f}%')
```

<pre>
Accuracy of the model on the 10000 test images: 58.79%
</pre>

```python
# Write your code here

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:

        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the', total, f'test images: {accuracy:.2f}%')
```

<pre>
Accuracy of the model on the 10000 test images: 48.68%
</pre>

```python
# Write your code here
torch.cuda.empty_cache()
# 모델과 손실 함수, 옵티마이저 정의
device = torch.device("cuda")
model = Your_Model().to(device)  # 모델을 GPU로 이동
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 설정
num_epochs = 100

train_losses, train_accuracies = train_model(model, criterion, optimizer, train_loader, num_epochs)
```

<pre>
Epoch [1/100], Loss: 4.3454, Accuracy: 0.0398
Epoch [2/100], Loss: 3.9002, Accuracy: 0.0886
Epoch [3/100], Loss: 3.6713, Accuracy: 0.1194
Epoch [4/100], Loss: 3.4919, Accuracy: 0.1486
Epoch [5/100], Loss: 3.3251, Accuracy: 0.1800
Epoch [6/100], Loss: 3.1511, Accuracy: 0.2104
Epoch [7/100], Loss: 3.0118, Accuracy: 0.2373
Epoch [8/100], Loss: 2.8955, Accuracy: 0.2596
Epoch [9/100], Loss: 2.7827, Accuracy: 0.2832
Epoch [10/100], Loss: 2.6615, Accuracy: 0.3080
Epoch [11/100], Loss: 2.5674, Accuracy: 0.3255
Epoch [12/100], Loss: 2.5237, Accuracy: 0.3364
Epoch [13/100], Loss: 2.4449, Accuracy: 0.3528
Epoch [14/100], Loss: 2.3905, Accuracy: 0.3661
Epoch [15/100], Loss: 2.3114, Accuracy: 0.3807
Epoch [16/100], Loss: 2.2706, Accuracy: 0.3908
Epoch [17/100], Loss: 2.2161, Accuracy: 0.4046
Epoch [18/100], Loss: 2.1696, Accuracy: 0.4135
Epoch [19/100], Loss: 2.1245, Accuracy: 0.4237
Epoch [20/100], Loss: 2.0726, Accuracy: 0.4339
Epoch [21/100], Loss: 2.0379, Accuracy: 0.4454
Epoch [22/100], Loss: 1.9994, Accuracy: 0.4555
Epoch [23/100], Loss: 1.9730, Accuracy: 0.4604
Epoch [24/100], Loss: 1.9527, Accuracy: 0.4655
Epoch [25/100], Loss: 1.9203, Accuracy: 0.4720
Epoch [26/100], Loss: 1.8668, Accuracy: 0.4842
Epoch [27/100], Loss: 1.8364, Accuracy: 0.4886
Epoch [28/100], Loss: 1.8136, Accuracy: 0.4959
Epoch [29/100], Loss: 1.7916, Accuracy: 0.5021
Epoch [30/100], Loss: 1.7516, Accuracy: 0.5092
Epoch [31/100], Loss: 1.7308, Accuracy: 0.5150
Epoch [32/100], Loss: 1.7051, Accuracy: 0.5256
Epoch [33/100], Loss: 1.6908, Accuracy: 0.5274
Epoch [34/100], Loss: 1.6755, Accuracy: 0.5291
Epoch [35/100], Loss: 1.6812, Accuracy: 0.5316
Epoch [36/100], Loss: 1.6285, Accuracy: 0.5413
Epoch [37/100], Loss: 1.6038, Accuracy: 0.5476
Epoch [38/100], Loss: 1.5851, Accuracy: 0.5516
Epoch [39/100], Loss: 1.5704, Accuracy: 0.5539
Epoch [40/100], Loss: 1.5407, Accuracy: 0.5612
Epoch [41/100], Loss: 1.5459, Accuracy: 0.5602
Epoch [42/100], Loss: 1.5124, Accuracy: 0.5643
Epoch [43/100], Loss: 1.5086, Accuracy: 0.5689
Epoch [44/100], Loss: 1.4843, Accuracy: 0.5751
Epoch [45/100], Loss: 1.4633, Accuracy: 0.5799
Epoch [46/100], Loss: 1.4412, Accuracy: 0.5856
Epoch [47/100], Loss: 1.4462, Accuracy: 0.5843
Epoch [48/100], Loss: 1.4294, Accuracy: 0.5873
Epoch [49/100], Loss: 1.4080, Accuracy: 0.5931
Epoch [50/100], Loss: 1.3817, Accuracy: 0.6010
Epoch [51/100], Loss: 1.3844, Accuracy: 0.5986
Epoch [52/100], Loss: 1.3515, Accuracy: 0.6086
Epoch [53/100], Loss: 1.3575, Accuracy: 0.6048
Epoch [54/100], Loss: 1.3443, Accuracy: 0.6083
Epoch [55/100], Loss: 1.3240, Accuracy: 0.6152
Epoch [56/100], Loss: 1.3078, Accuracy: 0.6179
Epoch [57/100], Loss: 1.3046, Accuracy: 0.6190
Epoch [58/100], Loss: 1.2777, Accuracy: 0.6240
Epoch [59/100], Loss: 1.2586, Accuracy: 0.6292
Epoch [60/100], Loss: 1.2450, Accuracy: 0.6345
Epoch [61/100], Loss: 1.2541, Accuracy: 0.6314
Epoch [62/100], Loss: 1.2342, Accuracy: 0.6344
Epoch [63/100], Loss: 1.2363, Accuracy: 0.6373
Epoch [64/100], Loss: 1.2178, Accuracy: 0.6369
Epoch [65/100], Loss: 1.1959, Accuracy: 0.6428
Epoch [66/100], Loss: 1.1983, Accuracy: 0.6460
Epoch [67/100], Loss: 1.1811, Accuracy: 0.6486
Epoch [68/100], Loss: 1.1777, Accuracy: 0.6494
Epoch [69/100], Loss: 1.1579, Accuracy: 0.6533
Epoch [70/100], Loss: 1.1558, Accuracy: 0.6554
Epoch [71/100], Loss: 1.1401, Accuracy: 0.6597
Epoch [72/100], Loss: 1.1435, Accuracy: 0.6584
Epoch [73/100], Loss: 1.1242, Accuracy: 0.6642
Epoch [74/100], Loss: 1.1147, Accuracy: 0.6669
Epoch [75/100], Loss: 1.0956, Accuracy: 0.6698
Epoch [76/100], Loss: 1.0934, Accuracy: 0.6728
Epoch [77/100], Loss: 1.0919, Accuracy: 0.6720
Epoch [78/100], Loss: 1.0806, Accuracy: 0.6768
Epoch [79/100], Loss: 1.0785, Accuracy: 0.6740
Epoch [80/100], Loss: 1.0479, Accuracy: 0.6822
Epoch [81/100], Loss: 1.0364, Accuracy: 0.6872
Epoch [82/100], Loss: 1.0422, Accuracy: 0.6854
Epoch [83/100], Loss: 1.0266, Accuracy: 0.6903
Epoch [84/100], Loss: 1.0206, Accuracy: 0.6894
Epoch [85/100], Loss: 1.0043, Accuracy: 0.6958
Epoch [86/100], Loss: 0.9961, Accuracy: 0.6984
Epoch [87/100], Loss: 1.0042, Accuracy: 0.6977
Epoch [88/100], Loss: 1.0053, Accuracy: 0.6943
Epoch [89/100], Loss: 0.9912, Accuracy: 0.6967
Epoch [90/100], Loss: 0.9825, Accuracy: 0.7018
Epoch [91/100], Loss: 0.9597, Accuracy: 0.7051
Epoch [92/100], Loss: 0.9601, Accuracy: 0.7083
Epoch [93/100], Loss: 0.9501, Accuracy: 0.7063
Epoch [94/100], Loss: 0.9424, Accuracy: 0.7114
Epoch [95/100], Loss: 0.9326, Accuracy: 0.7135
Epoch [96/100], Loss: 0.9247, Accuracy: 0.7169
Epoch [97/100], Loss: 0.9054, Accuracy: 0.7216
Epoch [98/100], Loss: 0.9226, Accuracy: 0.7175
Epoch [99/100], Loss: 0.9133, Accuracy: 0.7168
Epoch [100/100], Loss: 0.9068, Accuracy: 0.7195
</pre>

```python
# Write your code here

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in train_loader:

        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')
```

<pre>
Accuracy of the model on the 10000 test images: 77.74%
</pre>

```python
# Write your code here

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:

        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the', total, f'test images: {accuracy:.2f}%')
```

<pre>
Accuracy of the model on the 10000 test images: 58.53%
</pre>