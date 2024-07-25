

### 라이브러리 임포트 및 데이터셋 다운로드
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```
MNIST를 사용한다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 오토인코더 모델 생성
```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(noise(x))
        x = self.decoder(x)
        return x

#노이즈 생성 함수
def noise(x):
    noise = torch.randn(x.size()) * 0.1
    noise = noise.to(device)

    return x + noise
```
오토인코더는 인코더-디코더 구조로 이루어져있다.
인코더는 3개의 선형레이어와 ReLU로 이루어져있고
디코더는 인코더를 뒤집은 형태이다.
```python
print(model)
```

<pre>
Autoencoder(
  (encoder): Sequential(
    (0): Linear(in_features=784, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=32, bias=True)
  )
  (decoder): Sequential(
    (0): Linear(in_features=32, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=784, bias=True)
    (5): Tanh()
  )
)
</pre>


### 모델 학습
```python
# 모델 초기화
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    for data, _ in train_loader:
        data = data.view(-1, 28 * 28).to(device)

        # 순전파
        output = model(data)
        loss = criterion(output, data)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

<pre>
Epoch [1/20], Loss: 0.0926
Epoch [2/20], Loss: 0.0688
Epoch [3/20], Loss: 0.0714
Epoch [4/20], Loss: 0.0522
Epoch [5/20], Loss: 0.0407
Epoch [6/20], Loss: 0.0473
Epoch [7/20], Loss: 0.0507
Epoch [8/20], Loss: 0.0417
Epoch [9/20], Loss: 0.0359
Epoch [10/20], Loss: 0.0326
Epoch [11/20], Loss: 0.0336
Epoch [12/20], Loss: 0.0362
Epoch [13/20], Loss: 0.0397
Epoch [14/20], Loss: 0.0364
Epoch [15/20], Loss: 0.0395
Epoch [16/20], Loss: 0.0371
Epoch [17/20], Loss: 0.0298
Epoch [18/20], Loss: 0.0268
Epoch [19/20], Loss: 0.0312
Epoch [20/20], Loss: 0.0330
</pre>

### 모델 평가
```python
# 테스트 데이터셋에서 몇 가지 이미지를 복원하여 시각화
model.eval()
with torch.no_grad():
    for data, _ in test_loader:
        data = data.view(-1, 28 * 28).to(device)
        output = model(data)
        output = output.view(-1, 1, 28, 28).cpu()
        break

# 원본 이미지와 복원된 이미지 시각화
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))

for images, row in zip([data.view(-1, 1, 28, 28).cpu(), output], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.numpy().squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
```

