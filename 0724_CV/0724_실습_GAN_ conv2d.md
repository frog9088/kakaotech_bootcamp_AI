

### 라이브러리 임포트 및 데이터셋 다운로드
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 생성자 모델
```python
d_noise = 100
d_hidden_1 = 256
d_hidden_2 = 256
image_dim = 28*28
batch_size = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(d_noise, 256, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), d_noise, 1, 1)  # 입력 잡음 벡터를 (batch_size, d_noise, 1, 1) 형태로 변환
        x = self.generator(x)
        return x

    def forward(self, x):
        x = self.generator(x)
        return x
```
3개의 ConvTranspose2d layer를 연결해서 구성한다.
ConvTranspose2d를 통해 특성맵을 추출하고 마지막으로 28* 28 크기 이미지로 출력한다.
앞서 데이터셋을 transform을 통해 [-1,1]범위로 정규화 했기에 tanh를 통해 범위를 맞춰준다.


### 판별자 모델
```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(28*28, d_hidden_1),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden_1, d_hidden_2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden_2, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        x = self.discriminator(x)
        return x
```

```python
disc = Discriminator().to(device)
gen = Generator().to(device)

fixed_noise = torch.randn(batch_size, d_noise).view(batch_size, d_noise, 1, 1).to(device)

lr = 3e-4
num_epochs = 50

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

step = 0
```
옵티마이저는 따로 설정한다. (생성자와 판별자가 개별적으로 학습되어야 하기 때문에)


### 학습
```python
fake_list = []
real_list = []

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, d_noise).view(batch_size, d_noise, 1, 1).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.view(-1, 784)).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake.view(-1, 784)).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)

                fake_list.append(fake)
                real_list.append(data)
                step += 1
```

<pre>
Epoch [0/50] Loss D: 0.7218, loss G: 0.7052
Epoch [1/50] Loss D: 0.6322, loss G: 0.9314
Epoch [2/50] Loss D: 0.6251, loss G: 0.9769
Epoch [3/50] Loss D: 0.5735, loss G: 1.0137
Epoch [4/50] Loss D: 0.6645, loss G: 0.7921
Epoch [5/50] Loss D: 0.6416, loss G: 0.7984
Epoch [6/50] Loss D: 0.6553, loss G: 0.7159
Epoch [7/50] Loss D: 0.6116, loss G: 0.9643
Epoch [8/50] Loss D: 0.6208, loss G: 0.8218
Epoch [9/50] Loss D: 0.6630, loss G: 0.7133
Epoch [10/50] Loss D: 0.6810, loss G: 0.7121
Epoch [11/50] Loss D: 0.6692, loss G: 0.7387
Epoch [12/50] Loss D: 0.6394, loss G: 0.7014
Epoch [13/50] Loss D: 0.7154, loss G: 0.6390
Epoch [14/50] Loss D: 0.6842, loss G: 0.6811
Epoch [15/50] Loss D: 0.5978, loss G: 0.9690
Epoch [16/50] Loss D: 0.7135, loss G: 0.7223
Epoch [17/50] Loss D: 0.7037, loss G: 0.7627
Epoch [18/50] Loss D: 0.6375, loss G: 0.9172
Epoch [19/50] Loss D: 0.6840, loss G: 0.6674
Epoch [20/50] Loss D: 0.6545, loss G: 0.7367
Epoch [21/50] Loss D: 0.7034, loss G: 0.6993
Epoch [22/50] Loss D: 0.6657, loss G: 0.7499
Epoch [23/50] Loss D: 0.6705, loss G: 0.7837
Epoch [24/50] Loss D: 0.6318, loss G: 0.8863
Epoch [25/50] Loss D: 0.6933, loss G: 0.7991
Epoch [26/50] Loss D: 0.6937, loss G: 0.7370
Epoch [27/50] Loss D: 0.6860, loss G: 0.7161
Epoch [28/50] Loss D: 0.6850, loss G: 0.8273
Epoch [29/50] Loss D: 0.6625, loss G: 0.7561
Epoch [30/50] Loss D: 0.6685, loss G: 0.6814
Epoch [31/50] Loss D: 0.6752, loss G: 0.7202
Epoch [32/50] Loss D: 0.6847, loss G: 0.7957
Epoch [33/50] Loss D: 0.6691, loss G: 0.6949
Epoch [34/50] Loss D: 0.6612, loss G: 0.7641
Epoch [35/50] Loss D: 0.6962, loss G: 0.7501
Epoch [36/50] Loss D: 0.6539, loss G: 0.7959
Epoch [37/50] Loss D: 0.6760, loss G: 0.7035
Epoch [38/50] Loss D: 0.6797, loss G: 0.6699
Epoch [39/50] Loss D: 0.6897, loss G: 0.8323
Epoch [40/50] Loss D: 0.6739, loss G: 0.7861
Epoch [41/50] Loss D: 0.6645, loss G: 0.8020
Epoch [42/50] Loss D: 0.6917, loss G: 0.7506
Epoch [43/50] Loss D: 0.6428, loss G: 0.7642
Epoch [44/50] Loss D: 0.6626, loss G: 0.7577
Epoch [45/50] Loss D: 0.6817, loss G: 0.7809
Epoch [46/50] Loss D: 0.6775, loss G: 0.7401
Epoch [47/50] Loss D: 0.6584, loss G: 0.7153
Epoch [48/50] Loss D: 0.6824, loss G: 0.7855
Epoch [49/50] Loss D: 0.6459, loss G: 0.7783
</pre>

```python
# 원본 이미지와 복원된 이미지 시각화
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))

for images, row in zip([fake_list[-1], real_list[-1]], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.cpu().numpy().squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
```

![gan_conv2d](./gan_conv2d.png)
