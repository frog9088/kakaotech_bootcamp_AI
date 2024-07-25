---

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

(동일 라이브러리, 데이터셋을 사용한다)
```python
def add_space_before_replace(text, chars):
    for char in chars:
        text = text.replace(char, f' {char} ')
    return text


chars_to_replace = ['\n', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?']

processed_text = add_space_before_replace(text, chars_to_replace)
```
a-zA-Z를 제외한 특수문자 앞뒤로 공백을 추가한다
(단어와 특수문자를 분리시키기 위함)
```python
words = sorted(list(set(processed_text.lower().split())))
len(words)
```

<pre>
11466
</pre>

```python
words[:100]
```

<pre>
['!',
 '$',
 '&',
 "'",
 ',',
 '-',
 '.',
 '3',
 ':',
 ';',
 '?',
 'a',
 'abandon',
 'abase',
 'abate',
 'abated',
 ...]
</pre>

```python
words_to_idx = {words: idx for idx, words in enumerate(words)}
idx_to_words = {idx: words for idx, words in enumerate(words)}
```


```python
# 시퀀스 데이터 생성 함수 정의
def create_sequences(text, seq_length):
    sequences = []
    targets = []
    for i in range(0, len(text) - seq_length):
        seq = text[i:i+seq_length]   # 시퀀스 생성
        target = text[i+seq_length]  # 시퀀스 다음에 오는 문자
        sequences.append([words_to_idx[char] for char in seq])
        targets.append(words_to_idx[target])
    return sequences, targets
```


```python
# 시퀀스 길이 설정
seq_length = 20

# 시퀀스 데이터 생성
sequences, targets = create_sequences(processed_text.lower().split(), seq_length)
```


```python
print(sequences[:10])
```

<pre>
[[3820, 1759, 8, 841, 11061, 7619, 413, 4177, 4, 4716, 6150, 9240, 6, 291, 8, 9240, 4, 9240, 6, 3820], [1759, 8, 841, 11061, 7619, 413, 4177, 4, 4716, 6150, 9240, 6, 291, 8, 9240, 4, 9240, 6, 3820, 1759], [8, 841, 11061, 7619, 413, 4177, 4, 4716, 6150, 9240, 6, 291, 8, 9240, 4, 9240, 6, 3820, 1759, 8], [841, 11061, 7619, 413, 4177, 4, 4716, 6150, 9240, 6, 291, 8, 9240, 4, 9240, 6, 3820, 1759, 8, 11448], [11061, 7619, 413, 4177, 4, 4716, 6150, 9240, 6, 291, 8, 9240, 4, 9240, 6, 3820, 1759, 8, 11448, 487], [7619, 413, 4177, 4, 4716, 6150, 9240, 6, 291, 8, 9240, 4, 9240, 6, 3820, 1759, 8, 11448, 487, 291], [413, 4177, 4, 4716, 6150, 9240, 6, 291, 8, 9240, 4, 9240, 6, 3820, 1759, 8, 11448, 487, 291, 8201], [4177, 4, 4716, 6150, 9240, 6, 291, 8, 9240, 4, 9240, 6, 3820, 1759, 8, 11448, 487, 291, 8201, 7935], [4, 4716, 6150, 9240, 6, 291, 8, 9240, 4, 9240, 6, 3820, 1759, 8, 11448, 487, 291, 8201, 7935, 10153], [4716, 6150, 9240, 6, 291, 8, 9240, 4, 9240, 6, 3820, 1759, 8, 11448, 487, 291, 8201, 7935, 10153, 2776]]
</pre>

```python
# PyTorch 데이터셋 및 데이터로더 생성
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])
```


```python
# 데이터셋 및 데이터로더 인스턴스 생성
dataset = TextDataset(sequences, targets)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
```


```python
# 하이퍼파라미터 설정
vocab_size = len(words)
hidden_size = 256
output_size = len(words)
input_size = 2048
num_layers = 2
```


```python

class RNNModel(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, input_size)  # 임베딩 층
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # RNN 레이어
        self.fc = nn.Linear(hidden_size, vocab_size)  # 완전 연결층

    def forward(self, x, hidden):
        # x: (batch_size, seq_length)
        x = self.embeddings(x)  # 임베딩 층 통과 -> (batch_size, seq_length, input_size)
        out, hidden = self.rnn(x, hidden)  # RNN 순전파
        out = self.fc(out[:, -1, :])  # 마지막 시퀀스 출력만 사용 -> (batch_size, output_size)
        return out, hidden

    def init_hidden(self, batch_size):
        # 초기 hidden state 설정
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
```


```python
# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 인스턴스 생성 및 GPU로 이동
model = RNNModel(vocab_size, hidden_size, output_size, num_layers).to(device)
```


```python
# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```


```python
# 모델 훈련 함수
def train_model(model, dataloader, criterion, optimizer, num_epochs=20):
    model.train()  # 모델을 훈련 모드로 설정
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            #inputs = nn.functional.one_hot(inputs, num_classes=vocab_size).float() # 원-핫 인코딩 및 GPU로 이동
            inputs = inputs.to(device)
            labels = labels.to(device)

            hidden = model.init_hidden(inputs.size(0))  # 각 배치마다 hidden 상태 초기화

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 순전파, 역전파, 최적화
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # hidden 상태를 detach하여 그래프의 연결을 끊음
            hidden = hidden.detach()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}')

    print('Finished Training')
```


```python
def generate_text(model, start_str, length, temperature=1.0):
    model.eval()  # 모델을 평가 모드로 설정
    hidden = model.init_hidden(1)  # 초기 hidden 상태 설정

    input_seq = torch.tensor([words_to_idx[char] for char in start_str]).unsqueeze(0).to(device)
    generated_text = start_str

    with torch.no_grad():
        for _ in range(length):
            input_seq = nn.functional.one_hot(input_seq, num_classes=vocab_size).float()
            output, hidden = model(input_seq, hidden)

            # 다음 문자를 샘플링
            output = output.squeeze().div(temperature).exp().cpu()
            top_char = torch.multinomial(output, 1)[0]

            generated_char = idx_to_words[top_char.item()]
            generated_text += generated_char

            # 다음 입력 시퀀스 준비
            input_seq = torch.tensor([[top_char]]).to(device)

    return generated_text
```


```python
# 모델 훈련
train_model(model, dataloader, criterion, optimizer, num_epochs=5)
```

<pre>
Epoch 1/5, Loss: 64.4105
Epoch 2/5, Loss: 58.2193
Epoch 3/5, Loss: 53.2220
Epoch 4/5, Loss: 48.3512
</pre>

```python
# 테스트 시작 문자열 및 생성할 텍스트 길이
start_str = "To be, or not to be, that is the question:"
length = 100
```


```python
# 텍스트 생성
generated_text = generate_text(model, start_str, length, temperature=0.8)
print(generated_text)
```

### 결과
결과적으로 학습에도 너무 큰 시간이 소요되고 데이터셋을 전처리하는 과정에서 입력은 단어 token이지만 출력은 character인 오류가 있었다. 또한 ' '공백이 split과정에서 사라져 출력이 연속된 character로 나오게 되는 문제가 있었다. 