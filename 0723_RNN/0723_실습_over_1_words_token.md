


(동일 라이브러리, 데이터셋을 사용한다)
### 데이터 전처리
```python
def add_space_before_replace(text, chars):
    for char in chars:
        text = text.replace(char, f' {char} ')
    return text

chars_to_replace = ['\n', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?']
processed_text = add_space_before_replace(text, chars_to_replace)

words_dict = {}

for word in processed_text.lower().split():
    if word not in words_dict:
        words_dict[word] = 1
    else:
        words_dict[word] += 1
        
word_dict = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)
len(word_dict)
```
<pre>
11466
</pre>

```python
word_dict = [word[0] for word in word_dict if word[1] > 1]
len(word_dict)
```
<pre>
6547
</pre>

등장 횟수가 1을 넘는 단어들에 대해서 단어 사전을 구축한다.

```python
words_to_idx = {words: idx for idx, words in enumerate(word_dict)}
idx_to_words = {idx: words for idx, words in enumerate(word_dict)}
words_to_idx['UNK'] = len(word_dict)
idx_to_words[len(word_dict)] = 'UNK'
words_to_idx[' '] = len(word_dict)+1
idx_to_words[len(word_dict)+1] = ' '
```
단어 사전에 없는 단어들에 대해서는 ['UNK']토큰을 사용한다.
공백 토큰도 추가해서 넣어준다. (split에서 사라지므로)

```python
# 시퀀스 데이터 생성 함수 정의
def create_sequences(text, seq_length):
    sequences = []
    targets = []
    for i in range(0, len(text) - seq_length):
        seq = text[i:i+seq_length]   # 시퀀스 생성
        target = text[i+seq_length]  # 시퀀스 다음에 오는 문자
        sequences.append([words_to_idx[char] if char in words_to_idx else 
        words_to_idx['UNK'] for char in seq ])
        targets.append(words_to_idx[target] if target in words_to_idx else 
        words_to_idx['UNK'])
    return sequences, targets
```

```python
sentence = text.lower()
result = sentence.split()
result_with_spaces = []
for word in result:
    result_with_spaces.extend([word])
    result_with_spaces.append(' ')
result_with_spaces.pop() 
```
소문자로 변형, 공백 기준으로 split, 이후 각 단어 사이에 공백을 추가.
```python
sequences, targets = create_sequences(result_with_spaces, 20)
```
전처리한 text에 대해서 시퀀스 생성

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

dataset = TextDataset(sequences, targets)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
```


```python
# 하이퍼파라미터 설정
vocab_size = len(word_dict) + 2 #['UNK'], [' ']을 추가해서 +2
hidden_size = 1024
output_size = len(word_dict) + 2
input_size = 2048
num_layers = 2

hidden_size_1 = 1024
hidden_size_2 = 2048
```


### 모델 생성
```python

class RNNModel(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size_1, hidden_size_2, 
    num_layers=1):
        super(RNNModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, input_size)  # 임베딩 층
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size_1, num_layers, 
        batch_first=True)  # RNN 레이어
        self.fc1 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc2 = nn.Linear(hidden_size_2, vocab_size)  # 완전 연결층
        self.RelU = nn.ReLU()

    def forward(self, x, hidden):
        # x: (batch_size, seq_length)
        x = self.embeddings(x)  # 임베딩 층 통과 -> (batch_size, seq_length, input_size)
        out, hidden = self.rnn(x, hidden)  # RNN 순전파
        x = self.fc1(out[:, -1, :])
        x = self.RelU(x)
        out = self.fc2(x)
        # 마지막 시퀀스 출력만 사용 -> (batch_size, output_size)
        return out, hidden

    def init_hidden(self, batch_size):
        # 초기 hidden state 설정
        return torch.zeros(self.num_layers, batch_size, 
        self.hidden_size_1).to(device)
```
임베딩 레이어를 추가하여 단어 토큰을 임베딩시켰다.

```python
# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 모델 인스턴스 생성 및 GPU로 이동
model = RNNModel(vocab_size, hidden_size, output_size, num_layers).to(device)
# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 모델 훈련
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

### 텍스트 생성
```python
def generate_text(model, start_str, length, temperature=1.0):
    model.eval()  # 모델을 평가 모드로 설정
    hidden = model.init_hidden(1)  # 초기 hidden 상태 설정

    input_seq = torch.tensor([words_to_idx[char] if char in words_to_idx else words_to_idx['UNK'] for char in start_str]).unsqueeze(0).to(device)
    generated_text = start_str

    with torch.no_grad():
        for _ in range(length):
            #input_seq = nn.functional.one_hot(input_seq, num_classes=vocab_size).float()
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
train_model(model, dataloader, criterion, optimizer, num_epochs=3)
```

<pre>
Epoch 1/3, Loss: 5.4141
Epoch 2/3, Loss: 3.2415
Epoch 3/3, Loss: 3.1897
Finished Training
</pre>

### 결과
```python
# 테스트 시작 문자열 및 생성할 텍스트 길이
start_str = "To be, or not to be, that is the question:".lower()
length = 100
# 텍스트 생성
generated_text = generate_text(model, start_str, length, temperature=0.8)
print(generated_text)
```
<pre>
to be, or not to be, that is the question:        UNK UNK     UNKand               UNK 
UNKUNKloveUNK       UNK        UNK    UNK you UNK       UNK      of UNK 
their UNK  UNK UNK   UNKmust UNK
</pre>

(해당 결과는 단어 등장 횟수 thresholds = 1)

```python
# 텍스트 생성
generated_text = generate_text(model, start_str, length, temperature=0.8)
print(generated_text)
```
<pre>
To be, or not to be, that is the question:a again are UNK UNK simple 
UNK UNK UNK UNK UNKUNK alack you UNK that he 
UNK alack UNK UNK cannot UNK UNK UNK UNK he UNK look 
and i and he he fortune UNK UNK been UNK had UNK been alack 
UNK some the UNK the UNK again 
</pre>

(해당 결과는 단어 등장 횟수 thresholds = 10)


> 생성 결과에 ['UNK']토큰이 너무 많이 생성되었는데, 총 단어의 절반이상을 ['UNK']로 설정하여 해당 현상이 일어난 것으로 보인다. 보다 나은 토크나이징을 통해 더 좋은 생성 결과를 얻을 수 있을 것으로 추측한다. 