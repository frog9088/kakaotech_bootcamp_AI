


(동일 라이브러리, 데이터셋을 사용한다)
### 데이터 전처리

```python
def add_space_before_replace(text, chars):
    for char in chars:
        text = text.replace(char, f' {char} ')
    return text


chars_to_replace = ['\n', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?']

processed_text = add_space_before_replace(text, chars_to_replace)
```
```python
#sorted(words_dict.keys())
word_dict = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)

words = sorted(list(set(processed_text.lower().split())))

words_list = set(processed_text.lower().split())

words = [word for word in words if len(word) > 10]

words_dict = {}
for word in processed_text.lower().split():
    if word not in words_dict:
        words_dict[word] = 1
    else:
        words_dict[word] += 1

word_dict = [word[0] for word in word_dict if word[1] > 10]
len(word_dict)
```

<pre>
1731
</pre>

```python
words_to_idx = {words: idx for idx, words in enumerate(word_dict)}
idx_to_words = {idx: words for idx, words in enumerate(word_dict)}
```
```python
words_to_idx['UNK'] = len(word_dict)
idx_to_words[len(word_dict)] = 'UNK'
```
등장횟수가 10보다 큰 단어들에 대해서 단어 사전 구축, 
단어 사전에 없는 단어들은 ['UNK']토큰으로 처리한다.
단어 사전 내 단어 개수 11465 -> 1731

```python
# 시퀀스 데이터 생성 함수 정의
def create_sequences(text, seq_length):
    sequences = []
    targets = []
    for i in range(0, len(text) - seq_length):
        seq = text[i:i+seq_length]   # 시퀀스 생성
        target = text[i+seq_length]  # 시퀀스 다음에 오는 문자
        sequences.append([words_to_idx[char] if char in words_to_idx else words_to_idx['UNK'] for char in seq ])
        targets.append(words_to_idx[target] if target in words_to_idx else words_to_idx['UNK'])
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
[[100, 283, 1, 152, 40, 985, 158, 678, 0, 137, 24, 114, 2, 43, 1, 114, 0, 114, 2, 100], [283, 1, 152, 40, 985, 158, 678, 0, 137, 24, 114, 2, 43, 1, 114, 0, 114, 2, 100, 283], [1, 152, 40, 985, 158, 678, 0, 137, 24, 114, 2, 43, 1, 114, 0, 114, 2, 100, 283, 1], [152, 40, 985, 158, 678, 0, 137, 24, 114, 2, 43, 1, 114, 0, 114, 2, 100, 283, 1, 10], [40, 985, 158, 678, 0, 137, 24, 114, 2, 43, 1, 114, 0, 114, 2, 100, 283, 1, 10, 50], [985, 158, 678, 0, 137, 24, 114, 2, 43, 1, 114, 0, 114, 2, 100, 283, 1, 10, 50, 43], [158, 678, 0, 137, 24, 114, 2, 43, 1, 114, 0, 114, 2, 100, 283, 1, 10, 50, 43, 1274], [678, 0, 137, 24, 114, 2, 43, 1, 114, 0, 114, 2, 100, 283, 1, 10, 50, 43, 1274, 363], [0, 137, 24, 114, 2, 43, 1, 114, 0, 114, 2, 100, 283, 1, 10, 50, 43, 1274, 363, 7], [137, 24, 114, 2, 43, 1, 114, 0, 114, 2, 100, 283, 1, 10, 50, 43, 1274, 363, 7, 213]]
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
vocab_size = len(word_dict) + 1
hidden_size = 256
output_size = len(word_dict) + 1
input_size = 2048
num_layers = 2
```

### 모델 생성
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
# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 인스턴스 생성 및 GPU로 이동
model = RNNModel(vocab_size, hidden_size, output_size, num_layers).to(device)
```
RNN모델을 사용하였다.

```python
# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 모델 훈련 함수
```python
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

### 텍스트 생성 함수
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
train_model(model, dataloader, criterion, optimizer, num_epochs=5)
```
<pre>
Epoch 1/5, Loss: 7.1866
Epoch 2/5, Loss: 6.8305
Epoch 3/5, Loss: 6.5715
Epoch 4/5, Loss: 6.5109
Epoch 5/5, Loss: 6.5289
Finished Training
</pre>

```python
# 테스트 시작 문자열 및 생성할 텍스트 길이
start_str = "To be, or not to be, that is the question:"
length = 100

# 텍스트 생성
generated_text = generate_text(model, start_str, length, temperature=0.8)
print(generated_text)
```

<pre>
To be, or not to be, that is the question:usanywithqueenhe,the?firepoorlady?
daysetmenwithlikeme,left??,UNK,himwrongallyouveryUNKatis,withi,UNKand?
UNKlikemustfriarthouandi',,?
must:twotoUNKUNK,fatherbeUNKsetisetlady.thusillandthevalour,inyou?
theangeloenterstands,thouUNK,queen,ofttrickbanishshallat:,aatoenterlikehimthe
</pre>

### 결과
학습 시간은 대폭 감소했으나 역시 동일하게 데이터셋을 전처리하는 과정에서 입력은 단어 token이지만 출력은 character인 오류가 있었다. 또한 ' '공백이 split과정에서 사라져 출력이 연속된 character로 나오게 되는 문제가 있었다. 
