# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/1 15:27
# Description: 在Pytorch中构建字符级语言建模
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

with open("../data/1268-0.txt", 'r', encoding='utf8') as fp:
    text = fp.read()

start_idx = text.find("THE MYSTERIOUS ISLAND")
end_idx = text.find("End of the Project Gutenberg")
text = text[start_idx: end_idx]
char_set = set(text)
print("Total Length:", len(text), "Unique Characters:", len(char_set))
char_sorted = sorted(char_set)
char2int = {ch: i for i, ch in enumerate(char_sorted)}
# 将文本中的所有字符转换为数字
char_array = np.array(char_sorted)
text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)
print("Text encoded shape:", text_encoded.shape)
print(text[:15], "Encode to", text_encoded[:15])
print(text_encoded[15:21], "Reserve to", ''.join(char_array[text_encoded[15:21]]))

# 序列长度暂时设定为40，即张量是由40个词元构成，较短的序列可能专注于正确的预测每个单词，而忽略上下文信息，
# 较长的序列会影响文本生成的质量
# 每一组数组是由41个数字组成
seq_length = 40
chunk_size = seq_length + 1
# 即将文本构建成 含有len(text) - chunk_size + 1组样本的 41长度的序列
text_chunks = [text_encoded[i:i + chunk_size] for i in range(len(text_encoded) - chunk_size + 1)]


# 将41长度的序列分为前40个和后40个 合并为二元组
class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()


# 转换为可以训练的数据
# [samples, [features, labels]]
seq_dataset = TextDataset(torch.tensor(np.array(text_chunks)))
batch_size = 64
torch.manual_seed(1)
seq_dl = DataLoader(seq_dataset, batch_size, shuffle=True, drop_last=True)


# 构建循环神经网络
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell


# 统计词元个数
vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512
torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size)

# 定义loss
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_epochs = 10000
torch.manual_seed(1)
for epoch in range(num_epochs):
    hidden, cell = model.init_hidden(batch_size=batch_size)
    seq_batch, target_batch = next(iter(seq_dl))
    loss = 0
    optimizer.zero_grad()
    for c in range(seq_length):
        pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
        loss += loss_fn(pred, target_batch[:, c])
    loss.backward()
    optimizer.step()
    loss = loss.item() / seq_length
    if epoch % 500 == 0:
        print(f"Epoch {epoch} loss: {loss:.4f}")


def sample(model, start_str, len_generated_text=500, scale_factor=1.0):
    # 将输入的起始文本转化为数字
    encode_input = torch.tensor([char2int[ch] for ch in start_str])
    encode_input = torch.reshape(encode_input, (1, -1))
    generated_str = start_str
    model.eval()

    hidden, cell = model.init_hidden(1)
    for c in range(len(start_str) - 1):
        _, hidden, cell = model(encode_input[:, c].view(1), hidden, cell)

    last_char = encode_input[:, -1]
    for i in range(len_generated_text):
        logit, hidden, cell = model(last_char.view(1), hidden, cell)
        logit = torch.squeeze(logit, 0)
        scale_logit = logit * scale_factor
        m = Categorical(logits=scale_logit)
        last_char = m.sample()
        generated_str += str(char_array[last_char])

    return generated_str


print(sample(model, start_str="The island"))
