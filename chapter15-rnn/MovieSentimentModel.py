# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/3 10:54
# Description: 电影情感分析
import torch
from torch import nn
from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split
import re
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torch.utils.data import DataLoader

train_dataset = IMDB(root='../data/movie_data.txt', split="train")
test_dataset = IMDB(root='../data/movie_data.txt', split="test")

# 把训练数据拆分为训练子集20000和验证子集5000
torch.manual_seed(1)
train_dataset, valid_dataset = random_split(list(train_dataset), [20000, 5000])


# 找出训练子集中独特的单词
def tokenizer(text):
    # 删除所有的html标签
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    # 删除文本中所有非单词字符 并且大写转小写
    text = (re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace('-', ''))
    tokenized = text.split()
    return tokenized


# 统计出文章所有单词的以及单词对应的个数
token_counts = Counter()
for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)

# print("Vocab-size:", len(token_counts), "\n heap 10:\n", token_counts.keys())

# 把每个独特的单词映射成一个唯一的数字，并将影评转化为一串数字
sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocab = vocab(ordered_dict)
vocab.insert_token("<pad>", 0)
vocab.insert_token("<unk>", 1)
vocab.set_default_index(1)

# 定义处理文本的函数
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1 if x == 'pos' else 0


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        # 将评论的标签记录下
        label_list.append(label_pipeline(_label))
        # 将评论处理加工成数字
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        # 记录下当前评论的长度
        lengths.append(processed_text.size(0))

    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return padded_text_list, label_list, lengths


# 一批32个数据
batch_size = 32
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_batch)
valid_dl = DataLoader(valid_dataset, batch_size, shuffle=False, collate_fn=collate_batch)
test_dl = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_batch)


# 一种降低词元向量维度的方法，叫做特征嵌入
# 由于词元数据较多，导致向量维度较大，非常不便于计算，因此采用特征嵌入的方式 使长度远小于词元的类别数
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        # 就是将词元数量缩减为embed_dim 且特征不受影响
        # padding_idx=0 通常用于指定一个特定的索引值，该值用于填充那些长度较短的序列，使它们与最长序列具有相同的长度，从而方便批处理
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 单层测LSTM
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pad_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64
model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(dataloader):
    model.train()
    total_acc, total_loss = 0, 0
    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += (
                (pred >= 0.5).float() == label_batch
        ).float().sum().item()

        total_loss += loss.item() * label_batch.size(0)

    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)


# 负责使用模型验证数据
def evaluate(dataloader):
    model.eval()
    total_acc, total_loss = 0, 0
    for text_batch, label_batch, lengths in dataloader:
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        total_acc += (
                (pred >= 0.5).float() == label_batch
        ).float().sum().item()

        total_loss += loss.item() * label_batch.size(0)

    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)


num_epochs = 10
for epoch in range(num_epochs):
    acc_train, loss_train = train(train_dl)
    acc_valid, loss_valid = evaluate(valid_dl)
    print(f"Epoch {epoch + 1} accuracy: {acc_train:.4f} valid_acc: {acc_valid:.4f}")
