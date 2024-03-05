# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/5 10:32
# Description:
import time

import numpy as np
from transformers import Trainer, TrainingArguments
import pandas as pd
import torch.utils.data as data
import torch
from datasets import load_metric

# 通过使用 DistilBertTokenizerFast，你可以轻松地准备文本数据以供 DistilBERT 模型使用，
# 从而进行各种自然语言处理任务，如文本分类、实体识别、问答等。这个工具对于构建和训练基于 DistilBERT 的模型非常重要，它确保了输入数据的格式与模型期望的格式一致。
from transformers import DistilBertTokenizerFast

# 当你有一个文本分类任务时，你可以使用 DistilBertForSequenceClassification 模型。
#       你需要首先通过 DistilBertTokenizerFast 将文本转换为模型可以理解的数字 ID 序列。然后，这个序列会被输入到 DistilBERT 模型中，
#       经过多层的 Transformer 编码器结构进行特征提取，生成一个固定长度的向量表示。最后，这个向量会被输入到分类层中，得到文本所属类别的概率分布。
# DistilBertForSequenceClassification 模型在训练过程中会学习如何根据输入的文本生成准确的分类结果。
# 你可以使用带有标签的数据集来训练这个模型，通过最小化预测标签与实际标签之间的差异来优化模型的参数。一旦模型训练完成，你就可以使用它来对新的文本进行分类了
from transformers import DistilBertForSequenceClassification

torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

num_epoch = 3

df = pd.read_csv("../data/movie_data.csv")


def load_dataset(df, tokenizer, start_index=None, end_index=None):
    if start_index is None:
        # 如果是训练集，从开头取到指定的结束索引
        texts = df['review'].iloc[:end_index].values
        labels = df['sentiment'].iloc[:end_index].values
    else:
        # 如果是验证集或测试集，从指定的起始索引取到结束索引
        texts = df['review'].iloc[start_index:end_index].values
        labels = df['sentiment'].iloc[start_index:end_index].values

    # encodings keys: {'input_ids', 'attention_mask'}
    encodings = tokenizer(list(texts), truncation=True, padding=True)
    datasets = IMDBDataset(encodings, labels)
    return datasets


def load_data(df, tokenizer, batch_size, start_index=None, end_index=None):
    datasets = load_dataset(df, tokenizer, start_index, end_index)
    return data.DataLoader(datasets, batch_size=batch_size, shuffle=True)


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


batch_size = 16
# 数据集分词
# distilbert: 这是指 DistilBERT 模型，它是一个轻量级的 BERT 模型，通过知识蒸馏技术从原始的 BERT 模型中提炼出来。DistilBERT 在保持与 BERT 相似的性能的同时，减少了模型的大小和计算需求。
# base: 这指的是模型的大小或配置。DistilBERT 有几种不同的预训练配置，如 base、large 等。base 配置通常比 large 配置更小、更快，但可能在某些任务上的性能稍逊一筹。
# uncased: 这指的是模型的词汇表是否区分大小写。uncased 表示模型在处理文本时不区分大小写，即它会将大写和小写字母视为相同。
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_dataloader = load_data(df, tokenizer, batch_size, end_index=35000)
valid_dataloader = load_data(df, tokenizer, batch_size, start_index=35000, end_index=40000)
test_dataloader = load_data(df, tokenizer, batch_size, start_index=40000)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

optim = torch.optim.Adam(model.parameters(), lr=5e-5)


def compute_accuracy(model, data_loader):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            input_idx = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch["labels"]

            outputs = model(input_idx, attention_mask=attention_mask)
            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
        return correct_pred.float() / num_examples * 100


def bert_train():
    start_time = time.time()

    for epoch in range(num_epoch):
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            input_idx = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch["labels"]

            outputs = model(input_idx, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs['loss'], outputs["logits"]
            optim.zero_grad()
            loss.backward()
            optim.step()

            if not batch_idx % 250:
                print(f"Epoch: {epoch + 1:04d} / {num_epoch:04d} | Batch {batch_idx:04d}/{len(train_dataloader):04d} "
                      f"| Loss: {loss:.4f}")
        model.eval()
        with torch.set_grad_enabled(False):
            print(
                f"Training accuracy: {compute_accuracy(model, train_dataloader):.2f}%\n Valid Accuracy: {compute_accuracy(model, valid_dataloader):.2f}%")

        print(f'Time Elapsed: {(time.time() - start_time) / 60:.2f} min')

    print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')
    print(f"Test Accuracy: {compute_accuracy(model, test_dataloader):.2f}%")


# 使用trainer api进行测试
def trainer_api_train():
    train_dataset = load_dataset(df, tokenizer, end_index=35000)
    valid_dataset = load_dataset(df, tokenizer, start_index=35000, end_index=40000)
    test_dataset = load_dataset(df, tokenizer, start_index=40000)
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predications = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predications, references=labels)

    train_args = TrainingArguments(output_dir='./results', num_train_epochs=3,
                                   per_device_train_batch_size=16,
                                   per_device_eval_batch_size=16,
                                   logging_dir='./logs',
                                   logging_steps=10)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optim, None)
    )

    trainer.train()
