# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/5 10:10
# Description:
from transformers import pipeline, set_seed

generator = pipeline("text-generation", model='gpt2')
set_seed(123)

# 使用gpt2文本生成
print(generator("Hello, students, Today Let's study", max_length=30, truncation=True, num_return_sequences=3,
                pad_token_id=0))
