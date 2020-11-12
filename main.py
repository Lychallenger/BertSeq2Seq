from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np

config_path = './data/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './data/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './data/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')

print('\n ===== predicting =====\n')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))