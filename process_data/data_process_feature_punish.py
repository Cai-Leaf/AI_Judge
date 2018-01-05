from utils import file_pickle as fp
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import codecs
import re

# 构造训练集的特征
# train_texts = fp.get_pickle_data('../data_mid/keyword_tfidf_200.pkl')
# train_texts = train_texts['train']
# data = fp.get_pickle_data('../data_mid/train_data.pkl')
# law_index = data['law']
# punish_class = data['panish_class']
# del data
# print(len(law_index), len(punish_class), len(train_texts))
# word_index = fp.get_pickle_data('../data_result/word_embedding.pkl')
# word_index = word_index['word_index']
#
# # 构造标签数据
# enc = OneHotEncoder().fit([[i] for i in set(punish_class)])
# train_y = enc.transform([[i] for i in punish_class]).toarray()
# print(len(train_y))
#
#
# train_x = []
# mutiple = 4
# law_num = 452
# max_num = max([len(texts) for texts in train_texts])
# for i in range(len(train_texts)):
#     text_feature = np.zeros(max_num)
#     for j in range(len(train_texts[i])):
#         text_feature[j] = word_index[train_texts[i][j]]
#     law_vec = np.zeros(law_num)
#     for j in law_index[i]:
#         law_vec[j-1] = 1
#     train_x.append([text_feature, law_vec])
#
# print(len(train_x))
#
# train = {
#     'train_x': train_x,
#     'train_y': train_y
# }
#
# fp.save_pickle_data('../data_feature/train_punish.pkl', train)

# 构造训练集 无标签，只有文本
# train_texts = fp.get_pickle_data('../data_mid/keyword_tfidf_200_clean.pkl')
# train_texts = train_texts['train']
# punish_class = fp.get_pickle_data('../data_mid/train_data.pkl')['panish_class']
# print(len(punish_class), len(train_texts))
# word_index = fp.get_pickle_data('../data_result/word_embedding.pkl')
# word_index = word_index['word_index']
#
# # 构造标签数据
# enc = OneHotEncoder().fit([[i] for i in set(punish_class)])
# train_y = enc.transform([[i] for i in punish_class]).toarray()
# print(len(train_y))
#
#
# train_x = []
# mutiple = 4
# law_num = 452
# max_num = max([len(texts) for texts in train_texts])
# print(max_num)
# for i in range(len(train_texts)):
#     text_feature = np.zeros(max_num)
#     for j in range(len(train_texts[i])):
#         text_feature[j] = word_index[train_texts[i][j]]
#     train_x.append(text_feature)
# print(len(train_x))
# train = {
#     'train_x': train_x,
#     'train_y': train_y
# }
#
# fp.save_pickle_data('../data_feature/train_punish2.pkl', train)

# 构造测试集的特征
test_texts = fp.get_pickle_data('../data_mid/keyword_tfidf_200_clean.pkl')
test_texts = test_texts['test']
word_index = fp.get_pickle_data('../data_result/word_embedding.pkl')
word_index = word_index['word_index']

test = []
max_num = max([len(texts) for texts in test_texts])
print(max_num)
for i in range(len(test_texts)):
    text_feature = np.zeros(max_num)
    for j in range(len(test_texts[i])):
        text_feature[j] = word_index[test_texts[i][j]]
    test.append(text_feature)

file = codecs.open("../data_origin/test.txt", "rb", "utf-8")
test_id = []
for line in file:
    tmp = re.split('\t|\n', line)
    test_id.append(tmp[0])

file.close()


print(len(test_id), len(test))
print(test_id[0], test[0])
test = {
    'id': test_id,
    'value': test
}
fp.save_pickle_data('../data_feature/test_punish2.pkl', test)
