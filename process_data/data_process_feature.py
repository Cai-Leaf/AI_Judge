from utils import file_pickle as fp
import numpy as np
import random


# # 构造训练集特征——————————————————————————————————————————————————
# train_texts = fp.get_pickle_data('../data_mid/keyword_tfidf_200.pkl')
# print(len(train_texts['train']), len(train_texts['test']))
# train_texts = train_texts['train']
# law_word = fp.get_pickle_data('../data_mid/law_data.pkl')
# law_index = fp.get_pickle_data('../data_mid/train_data.pkl')
# law_index = law_index['law']
# print(len(law_index))
# word_index = fp.get_pickle_data('../data_result/word_embedding.pkl')
# word_index = word_index['word_index']
# # 处理法律条文
# law_vector = []
# max_num = max([len(law) for law in law_word])
# print(max_num)
# for law in law_word:
#     tmp_frature = [0]*max_num
#     for i in range(len(law)):
#         tmp_frature[i] = word_index[law[i]]
#     law_vector.append(np.array(tmp_frature))
#
# # 构建训练集, 对于每个案情，随机选择与其关联的法条和4倍的未关联的法条来构建训练集
# train = []
# mutiple = 4
# law_num = 452
# max_num = max([len(texts) for texts in train_texts])
# # len(train_texts)
# for i in range(len(train_texts)):
#     text_feature = np.zeros(max_num)
#     for j in range(len(train_texts[i])):
#         text_feature[j] = word_index[train_texts[i][j]]
#     # 选择关联的法条构建训练集
#     for j in law_index[i]:
#         train.append([text_feature, law_vector[j - 1], 1])
#     # 选择未关联法条
#     no_law_index = set()
#     while len(no_law_index) < len(law_index[i])*mutiple:
#         tmp_index = random.randint(1, law_num)
#         if tmp_index not in law_index[i]:
#             no_law_index.add(tmp_index)
#     # 使用未关联法条构建训练集
#     for j in no_law_index:
#         train.append([text_feature, law_vector[j - 1], 0])
#
# print(max_num)
# print(len(train))
# fp.save_pickle_data('../data_feature/train_sample3.pkl', train)

# 构造测试集特征——————————————————————————————————————————————————
test_texts = fp.get_pickle_data('../data_mid/keyword_tfidf_200.pkl')
test_texts = test_texts['test']
law_word = fp.get_pickle_data('../data_mid/law_data.pkl')
law_index = fp.get_pickle_data('../data_mid/train_data.pkl')
law_index = law_index['law']
print(len(test_texts), len(law_index))
word_index = fp.get_pickle_data('../data_result/word_embedding.pkl')
word_index = word_index['word_index']

# 处理法律条文
law_vector = []
max_num = max([len(law) for law in law_word])
for law in law_word:
    tmp_frature = [0]*max_num
    for i in range(len(law)):
        tmp_frature[i] = word_index[law[i]]
    law_vector.append(np.array(tmp_frature))

# 处理案情
case_vector = []
mutiple = 4
law_num = 452
max_num = max([len(texts) for texts in test_texts])
print(max_num)
# len(train_texts)
for i in range(len(test_texts)):
    text_feature = np.zeros(max_num)
    for j in range(len(test_texts[i])):
        text_feature[j] = word_index[test_texts[i][j]]
    case_vector.append(text_feature)
print(case_vector[0:5])
result = {
    'law_vector': law_vector,
    'case_vector': case_vector
}
fp.save_pickle_data('../data_feature/test.pkl', result)







