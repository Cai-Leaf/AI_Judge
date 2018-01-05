from utils import file_pickle as fp
import re

# 提取关键词
# train_word = fp.get_pickle_data('../data_mid/train_data.pkl')
# train_word = train_word['word']
# len_train_word = len(train_word)
# test_word = fp.get_pickle_data('../data_mid/test_data.pkl')
# law_word = fp.get_pickle_data('../data_mid/law_data.pkl')
# texts = train_word + test_word
# del train_word, test_word
#
# law_str = ''
# for law in law_word:
#     for word in law:
#         law_str += word
#
# keyword = []
# for text in texts:
#     tmp_set = set()
#     for word in text:
#         for char in word:
#             if char in law_str:
#                 tmp_set.add(word)
#                 break
#     keyword.append(list(tmp_set))
#
# for i in range(10):
#     print(keyword[i])
# print(max([len(word) for word in keyword]))
# leyword_result = {
#     'train': keyword[0:len_train_word],
#     'test': keyword[len_train_word:]
# }
# fp.save_pickle_data('../data_mid/keyword.pkl', leyword_result)

# 提取tfidf大于阈值且在keyword中的词
key_word = fp.get_pickle_data('../data_mid/keyword.pkl')
train_len = len(key_word['train'])
key_word = key_word['train'] + key_word['test']
tf_id_midle = fp.get_pickle_data('../data_result/tf_idf_modle.pkl')
text_tfidf = tf_id_midle['text_tfidf']
print(len(text_tfidf))
index_word = tf_id_midle['text_index']
word_index = {}
for key in index_word.keys():
    word_index[index_word[key]] = key

# 提取在keyword中tfidf排名前100的词和tfidf排名前100的词，按照在原文中的相对位置排序
key_word_num = 100
word_num = 200
keyword_tfidf_result = []

for i in range(len(text_tfidf)):
    key_tfidf_rank = []
    # 将keyword中的词按照tfidf排序
    for word in key_word[i]:
        for tfidf in text_tfidf[i]:
            if word_index[word] == tfidf[0]:
                key_tfidf_rank.append(tfidf)
                break
    key_tfidf_rank = sorted(key_tfidf_rank, key=lambda d: d[1], reverse=True)
    # 将排好序的keyword中的前100个词加入到key_sorted_list
    key_sorted_list = set()
    j = 0
    contain_number = re.compile('^.*\\d+.*')
    while len(key_sorted_list) < key_word_num and j < len(key_tfidf_rank):
        word = index_word[key_tfidf_rank[j][0]]
        if len(word) > 1 and not contain_number.match(word):
            key_sorted_list.add(key_tfidf_rank[j])
        j += 1
    # 将keyword中的词加到结果里
    tmp_word_set = set()
    for key, value in key_sorted_list:
        tmp_word_set.add(key)
    j = 0
    # 将其他词按照tfidf排序并加入到结果
    sorted_list = sorted(text_tfidf[i], key=lambda d: d[1], reverse=True)

    while len(tmp_word_set) < word_num and j < len(sorted_list):
        word = index_word[sorted_list[j][0]]
        if len(word) > 1 and not contain_number.match(word):
            tmp_word_set.add(sorted_list[j][0])
        j += 1
    result = []
    for key, value in text_tfidf[i]:
        if key in tmp_word_set:
            result.append(index_word[key])
    keyword_tfidf_result.append(result)
    print(i, len(result), result)
    print('-------------------------------------------------------------------------------------------------------')

print(len(keyword_tfidf_result))
print(len(key_word))

keyword_tfidf_result = {
    'train': keyword_tfidf_result[:train_len],
    'test': keyword_tfidf_result[train_len:],
}

fp.save_pickle_data('../data_mid/keyword_tfidf_200_clean.pkl', keyword_tfidf_result)





