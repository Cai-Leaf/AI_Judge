from utils import file_pickle as fp
from gensim import corpora, models
import logging

train_word = fp.get_pickle_data('../data_mid/train_data.pkl')
train_word = train_word['word']
len_train_word = len(train_word)
test_word = fp.get_pickle_data('../data_mid/test_data.pkl')
texts = train_word + test_word
len_test_word = len(test_word)
print(len_train_word, len_test_word)
print(len(texts))
del train_word, test_word

print(texts[:10])
#
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 文本数据构建成词袋
dictionary = corpora.Dictionary(texts)
# 根据词袋构建文档向量
corpus = [dictionary.doc2bow(text) for text in texts]
# 训练tf-idf模型
tfidf = models.TfidfModel(corpus)
# 获取文档的所有文档的tf-idf
# text_tfidf数据格式: [[(index1, tf-idf1), (index2, tf-idf2), (index3, tf-idf3)], [...], ...]
# text_index数据格式：{index1：word1, index2：word2, index3：word3,}
text_tfidf = tfidf[corpus]
text_index = dict([val, key] for key, val in (dictionary.token2id).items())
save_midle = {
    'text_tfidf': text_tfidf,
    'text_index': text_index
}
#保存模型
fp.save_pickle_data('../data_result/tf_idf_modle.pkl', save_midle)
tf_id_midle = fp.get_pickle_data('../data_result/tf_idf_modle.pkl')
# text_tfidf = tf_id_midle['text_tfidf']
# text_index = tf_id_midle['text_index']
#
# # 找出每个文档中最大的前100个词
# len_train_word = 40000
# word_num = 100
# max_result = []
#
# for doc_tfidf in text_tfidf:
#     tmp_text_word = []
#     sorted_list = sorted(doc_tfidf, key=lambda d: d[1], reverse=True)
#     for key, value in sorted_list[0:word_num]:
#         tmp_text_word.append(text_index[key])
#     max_result.append(tmp_text_word)
#
# # top_50_word_result = {
# #     'train': max_result[0:len_train_word],
# #     'test': max_result[len_train_word:]
# # }
# # fp.save_pickle_data('../data_result/top_50_word_result.pkl', top_50_word_result)
#
# for data in max_result:
#     if len(data) < 100:
#         print(data)
