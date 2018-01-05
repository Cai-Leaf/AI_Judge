from utils import file_pickle as fp
import gensim
import numpy as np

train_word = fp.get_pickle_data('../data_mid/train_data.pkl')
train_word = train_word['word']
test_word = fp.get_pickle_data('../data_mid/test_data.pkl')
law_word = fp.get_pickle_data('../data_mid/law_data.pkl')
texts = train_word + test_word + law_word
del train_word, test_word, law_word

# 将词打上索引
i = 1
word_index = {}
for text in texts:
    for word in text:
        if word not in word_index:
            word_index[word] = i
            i += 1

# 把索引和向量结合
model = gensim.models.Word2Vec.load('../data_result/word2vec_100.model')
index_vector = {}
for word in word_index.keys():
    index_vector[word_index[word]] = model[word]
print(len(index_vector))

# 从索引为1的词语开始，用词向量填充矩阵
embedding_weights = np.zeros((len(index_vector)+1, 100))
for index, w in index_vector.items():
    embedding_weights[index, :] = w
print(embedding_weights[0:3])

result = {
    'word_index': word_index,
    'index_vector': index_vector,
    'embedding_weights': embedding_weights
}
fp.save_pickle_data('../data_result/word_embedding.pkl', result)

