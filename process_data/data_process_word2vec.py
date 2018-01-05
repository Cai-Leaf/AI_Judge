from utils import file_pickle as fp
import gensim
import logging

train_word = fp.get_pickle_data('../data_mid/train_data.pkl')
train_word = train_word['word']
test_word = fp.get_pickle_data('../data_mid/test_data.pkl')
law_word = fp.get_pickle_data('../data_mid/law_data.pkl')
word = train_word + test_word + law_word
del train_word, test_word, law_word

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec(word, size=100, window=5, workers=4, min_count=1)
model.save('data_mid/word2vec_100.model')
