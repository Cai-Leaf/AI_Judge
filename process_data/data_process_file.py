import codecs
import re
import jieba
from utils import file_pickle as fp

file = codecs.open("data_origin/train.txt", "rb", "utf-8")
word = []
panish_class = []
law = []
stop_word_list = [' ', '', '。', '，', '-', '：', '“', '”', '"', '‘', '’', '！', '（', '）', '~', '、', ',', '.',
                  '(', ')', ';', '；', ':', '？', '?',  '～', '《', '》', '<', '>', '──', '─', '…', '……', '[', ']',
                  '【', '】', '~', ']', ']']
for line in file:
    tmp = re.split('\t|\n', line)
    word.append([word for word in jieba.cut(tmp[1]) if word not in stop_word_list])
    panish_class.append(int(tmp[2]))
    law.append([int(num) for num in tmp[3].split(',')])
file.close()
print(word[666])
print(panish_class[666])
print(law[666])
result = {
    'word': word,
    'panish_class': panish_class,
    'law': law
}
fp.save_pickle_data('data_mid/train_data.pkl', result)
# ————————————————————————————————————————————————————————————————————————————
file = codecs.open("data_origin/test.txt", "rb", "utf-8")
word = []
stop_word_list = [' ', '', '。', '，', '-', '：', '“', '”', '"', '‘', '’', '！', '（', '）', '~', '、', ',', '.',
                  '(', ')', ';', '；', ':', '？', '?',  '～', '《', '》', '<', '>', '──', '─', '…', '……', '[', ']',
                  '【', '】', '~', ']', ']']
for line in file:
    tmp = re.split('\t|\n', line)
    word.append([word for word in jieba.cut(tmp[1]) if word not in stop_word_list])
file.close()
print(word[666])
fp.save_pickle_data('data_mid/test_data.pkl', word)
# ————————————————————————————————————————————————————————————————————————————
file = codecs.open("data_origin/law.txt", "rb", "utf-8")
word = []
stop_word_list = [' ', '', '。', '，', '-', '：', '“', '”', '"', '‘', '’', '！', '（', '）', '~', '、', ',', '.',
                  '(', ')', ';', '；', ':', '？', '?',  '～', '《', '》', '<', '>', '──', '─', '…', '……', '[', ']',
                  '【', '】', '~', ']', ']']
for line in file:
    tmp = re.split('\t|\n', line)
    word.append([word for word in jieba.cut(tmp[1]) if word not in stop_word_list])
file.close()
print(word[111])
fp.save_pickle_data('data_mid/law_data.pkl', word)


