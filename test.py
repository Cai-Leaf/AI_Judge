from utils import file_pickle as fp
import numpy as np
import re
# train = fp.get_pickle_data('data_mid/keyword_tfidf_200.pkl')
# texts = train['train']
# for i in range(100):
#     print(texts[i])
# law = train['law']
# for i in range(len(texts)):
#     if len(texts[i]) > 50 and len(texts[i]) < 100:
#         print(i, law[i], texts[i])

# data = fp.get_pickle_data('data_mid/train_data.pkl')
#
# law = data['law']
# print(len(law))
# panish_class = data['panish_class']
#
# data = {
#     'law': law,
#     'panish_class': panish_class
# }
# fp.save_pickle_data('law_panisclass.pkl', data)
# test = fp.get_pickle_data('data_result/result1.h5')
# print(test[0:5])

test = '2016.4'
contain_number = re.compile('^.*\\d+.*')
result = contain_number.match(test)

if result:
    print('yes')
else:
    print('no')
