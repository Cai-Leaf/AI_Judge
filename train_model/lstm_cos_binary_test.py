# 指定GPU及显存使用
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
# -----------------------------------------------------------------------------------------------------------------
from utils import file_pickle as fp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from keras.models import Model, load_model

# 验证集测试
# train_data = fp.get_pickle_data('../data_feature/train_sample1.pkl')
# embedding_weights = fp.get_pickle_data('../data_result/word_embedding.pkl')['embedding_weights']
# data_x = [data[0:2] for data in train_data]
# data_y = [data[2] for data in train_data]
#
# x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, stratify=data_y, random_state=1)
#
#
#
# model = load_model('../data_model/lstm_model1.h5')
#
# pred = model.predict({'case_input': np.array([data[0] for data in x_test[:1000]]),
#                'law_input': np.array([data[1] for data in x_test[:1000]])})
#
# pred = pred[:, 0]
# pred = np.where(pred > 0.1, pred, 0)
# pred = np.where(pred <= 0.1, pred, 1)
# print('precision_score', precision_score(y_true=data_y[:1000], y_pred=pred))
# print('recall', recall_score(y_true=data_y[:1000], y_pred=pred))
# print('f1', f1_score(y_true=data_y[:1000], y_pred=pred))


# 测试集测试
data = fp.get_pickle_data('../data_feature/test.pkl')
laws = data['law_vector']
caces = data['case_vector']
test_case = fp.get_pickle_data('../data_mid/keyword_tfidf_200.pkl')['test']
del data

model = load_model('../data_model/lstm_model_plus1.h5')
cace_i = 0
result = []
for cace in caces:
    case_input = [cace]*len(laws)
    pred = model.predict({'case_input': np.array(case_input), 'law_input': np.array(laws)})
    pred = pred[:, 0]
    # 选择得分最高的前5个法条
    cur_index = np.where(pred >= 0.85)[0]
    cur_val = pred[cur_index]
    sort_val = sorted([(cur_index[i]+1, cur_val[i]) for i in range(len(cur_index))], key=lambda k: k[1], reverse=True)
    tmp_result = [val for val in sort_val[0:4]]
    result.append(tmp_result)
    # 每隔20次存结果
    if cace_i % 20 == 0:
        fp.save_pickle_data('../data_result/result1.h5', result)
    # 输出信息
    print(test_case[cace_i])
    print(cace_i, tmp_result)
    print('-----------------------------')
    cace_i += 1





