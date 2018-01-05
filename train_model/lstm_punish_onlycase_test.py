# 指定GPU及显存使用
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
# _________________________________________________________________________________________________________
from keras.layers import Input, Dense, LSTM, Masking, Embedding, Dropout
from keras.models import Model, load_model
from keras.layers.merge import Dot, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
import utils.file_pickle as fp
from sklearn.model_selection import train_test_split
import numpy as np
import json

data = fp.get_pickle_data('../data_feature/test_punish2.pkl')
data_x = data['value']
data_id = data['id']
del data
model = load_model('../data_model/lstm_punish_model3_all_3e_100batch.h5')
pred = model.predict({'case_input': np.array(data_x)})
pred = np.argmax(pred, axis=1)+1
filename = "../predict_result/punish_result2.json"
with open(filename, "w") as f:
    for i in range(len(pred)):
        tmp_dict = json.dumps({"id": str(data_id[i]), "penalty": int(pred[i]), "laws": [351, 67, 72, 73]})
        f.writelines(tmp_dict+'\n')
        f.flush()
    f.close()
    print(filename+" 写入文件完成...")

