# 指定GPU及显存使用
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
# -----------------------------------------------------------------------------------------------------------------
from keras.layers import Input, Dense, LSTM, Masking, Embedding
from keras.models import Model, load_model
from keras.layers.merge import Dot
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import file_pickle as fp
from sklearn.model_selection import train_test_split
import numpy as np


class LstmCosBinarymodel:
    def __init__(self, case_inputShape, law_inputShape, embedding_weights):
        case_input = Input(shape=case_inputShape, name='case_input')
        case_x = Embedding(input_dim=len(embedding_weights), output_dim=100,
                          weights=[embedding_weights], input_length=case_inputShape[-1])(case_input)
        case_x = Masking(mask_value=0)(case_x)
        case_x = LSTM(100)(case_x)

        law_input = Input(shape=law_inputShape, name='law_input')
        law_x = Embedding(input_dim=len(embedding_weights), output_dim=100,
                          weights=[embedding_weights], input_length=law_inputShape[-1])(law_input)
        law_x = Masking(mask_value=0)(law_x)
        law_x = LSTM(100)(law_x)

        x = Dot(axes=-1, normalize=True)([case_x, law_x])
        output = Dense(1, activation='sigmoid', name='output')(x)

        self.model = Model(inputs=[case_input, law_input], outputs=[output])
        self.model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def fit(self, x_train_case, x_train_law, y_train):
        print(np.array(x_train_case).shape)
        early_stopping = EarlyStopping(monitor='loss', patience=2, verbose=0, mode='auto')
        modle_check_point = ModelCheckpoint(filepath='../data_model/lstm_model_plus1.h5')
        self.model.fit({'case_input': np.array(x_train_case), 'law_input': np.array(x_train_law)},
                       {'output': np.array(y_train)},
                       callbacks=[early_stopping, modle_check_point],
                       batch_size=1000, epochs=100)

    def predict(self, x_test):
        return self.model.predict(np.array(x_test))

    def save(self, filename):
        self.model.save(filename)

train_data = fp.get_pickle_data('../data_feature/train_sample1.pkl')
embedding_weights = fp.get_pickle_data('../data_result/word_embedding.pkl')['embedding_weights']
data_x = [data[0:2] for data in train_data]
data_y = [data[2] for data in train_data]


# x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, stratify=data_y, random_state=1)
print('go')
model = LstmCosBinarymodel(case_inputShape=data_x[0][0].shape,
                           law_inputShape=data_x[0][1].shape,
                           embedding_weights=embedding_weights)

model.fit(x_train_case=[data[0] for data in data_x],
          x_train_law=[data[1] for data in data_x],
          y_train=data_y)

model.save('../data_model/lstm_model_plus1.h5')






