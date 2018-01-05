# 指定GPU及显存使用
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
# _________________________________________________________________________________________________________
from keras.layers import Input, Dense, LSTM, Masking, Embedding, Dropout
from keras.models import Model, load_model
from keras.layers.merge import Dot, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.file_pickle import get_pickle_data
from sklearn.model_selection import train_test_split
import numpy as np


class LstmPunishModel:
    def __init__(self, case_inputShape, embedding_weights):
        case_input = Input(shape=case_inputShape, name='case_input')
        case_x = Embedding(input_dim=len(embedding_weights), output_dim=100,
                          weights=[embedding_weights], input_length=case_inputShape[-1])(case_input)
        case_x = Masking(mask_value=0)(case_x)
        case_x = LSTM(100)(case_x)
        # case_x = Dense(20, activation='selu', )(case_x)
        case_x = Dropout(0.5)(case_x)
        output = Dense(8, activation='softmax', name='output')(case_x)

        self.model = Model(inputs=[case_input], outputs=[output])
        self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    def fit(self, x_train_case, y_train, validation_data=None):
        print(np.array(x_train_case).shape)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
        modle_check_point = ModelCheckpoint(filepath='../data_model/lstm_punish_model3_dropout.h5')
        self.model.fit({'case_input': np.array(x_train_case)},
                       {'output': np.array(y_train)},
                       callbacks=None,
                       batch_size=100, epochs=10, validation_data=validation_data, validation_split=0.1)

    def predict(self, x_test):
        return self.model.predict(np.array(x_test))

    def save(self, filename):
        self.model.save(filename)
data = get_pickle_data('../data_feature/train_punish2.pkl')
embedding_weights = get_pickle_data('../data_result/word_embedding.pkl')['embedding_weights']
data_x = data['train_x']
data_y = data['train_y']
del data
model = LstmPunishModel(case_inputShape=data_x[0].shape,
                        embedding_weights=embedding_weights)

model.fit(x_train_case=[data for data in data_x],
          y_train=data_y)
