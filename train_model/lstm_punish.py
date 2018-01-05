# 指定GPU及显存使用
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
# __________________________________________________________________________________________________
from keras.layers import Input, Dense, LSTM, Masking, Embedding
from keras.models import Model, load_model
from keras.layers.merge import Dot, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
import utils.file_pickle as fp
from sklearn.model_selection import train_test_split
import numpy as np


class LstmPunishModel:
    def __init__(self, case_inputShape, law_inputShape, embedding_weights):
        case_input = Input(shape=case_inputShape, name='case_input')
        case_x = Embedding(input_dim=len(embedding_weights), output_dim=100,
                          weights=[embedding_weights], input_length=case_inputShape[-1])(case_input)
        case_x = Masking(mask_value=0)(case_x)
        case_x = LSTM(100)(case_x)
        case_x = Dense(50, activation='selu')(case_x)

        law_input = Input(shape=law_inputShape, name='law_input')
        law_x = Dense(100, activation='selu')(law_input)
        law_x = Dense(50, activation='selu')(law_x)

        x = Concatenate()([case_x, law_x])
        x = Dense(20, activation='selu')(x)
        output = Dense(8, activation='softmax', name='output')(x)

        self.model = Model(inputs=[case_input, law_input], outputs=[output])
        self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    def fit(self, x_train_case, x_train_law, y_train):
        print(np.array(x_train_case).shape)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
        # modle_check_point = ModelCheckpoint(filepath='model/lstm_punish_model.h5')
        self.model.fit({'case_input': np.array(x_train_case), 'law_input': np.array(x_train_law)},
                       {'output': np.array(y_train)},
                       callbacks=None,
                       batch_size=50, epochs=300)

    def predict(self, x_test):
        return self.model.predict(np.array(x_test))

    def save(self, filename):
        self.model.save(filename)

data = fp.get_pickle_data('../data_feature/train_punish.pkl')
embedding_weights = fp.get_pickle_data('../data_result/word_embedding.pkl')['embedding_weights']
data_x = data['train_x']
data_y = data['train_y']
del data

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, stratify=data_y, random_state=1)

model = LstmPunishModel(case_inputShape=x_train[0][0].shape,
                        law_inputShape=x_train[0][1].shape,
                        embedding_weights=embedding_weights)

model.fit(x_train_case=[data[0] for data in x_train],
          x_train_law=[data[1] for data in x_train],
          y_train=y_train)
