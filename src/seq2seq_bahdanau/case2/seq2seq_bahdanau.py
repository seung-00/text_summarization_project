import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
from tensorflow import keras

import urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/thushv89/attention_keras/master/src/layers/attention.py", filename="attention.py")
from attention import AttentionLayer
tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')



data = pd.read_csv("/home/data/cleaned_sample.csv", error_bad_lines = False)
data = data[['Text','Summary']]
data = data.rename(columns = {"Text":"src","Summary":"smry"})
print(data.columns)
print(data.shape)

src_max_len = 50
smry_max_len = 8
src = list(data['src'])
smry = list(data['smry'])

# partitiotn
X_train, X_test, y_train, y_test = train_test_split(src, smry, test_size=0.2, random_state=0, shuffle=True)
X_train_word, X_test_word, y_train_word, y_test_word = X_train, X_test, y_train, y_test


### Integer Encoding

## src data
src_tokenizer = keras.preprocessing.text.Tokenizer()
# fit_on_texts(corpus) generates a set of words based on fequency
src_tokenizer.fit_on_texts(X_train)

threshold = 7
total_src_cnt = len(src_tokenizer.word_index)
rare_src_cnt = 0
total_src_freq = 0
rare_src_freq = 0

for word, count in src_tokenizer.word_counts.items():
    # items: (word, count)
    total_src_freq = total_src_freq + count
    if(count < threshold):
        rare_src_cnt = rare_src_cnt + 1
        rare_src_freq = rare_src_freq + count

## smry data
smry_tokenizer = keras.preprocessing.text.Tokenizer()
smry_tokenizer.fit_on_texts(y_train)

threshold = 6
total_smry_cnt = len(smry_tokenizer.word_index)
rare_smry_cnt = 0
total_smry_freq = 0
rare_smry_freq = 0

for word, count in smry_tokenizer.word_counts.items():
    # items: (word, count)
    total_smry_freq = total_smry_freq + count
    if(count < threshold):
        rare_smry_cnt = rare_smry_cnt + 1
        rare_smry_freq = rare_smry_freq + count

src_vocab = 8000
src_tokenizer = keras.preprocessing.text.Tokenizer(num_words = src_vocab)
src_tokenizer.fit_on_texts(X_train)

# text to int sequences
X_train = src_tokenizer.texts_to_sequences(X_train)
X_test = src_tokenizer.texts_to_sequences(X_test)

#smry_vocab = total_smry_cnt - rare_smry_cnt
smry_vocab = 2000
smry_tokenizer = keras.preprocessing.text.Tokenizer(num_words = smry_vocab)
smry_tokenizer.fit_on_texts(y_train)

y_train = smry_tokenizer.texts_to_sequences(y_train)
y_test = smry_tokenizer.texts_to_sequences(y_test)

# delete empty samples
drop_train = [index for index, sentence in enumerate(y_train) if len(sentence) == 2]
drop_test = [index for index, sentence in enumerate(y_test) if len(sentence) == 2]

X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
X_test = np.delete(X_test, drop_test, axis=0)
y_test = np.delete(y_test, drop_test, axis=0)

### padding
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen = src_max_len, padding='post')
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen = src_max_len, padding='post')
y_train = keras.preprocessing.sequence.pad_sequences(y_train, maxlen = smry_max_len, padding='post')
y_test = keras.preprocessing.sequence.pad_sequences(y_test, maxlen = smry_max_len, padding='post')

### modeling
embedding_dim = 128
hidden_size = 256

# 인코더
encoder_inputs = keras.layers.Input(shape=(src_max_len,))

# 인코더의 임베딩 층
enc_emb_layer = keras.layers.Embedding(src_vocab, embedding_dim)
enc_emb = enc_emb_layer(encoder_inputs)

# 인코더의 LSTM 1
encoder_lstm1 = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True ,dropout = 0.4, recurrent_dropout = 0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# 인코더의 LSTM 2
encoder_lstm2 = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm2(encoder_output1)

# 디코더
decoder_inputs = keras.layers.Input(shape=(None,))

# 디코더의 임베딩 층
dec_emb_layer = keras.layers.Embedding(smry_vocab, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)

# 디코더의 LSTM
decoder_lstm = keras.layers.LSTM(hidden_size, return_sequences = True, return_state = True, dropout = 0.4, recurrent_dropout=0.2)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = [state_h, state_c])

attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])


# 어텐션의 결과와 디코더의 hidden state들을 연결
decoder_concat_input = keras.layers.Concatenate(axis = -1, name='concat_layer')([decoder_outputs, attn_out])

# 디코더의 출력층
decoder_softmax_layer = keras.layers.Dense(smry_vocab, activation='softmax')
decoder_softmax_outputs = decoder_softmax_layer(decoder_concat_input)

model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
print(model.summary())

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=adam, loss='sparse_categorical_crossentropy')

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 2)

hist = model.fit([X_train, y_train[:,:-1]], y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:,1:] \
                  ,epochs=50, callbacks=[es], batch_size = 256, validation_data=([X_test, y_test[:,:-1]], \
                  y_test.reshape(y_test.shape[0], y_test.shape[1], 1)[:,1:]))


# encoder model
encoder_model = tf.keras.models.Model(inputs=encoder_inputs, outputs= [encoder_outputs, state_h, state_c])

# encoder의 산출물, 및 이전 시점의 상태들을 받는 입력층을 정의
decoder_input_state_h = tf.keras.layers.Input(shape=(hidden_size,))
decoder_input_state_c = tf.keras.layers.Input(shape=(hidden_size,))

decoder_hidden_state_input = tf.keras.layers.Input(shape=(src_max_len,hidden_size))

# 임베딩 층
dec_emb2= dec_emb_layer(decoder_inputs)

# 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태인 state_h와 state_c를 버리지 않음.
decoder_outputs2, state_h, state_c = decoder_lstm(dec_emb2, initial_state= [decoder_input_state_h, decoder_input_state_c])
decoder_output_states = [state_h, state_c]



# 어텐션 함수
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])

decoder_inf_concat = tf.keras.layers.Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# 디코더의 출력층
decoder_outputs2 = decoder_softmax_layer(decoder_inf_concat)

# decoder model
decoder_model = tf.keras.models.Model(
    [decoder_inputs] + [decoder_hidden_state_input, decoder_input_state_h, decoder_input_state_c],
    [decoder_outputs2] + decoder_output_states)

print(encoder_model.summary())
print(decoder_model.summary())

encoder_model.save('encoder2.h5')
decoder_model.save('decoder2.h5')