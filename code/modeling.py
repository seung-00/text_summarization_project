import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing
from attention import AttentionLayer
import tensorflow as tf
keras = tf.keras
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint


if __name__ == "__main__":
    path = "/Users/seungyoungoh/workspace/text_summarization_project/"
    data = pd.read_csv(path+"/data/sample.csv", error_bad_lines = False)
    data = data.rename({'body':'src', 'key_point':'smry'}, axis = 'columns')[['src','smry']]
    pr = preprocessing.Preprocessor(data)
    src_max_len, smry_max_len, src_vocab, smry_vocab, X_train, X_test, y_train, y_test = pr.preprocess()

    ### modeling
    embedding_dim = 128
    hidden_size = 256

    # 인코더
    encoder_inputs = Input(shape=(src_max_len,))

    # 인코더의 임베딩 층
    enc_emb = Embedding(src_vocab, embedding_dim)(encoder_inputs)

    # 인코더의 LSTM 1
    encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True ,dropout = 0.4, recurrent_dropout = 0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    # 인코더의 LSTM 2
    encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    # 인코더의 LSTM 3
    encoder_lstm3 = LSTM(hidden_size, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

    # 디코더
    decoder_inputs = Input(shape=(None,))

    # 디코더의 임베딩 층
    dec_emb = Embedding(smry_vocab, embedding_dim)(decoder_inputs)

    # 디코더의 LSTM
    decoder_lstm = LSTM(hidden_size, return_sequences = True, return_state = True, dropout = 0.4, recurrent_dropout=0.2)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = [state_h, state_c])



    # # 어텐션 층(어텐션 함수)
    # attn_layer = AttentionLayer(name='attention_layer')
    # attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

    # # 어텐션의 결과와 디코더의 hidden state들을 연결
    # decoder_concat_input = Concatenate(axis = -1, name='concat_layer')([decoder_outputs, attn_out])

    # # 디코더의 출력층
    # decoder_softmax_layer = Dense(smry_vocab, activation='softmax')
    # decoder_softmax_outputs = decoder_softmax_layer(decoder_concat_input)

    # # 모델 정의
    # model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
    # model.summary()

    # model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 2)
    # history = model.fit([X_train, y_train[:,:-1]], y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:,1:] \
    #                 ,epochs=50, callbacks=[es], batch_size = 256, validation_data=([X_test, y_test[:,:-1]], \
    #                 y_test.reshape(y_test.shape[0], y_test.shape[1], 1)[:,1:]))

    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()
