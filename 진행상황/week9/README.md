> 2020년 2학기 데이터캡스톤 term-project 진행사항 문서입니다.

# 9주차 활동 상황 보고

seq2seq 와 어텐션 메커니즘을 이용한 텍스트 요약



## 1) 데이터 수집(완료)

* [cnbc_crawler](https://github.com/young-o/cnbc_crawler)

## 2) 데이터 정제(완료)

* [cleansing](https://github.com/young-o/text_summarization_project/blob/master/진행상황/cleansing/)

## 3) 데이터 전처리(완료)

* [preprocessing](https://github.com/young-o/text_summarization_project/blob/master/진행상황/preprocessing)

## 3) 모델 학습(진행 중)

* [modeling](https://github.com/young-o/text_summarization_project/blob/master/진행상황/modeling)

* [Bahdanau Attention](https://github.com/thushv89/attention_keras) 활용

* 모델 설계

  * seq2seq 모델, attention 기법 사용
  * tf 2.0 keras functional API 
  * 인코더(LSTM 3), 디코더(LSTM 1)의 seq2seq 구조
  * 옵티마이저: [rmsprop](https://keras.io/ko/optimizers/)
  * [early stopping](early stopping), 50 epoch

  ```python
  embedding_dim = 128
  hidden_size = 256
  # 임베딩 벡터 크기: 128, 은닉 상태 크기: 256
  
  ### encoder
  encoder_inputs = Input(shape=(src_max_len,))
  
  # Embedding
  enc_emb = Embedding(src_vocab, embedding_dim)(encoder_inputs)
  
  # LSTM 1
  encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True ,dropout = 0.4, recurrent_dropout = 0.4)
  encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)
  
  # LSTM 2
  encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
  encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)
  
  # LSTM 3
  encoder_lstm3 = LSTM(hidden_size, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
  encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)
  
  ### decoder
  decoder_inputs = Input(shape=(None,))
  
  # Embedding
  dec_emb = Embedding(smry_vocab, embedding_dim)(decoder_inputs)
  
  # LSTM
  decoder_lstm = LSTM(hidden_size, return_sequences = True, return_state = True, dropout = 0.4, recurrent_dropout=0.2)
  decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = [state_h, state_c])
  # initial_state를 인코더의 state로 준다
  
  # 디코더 출력층
  decoder_softmax_layer = Dense(smry_vocab, activation = 'softmax')
  decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs) 
  
  # 모델 정의
  model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
  model.summary()
  
  model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
  
  # 조기 종료, 50 epoch
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 2)
  history = model.fit([X_train, y_train[:,:-1]], y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:,1:] \
                    ,epochs=50, callbacks=[es], batch_size = 256, validation_data=([X_test, y_test[:,:-1]], \
                    y_test.reshape(y_test.shape[0], y_test.shape[1], 1)[:,1:]))
  
  ```

  

* 모델 학습이 ephoch 1 에서 죽는 현상

  * 원인: [메모리 부족](https://manage.dediserve.com/knowledgebase/article/145/what-is-ram-exhaustion---also-known-as-oom--/)

  * <img src="https://user-images.githubusercontent.com/46865281/82030283-32a28900-96d3-11ea-9871-ba55d8d333b0.png" width="600" height="500">

  * 학교 GPU 서버 발급

    

## 4) 모델 테스트 및 성능 파악

* [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))



## 5) 모델 개선 방향 조사

* issue

  * 전처리 과정 개선?
    * 워드 임베딩
    * pre-trained 임베딩 벡터
  * 모델 설계
    * 교사강요
  * 모델 튜닝
  * 모델 변경
    * transformer

  