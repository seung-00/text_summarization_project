> 2020년 2학기 데이터캡스톤 term-project 진행사항 문서입니다.

# 11주차 활동 상황 보고

seq2seq 와 어텐션 메커니즘을 이용한 텍스트 요약



## 1) 데이터 수집(완료)

* [cnbc_crawler](https://github.com/young-o/cnbc_crawler)
* [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews)

## 2) 데이터 정제(완료)

* [cleansing](https://github.com/young-o/text_summarization_project/blob/master/진행상황/cleansing/)

## 3) 데이터 전처리(완료)

* [preprocessing](https://github.com/young-o/text_summarization_project/blob/master/진행상황/preprocessing)

## 3) 모델 학습1 (완료)

* [modeling](https://github.com/young-o/text_summarization_project/blob/master/진행상황/modeling)
* Issues
  * 모델 학습이 epoch 1 에서 죽는 현상

    * 원인: [메모리 부족](https://manage.dediserve.com/knowledgebase/article/145/what-is-ram-exhaustion---also-known-as-oom--/)
  * 해결책: 학교 GPU 서버 발급, batch size 및 time steps 줄임
    * 데이터 세트 개수 
  * NaN loss 

* 모델 설계

  * seq2seq 모델
  * tf 2.0 keras functional API 
  * 인코더(LSTM 3), 디코더(LSTM 1), 디코더 출력층(activation = softmax)의 seq2seq 구조
  * 옵티마이저: [adam](https://keras.io/ko/optimizers/), loss: sparse_categorical_crossentropy 
  * early stopping, 50 epoch

* 학습 결과

  ```python
  __________________________________________________________________________________________________
  Layer (type)                    Output Shape         Param #     Connected to                     
  ==================================================================================================
  input_1 (InputLayer)            (None, 50)           0                                            
  __________________________________________________________________________________________________
  embedding (Embedding)           (None, 50, 128)      1024000     input_1[0][0]                    
  __________________________________________________________________________________________________
  lstm (LSTM)                     [(None, 50, 256), (N 394240      embedding[0][0]                  
  __________________________________________________________________________________________________
  input_2 (InputLayer)            (None, None)         0                                            
  __________________________________________________________________________________________________
  lstm_1 (LSTM)                   [(None, 50, 256), (N 525312      lstm[0][0]                       
  __________________________________________________________________________________________________
  embedding_1 (Embedding)         (None, None, 128)    256000      input_2[0][0]                    
  __________________________________________________________________________________________________
  lstm_2 (LSTM)                   [(None, 50, 256), (N 525312      lstm_1[0][0]                     
  __________________________________________________________________________________________________
  lstm_3 (LSTM)                   [(None, None, 256),  394240      embedding_1[0][0]                
                                                                   lstm_2[0][1]                     
                                                                   lstm_2[0][2]                     
  __________________________________________________________________________________________________
  dense (Dense)                   (None, None, 2000)   514000      lstm_3[0][0]                     
  ==================================================================================================
  Total params: 3,633,104
  Trainable params: 3,633,104
  Non-trainable params: 0
  __________________________________________________________________________________________________
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/gradients_impl.py:112: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
    "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
  Train on 51404 samples, validate on 12813 samples
  Epoch 1/50
  51404/51404 [==============================] - 73s 1ms/step - loss: 3.2903 - val_loss: 2.8126
  ... 생략
  Epoch 36/50
  51404/51404 [==============================] - 66s 1ms/step - loss: 1.8284 - val_loss: 2.1435
  Epoch 37/50
  51404/51404 [==============================] - 66s 1ms/step - loss: 1.8140 - val_loss: 2.1428
  Epoch 38/50
  51404/51404 [==============================] - 66s 1ms/step - loss: 1.8012 - val_loss: 2.1433
  Epoch 39/50
  51404/51404 [==============================] - 66s 1ms/step - loss: 1.7896 - val_loss: 2.1433
  Epoch 00039: early stopping
  ```

  

* 테스트

  ```python
  for i in range(500, 1000):
      X_test[i]
      print("Review : ",seq2text(X_test[i]))
      print("Original summary :",seq2summary(y_test[i]))
      print("Predicted summary :",decode_sequence(X_test[i].reshape(1,src_max_len)))
      print("\n")
  ```

  ```shell
  Review :  really like drink complaint would need drink cans time oz wee bit petite side came oz would much likely buy packaging would much less costly really tasty product though got ingredients yum 
  Original summary : tasty but too for me 
  Predicted summary :  great taste
  
  Review :  yummy mild delicious new favorite mild coffees called rocket fuel taste like oil truly medium roast pleasant morning makes glad good cup coffee 
  Original summary : my new favorite 
  Predicted summary :  great coffee
  
  Review :  always purchased star tuna thought would try brand change pace taste tuna pleasant much basil spices 
  Original summary : not the greatest tasting 
  Predicted summary :  not bad
  ```

  

## 4) 모델 학습2 (예정)

* [Bahdanau Attention](https://github.com/thushv89/attention_keras) 



## 5) 모델 테스트 및 성능 파악

* [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))



## 6) 모델 개선 방향 조사

* issues

  * 전처리 과정 개선?
    * 워드 임베딩
    * pre-trained 임베딩 벡터
  * 모델 설계
    * 교사강요
  * 모델 튜닝
  * 모델 변경
    * transformer

  
  
  
  
  
  
  