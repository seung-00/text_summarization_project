> 2020년 2학기 데이터캡스톤 term-project 진행사항 문서입니다.

# 12주차 활동 상황 보고

seq2seq 와 어텐션 메커니즘을 이용한 텍스트 요약



## 1) 데이터 수집(완료)

* [cnbc_crawler](https://github.com/young-o/cnbc_crawler)
* [NEWS SUMMARY](https://www.kaggle.com/sunnysai12345/news-summary)
* [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews)

## 2) 데이터 정제(완료)

* [cleansing](https://github.com/young-o/text_summarization_project/blob/master/진행상황/cleansing/)

## 3) 데이터 전처리(완료)

* [preprocessing](https://github.com/young-o/text_summarization_project/blob/master/진행상황/preprocessing)

## 3) 모델링1 (완료)

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

  * embedding_dim = 128, hidden_size = 256, batch_size = 256

  * early stopping, 50 epoch

    ```python
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_29 (InputLayer)           (None, 50)           0                                            
    __________________________________________________________________________________________________
    embedding_14 (Embedding)        (None, 50, 128)      1024000     input_29[0][0]                   
    __________________________________________________________________________________________________
    lstm_29 (LSTM)                  [(None, 50, 256), (N 394240      embedding_14[0][0]               
    __________________________________________________________________________________________________
    input_30 (InputLayer)           (None, None)         0                                            
    __________________________________________________________________________________________________
    lstm_30 (LSTM)                  [(None, 50, 256), (N 525312      lstm_29[0][0]                    
    __________________________________________________________________________________________________
    embedding_15 (Embedding)        (None, None, 128)    256000      input_30[0][0]                   
    __________________________________________________________________________________________________
    lstm_31 (LSTM)                  [(None, 50, 256), (N 525312      lstm_30[0][0]                    
    __________________________________________________________________________________________________
    lstm_32 (LSTM)                  [(None, None, 256),  394240      embedding_15[0][0]               
                                                                     lstm_31[0][1]                    
                                                                     lstm_31[0][2]                    
    __________________________________________________________________________________________________
    attention_layer (AttentionLayer [(None, None, 256),  131328      lstm_31[0][0]                    
                                                                     lstm_32[0][0]                    
    __________________________________________________________________________________________________
    concat_layer (Concatenate)      (None, None, 512)    0           lstm_32[0][0]                    
                                                                     attention_layer[0][0]            
    _____________________________________________________________________________________
    dense_3 (Dense)                 (None, None, 2000)   1026000     concat_layer[0][0]               
    ==================================================================================================
    Total params: 4,276,432
    Trainable params: 4,276,432
    Non-trainable params: 0
    __________________________________________________________________________________________________
    
    ```

* 테스트

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
  
  

## 4) 모델링2 (완료)

* 버다나우 어텐션 메커니즘 적용 [Bahdanau Attention](https://github.com/thushv89/attention_keras) 

* 디코더에 어텐션 레이어 추가, 나머지는 위와 동일한 모델

  ```python
  __________________________________________________________________________________________________
  Layer (type)                    Output Shape         Param #     Connected to                     
  ==================================================================================================
  input_29 (InputLayer)           (None, 50)           0                                            
  __________________________________________________________________________________________________
  embedding_14 (Embedding)        (None, 50, 128)      1024000     input_29[0][0]                   
  __________________________________________________________________________________________________
  lstm_29 (LSTM)                  [(None, 50, 256), (N 394240      embedding_14[0][0]               
  __________________________________________________________________________________________________
  input_30 (InputLayer)           (None, None)         0                                            
  __________________________________________________________________________________________________
  lstm_30 (LSTM)                  [(None, 50, 256), (N 525312      lstm_29[0][0]                    
  __________________________________________________________________________________________________
  embedding_15 (Embedding)        (None, None, 128)    256000      input_30[0][0]                   
  __________________________________________________________________________________________________
  lstm_31 (LSTM)                  [(None, 50, 256), (N 525312      lstm_30[0][0]                    
  __________________________________________________________________________________________________
  lstm_32 (LSTM)                  [(None, None, 256),  394240      embedding_15[0][0]               
                                                                   lstm_31[0][1]                    
                                                                   lstm_31[0][2]                    
  __________________________________________________________________________________________________
  attention_layer (AttentionLayer [(None, None, 256),  131328      lstm_31[0][0]                    
                                                                   lstm_32[0][0]                    
  __________________________________________________________________________________________________
  concat_layer (Concatenate)      (None, None, 512)    0           lstm_32[0][0]                    
                                                                   attention_layer[0][0]            
  __________________________________________________________________________________________________
  dense_3 (Dense)                 (None, None, 2000)   1026000     concat_layer[0][0]               
  ==================================================================================================
  Total params: 4,276,432
  Trainable params: 4,276,432
  Non-trainable params: 0
  __________________________________________________________________________________________________
  ```

  

* 테스트

  ```
  Review : perfect stress free afternoon aroma tea makes house smell great drink grade honey bliss 
  Original summary : relax cup of tea 
  Predicted summary: great tea
  
  Review : dog loves stuff ground sprinkled dry food gobbles additives fillers carbs also use treat best price amazon quick delivery 
  Original summary : great 
  Predicted summary: great dog food
  
  Review : got bbq popchips amazon promotion price came taste good wish less salty would certainly purchase came less salty version 
  Original summary : tasty but wish it was less salty 
  Predicted summary : not the best
  
  Review : product arrived broken pieces flavor good actually threw garbage disappointing 
  Original summary : very disappointed 
  Predicted summary: not as described
  
  Review : buying quaker oats granola bars nature valley chewy bars better tasting make great snack go chocolate peanuts raisins get better 
  Original summary : my new granola bar 
  Predicted summary: great snack
  ```

  

## 5) 모델 테스트 및 성능 파악

* [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))

* https://github.com/pltrdy/rouge 사용

* 평가 결과

  * seq2seq

    ```python
    {'rouge-1': {'f': 0.12919550942279673, 'p': 0.16129191706339857, 'r': 0.12045058404224902}, 
     'rouge-2': {'f': 0.02336627905557511, 'p': 0.029295767319649307, 'r': 0.022062228465880983}, 
     'rouge-l': {'f': 0.1297852671404133, 'p': 0.16182392882228927, 'r': 0.12092387567686128}}
    ```

  * seq2seq, 버다나우 어텐션 적용

    ```python
    {'rouge-1': {'f': 0.2608695605353498, 'p': 0.3488372093023256, 'r': 0.20833333333333334}, 
     'rouge-2': {'f': 0.0786324739396596, 'p': 0.0363636363636364, 'r': 0.6363636363636364}, 
     'rouge-l': {'f': 0.2297852671404133, 'p': 0.36182392882228927, 'r': 0.24192387567686128}}
    ```

    

## 6) 모델 개선(진행 중)

* hyper parameter tuning
* Luong 어텐션 메커니즘
* 추가적인 조사?



