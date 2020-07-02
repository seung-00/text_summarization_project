# text summarization project

* 텍스트 자동 요약 모델
  * tensoflow 2.2v, titan xp
  * seq2seq, [Bahdanau Attention](https://github.com/thushv89/attention_keras), [ROUGE-1](https://github.com/pltrdy/rouge)
* [kaggle 음식 리뷰 데이터](https://www.kaggle.com/snap/amazon-fine-food-reviews) 를 학습시켰습니다.
* 경희대 데이터분석 캡스톤 프로젝트로 진행했습니다.



## 수행 방법

### 1. 전처리

* 아마존 음식 리뷰 데이터 핸들링
* 불용어 처리(NLTK 사용)
* 정규화 사전 생성
* 전처리 함수 정의
* 등장 빈도수 낮은 단어 삭제
* 정수 인코딩 수행
* 데이터 분포 확인 후 패딩

### 2. seq2seq 모델 구성

* Amazone Fine Food Reviews 데이터 중 80% 학습에 사용(20% 테스트)
* 테스트 데이터 세트로 학습된 모델로부터 값 예측
* ROUGE-1 F1 Score 로 seq2seq 모델 평가
* 이후 매개변수 및 레이어 구성을 바꾼 후 평가 결과를 반복적으로 확인하여 최적의 모델을 찾음.
  * 인코더에서 LSTM 레이어 개수에 따른 성능 변화를 확인했습니다.

### 3. seq2seq 모델에 Bahdanau 어텐션 메커니즘 적용

* 위와 같은 절차로 최적의 모델을 찾는다.



## 수행 결과

### 1. 전처리

* 102913 행 리뷰 데이터, 원문과 요약문 2열로 구성
* 원문에 등장한 단어들 중 그 빈도가 7 아래인 단어, 요약문에 등장한 단어들 중 그 빈도가 6 아래인 단어를 제외하고 단어 집합 생성 (전체 단어 빈도의 약 4.4%, 6.7%)
* 정수 인코딩 진행
* 텍스트 데이터 약 80% 정도를 포함할 수 있도록 최대 본문 길이 50, 요약 길이 8로 패딩

### 2. seq2seq 모델 구성

* 인코더: 임베딩 레이어(크기 128), LSTM 레이어 (1~3)(크기 256), 인코더 길이: 50
* 디코더: 임베딩 레이어(크기 128), LSTM 레이어 1(크기 256), Dense 레이어(Softmax), 디코더 길이: 8
* 초매개변수(Hyper Parameter) 및 학습 구성
  * 옵티마이저: adam
  * loss function: sparse_categorical_crossentropy
  * 배치 크기: 256
  * 에포크: 50, 학습 조기 종료(Early Stopping)으로 과적합 방지
### 3. seq2seq with bahdanau attention 구성
* 인코더: 2와 동일
* 디코더: 임베딩 레이어, LSTM 레이어, Attention 레이어, Dense 레이어(Softmax)
* 임베딩 벡터 크기: 128, 인코더 및 디코더 레이어 크기: 256, 배치 크기: 256, 인코더 길이: 50, 디코더 길이:8
* 초매개변수(Hyper Parameter) 및 학습 구성
  * 2와 동일

### 4. ROUGE-1 평가 결과

* seq2seq

|                                                              | precision | recall | F1     |
| ------------------------------------------------------------ | --------- | ------ | ------ |
| [인코더 LSTM1](https://github.com/seung-00/text_summarization_project/tree/master/src/seq2seq/case1) | 0.1412    | 0.1083 | 0.1148 |
| [인코더 LSTM 1,2](https://github.com/seung-00/text_summarization_project/tree/master/src/seq2seq/case2) | 0.1441    | 0.1079 | 0.1172 |
| [인코더 LSTM3 1, 2, 3](https://github.com/seung-00/text_summarization_project/tree/master/src/seq2seq/case3) | 0.1410    | 0.1072 | 0.1141 |

* seq2seq with Bahdanau Attention

|                                                              | precision | recall | F1     |
| ------------------------------------------------------------ | --------- | ------ | ------ |
| [인코더 LSTM1](https://github.com/seung-00/text_summarization_project/tree/master/src/seq2seq_bahdanau/case1) | 0.1729    | 0.1320 | 0.1404 |
| [인코더 LSTM 1,2](https://github.com/seung-00/text_summarization_project/tree/master/src/seq2seq_bahdanau/case2) | 0.1603    | 0.1185 | 0.1273 |
| [인코더 LSTM3 1, 2, 3](https://github.com/seung-00/text_summarization_project/tree/master/src/seq2seq_bahdanau/case3) | 0.1606    | 0.1193 | 0.1284 |



### 5. 실행 예시

* 예시 1

| 원문        | product wonderful especially since lower fat content mayonnaise even low fat mayo made usa tastes terrific little tang lighter mayo pours easily container certainly makes healthier tastier refreshing alternative regular mayo highly recommend buy especially lower price offered amazon line sources one bottle listed way |
| ----------- | ------------------------------------------------------------ |
| 실제 요약문 | wonderful refreshing taste                                   |
| 예측 요약문 | best ever                                                    |

* 예시 2

| 원문        | really love vanilla hemp protein powder strong vanilla taste helps mask super nutty flavor otherwise mixes well tastes good throw banana fruit da healthy whole food satisfies hunger good long time definitely recommend |
| ----------- | ------------------------------------------------------------ |
| 실제 요약문 | good for you                                                 |
| 예측 요약문 | great product                                                |