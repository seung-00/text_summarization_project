> 2020년 2학기 데이터캡스톤 term-project 진행사항 문서입니다.

# 8주차 활동 상황 보고

seq2seq 와 어텐션 메커니즘을 이용한 텍스트 요약



## 1) 데이터 수집(완료)

* [cnbc_crawler](https://github.com/young-o/cnbc_crawler)

## 2) 데이터 정제(완료)

* [cleansing](https://github.com/young-o/text_summarization_project/blob/master/진행상황/cleansing/)

## 3) 데이터 전처리(완료)

* [preprocessing](https://github.com/young-o/text_summarization_project/blob/master/진행상황/preprocessing)

## 3) 모델 학습(진행 중)

* [modeling](https://github.com/young-o/text_summarization_project/blob/master/진행상황/modeling)
* seq2seq 모델, attention 기법 사용
* [Bahdanau Attention](https://github.com/thushv89/attention_keras) 활용
* 어텐션 함수 코드 작성 중 오류 발생하여 해결 중...



## 4) 모델 테스트 및 성능 파악

* [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))



## 5) 모델 개선 방향 조사

* 전처리 과정 개선?
  * 워드 임베딩
  * pre-trained 임베딩 벡터
* 모델 튜닝
* 모델 변경
  * transformer

