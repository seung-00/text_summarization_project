> 2020년 2학기 데이터캡스톤 term-project 진행사항 문서입니다.

# 7주차 활동 상황 보고

seq2seq 와 어텐션 메커니즘을 이용한 텍스트 요약



## 1) 데이터 수집(완료)



## 2) 데이터 정제(cleaning)

* 뉴스 corpus 로부터 noise 제거
* 대략 3800개 우선 이걸로 ok

* [cleansing](https://github.com/young-o/text_summarization_project/blob/master/code/cleansing.py)

| key_point                                                    | body                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| The Federal Reserve announced a barrage of new programs to help keep the  market functioning.Among the moves is an open-ended commitment to keep buying  assets under its quantitative easing measures.There are multiple other  programs, including one for Main Street business lending and others aimed at  keeping credit flowing.The Fed will be moving for the first time into  corporate bonds, purchasing the investment-grade securities in primary and  secondary markets and through exchange-traded funds. | The Federal Reserve said Monday it will launch a barrage of programs  aimed at helping markets function more efficiently amid the coronavirus  crisis.Among the initiatives is a commitment to continue its asset purchasing  program "in the amounts needed to support smooth market functioning and  effective transmission of monetary policy to broader financial conditions and  the economy."That represents a potentially new chapter in the Fed's  "money printing" as it commits to keep expanding its balance sheet  as necessary, rather than a commitment to a set amount.The Fed also will be  moving for the first time into corporate bonds, purchasing the  investment-grade securities in primary and secondary markets and through  exchange-traded funds. The move comes in a space that has seen considerable  turmoil since the crisis has intensified and market liquidity has been  sapped._Markets initially reacted positively to the moves but headed back  lower in early trading, with the Dow Jones Industrial Average down 260  points.Other initiatives include an unspecified lending program for Main  Street businesses and the Term Asset-Backed Loan Facility implemented during  the financial crisis. There will be a program worth $300 billion  "supporting the flow of credit" to employers consumers and  businesses and two facilities set up to provide credit to large  employers.There are no details yet on the Main Street program, with a news  release saying it will help "support lending to eligible small_and-medium  sized businesses, complementing efforts by the SBA."The Fed also said it  will purchase agency commercial mortgage-backed securities as part of an  expansion in its asset purchases, known in the market as quantitative easing.  The move represents an expansion into the commercial sector of real estate  for the central bank's acquisitions."We are now in QE infinity,  again," Peter Boockvar, chief investment officer at Bleakley Advisory  Group, said in a note.Additional measures include the issuance of  asset-backed securities backed by student loans, auto loans, credit card  loans, loans guaranteed by the Small Business Administration and certain  other assets._The moves come on top of programs the central bank announced  last week aimed at easing the flow of credit markets and the short-term  funding banks need to operate. The Fed said it will expand its money market  facility announced last week to include a wider range of securities that it  will accept."The coronavirus pandemic is causing tremendous hardship  across the United States and around the world. Our nation's first priority is  to care for those afflicted and to limit the further spread of the  virus," the Fed said in a statement. "While great uncertainty  remains, it has become clear that our economy will face severe disruptions.  Aggressive efforts must be taken across the public and private sectors to  limit the losses to jobs and incomes and to promote a swift recovery once the  disruptions abate."Monday's announcement represents the most aggressive  market intervention the Fed has made to date.Previously, it had announced it  would buy $500 billion worth of Treasurys and $200 billion in mortgage-backed  securities. The new move represents an open-ended commitment to the QE  program."Fed policy is shifting into a higher gear to try to help  support the economy which looks like it is in freefall at the moment,"  wrote Chris Rupkey, chief finacial economist at MUFG Union Bank. "The  central bank is shifting from being not just the lender of last resort, but  now it is the buyer of last resort. Don't ask how much they will buy, this is  truly QE infinity."The Fed announced it also is expanding its Commercial  Paper Funding Facility. The program now will include "high-quality,  tax-exempt commercial paper" and the pricing will be reduced._The  central bank also said it will lower the interest rate on its repo operations  to 0% from 0.1%. The operations are conducted daily to provide banks short-term  funding.The programs are backed by the Treasury Department to ensure the Fed  does not lose money."We are committed to providing relief for American  workers and businesses, particularly small and medium size businesses and  critical industries that are most impacted by the coronavirus. We will take  all necessary steps to support them and protect the U.S. economy,"  Treasury Secretary Steven Mnuchin said in a statement. |



## 3) 데이터 전처리

* [preprocessing](https://github.com/young-o/text_summarization_project/blob/master/code/preprocessing.py)

<img src="https://user-images.githubusercontent.com/46865281/80798403-459b6080-8bdf-11ea-8f79-8219c923908e.png" alt="image" style="zoom:25%;" />
<img src="https://user-images.githubusercontent.com/46865281/80798409-48965100-8bdf-11ea-9662-8fd3c4ec9aeb.png" alt="image" style="zoom:25%;" />



### 정수 인코딩

### train, test 데이터셋 분리 및 패딩



## 3) seq2seq 모델, attention 함수 설계 및 학습

* attention 함수 논문 조사해서 선별 후 고를 예정
  * [Bahdanau Attention](https://github.com/thushv89/attention_keras)



## 4) 모델 테스트 및 성능 파악

* [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))



## 5) 모델 개선 방향 조사

* 전처리 과정 개선?
  * 워드 임베딩
  * pre-trained 임베딩 벡터
* 모델 튜닝
* 모델 변경
  * transformer

