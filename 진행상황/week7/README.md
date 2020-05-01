> 2020년 2학기 데이터캡스톤 term-project 진행사항 문서입니다.

# 7주차 활동 상황 보고

seq2seq 와 어텐션 메커니즘을 이용한 텍스트 요약



## 1) 데이터 수집

### cnbc 뉴스 데이터 크롤러

* [크롤러](https://github.com/SeungYoungOh/cnbc_crawler)

* 병렬 처리
* 크롤러 차단 이슈 해결
* 약 8000개 데이터 파싱



## 2) 데이터 정제(cleaning)

* 뉴스 corpus 로부터 noise 제거
* 대략 3800개 우선 이걸로 ok

```python
import numpy as np
import pandas as pd
path = "/Users/seungyoungoh/workspace/text_summarization_project/"
data = pd.read_csv(path+"/data/news_data.csv", error_bad_lines = False, encoding = 'cp949')

data.columns
# Index(['Unnamed: 0', 'key_point', 'body', 'category', 'url'], dtype='object')

news = data[['key_point', 'body']]
len(news)
# 7377

news.dropna(axis=0, inplace=True)
len(news)
# 5235

print(data['key_point'].nunique())
# 3787
news.drop_duplicates(subset=['key_point'], inplace=True)
len(news)
# 3787

print(data.isnull().sum())
# key_point    0
# body         0

s = news.sample(1)
news.to_csv(path+"/data/sample.csv", mode='w')
```

| key_point                                                    | body                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| The Federal Reserve announced a barrage of new programs to help keep the  market functioning.Among the moves is an open-ended commitment to keep buying  assets under its quantitative easing measures.There are multiple other  programs, including one for Main Street business lending and others aimed at  keeping credit flowing.The Fed will be moving for the first time into  corporate bonds, purchasing the investment-grade securities in primary and  secondary markets and through exchange-traded funds. | The Federal Reserve said Monday it will launch a barrage of programs  aimed at helping markets function more efficiently amid the coronavirus  crisis.Among the initiatives is a commitment to continue its asset purchasing  program "in the amounts needed to support smooth market functioning and  effective transmission of monetary policy to broader financial conditions and  the economy."That represents a potentially new chapter in the Fed's  "money printing" as it commits to keep expanding its balance sheet  as necessary, rather than a commitment to a set amount.The Fed also will be  moving for the first time into corporate bonds, purchasing the  investment-grade securities in primary and secondary markets and through  exchange-traded funds. The move comes in a space that has seen considerable  turmoil since the crisis has intensified and market liquidity has been  sapped._Markets initially reacted positively to the moves but headed back  lower in early trading, with the Dow Jones Industrial Average down 260  points.Other initiatives include an unspecified lending program for Main  Street businesses and the Term Asset-Backed Loan Facility implemented during  the financial crisis. There will be a program worth $300 billion  "supporting the flow of credit" to employers consumers and  businesses and two facilities set up to provide credit to large  employers.There are no details yet on the Main Street program, with a news  release saying it will help "support lending to eligible small_and-medium  sized businesses, complementing efforts by the SBA."The Fed also said it  will purchase agency commercial mortgage-backed securities as part of an  expansion in its asset purchases, known in the market as quantitative easing.  The move represents an expansion into the commercial sector of real estate  for the central bank's acquisitions."We are now in QE infinity,  again," Peter Boockvar, chief investment officer at Bleakley Advisory  Group, said in a note.Additional measures include the issuance of  asset-backed securities backed by student loans, auto loans, credit card  loans, loans guaranteed by the Small Business Administration and certain  other assets._The moves come on top of programs the central bank announced  last week aimed at easing the flow of credit markets and the short-term  funding banks need to operate. The Fed said it will expand its money market  facility announced last week to include a wider range of securities that it  will accept."The coronavirus pandemic is causing tremendous hardship  across the United States and around the world. Our nation's first priority is  to care for those afflicted and to limit the further spread of the  virus," the Fed said in a statement. "While great uncertainty  remains, it has become clear that our economy will face severe disruptions.  Aggressive efforts must be taken across the public and private sectors to  limit the losses to jobs and incomes and to promote a swift recovery once the  disruptions abate."Monday's announcement represents the most aggressive  market intervention the Fed has made to date.Previously, it had announced it  would buy $500 billion worth of Treasurys and $200 billion in mortgage-backed  securities. The new move represents an open-ended commitment to the QE  program."Fed policy is shifting into a higher gear to try to help  support the economy which looks like it is in freefall at the moment,"  wrote Chris Rupkey, chief finacial economist at MUFG Union Bank. "The  central bank is shifting from being not just the lender of last resort, but  now it is the buyer of last resort. Don't ask how much they will buy, this is  truly QE infinity."The Fed announced it also is expanding its Commercial  Paper Funding Facility. The program now will include "high-quality,  tax-exempt commercial paper" and the pricing will be reduced._The  central bank also said it will lower the interest rate on its repo operations  to 0% from 0.1%. The operations are conducted daily to provide banks short-term  funding.The programs are backed by the Treasury Department to ensure the Fed  does not lose money."We are committed to providing relief for American  workers and businesses, particularly small and medium size businesses and  critical industries that are most impacted by the coronavirus. We will take  all necessary steps to support them and protect the U.S. economy,"  Treasury Secretary Steven Mnuchin said in a statement. |



## 3) 데이터 전처리

```python
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup 
import re
import matplotlib.pyplot as plt

path = "/Users/seungyoungoh/workspace/text_summarization_project/"
data = pd.read_csv(path+"/data/sample.csv", error_bad_lines = False)
data = data.rename({'body':'src', 'key_point':'smry'}, axis = 'columns')[['src','smry']]
data.shape
# (3787, 2)

# normalization
contractions = contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

# stop words
stop_words = set(stopwords.words('english'))

def preprocess_sentence(sentence, remove_stopwords = True):
    sentence = sentence.lower()
    sentence = BeautifulSoup(sentence, "lxml").text # remove html tags
    sentence = re.sub(r'\([^)]*\)', '', sentence) # remove useless str
    sentence = re.sub('"','', sentence) # remove quotation mark
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")]) # normalization, '' : sep
    sentence = re.sub(r"'s\b","",sentence) 
    # remove possessive . ex) roland's -> roland
    # \b means boundary of word
    sentence = re.sub("[^a-zA-Z]", " ", sentence) # convert non-English character to space
    sentence = re.sub('[m]{2,}', 'mm', sentence) #ex) ummmmmmm yeah -> umm yeah
    # remove stopwords, str of 1 length from the key points
    if remove_stopwords:
        tokens = ' '.join(word for word in sentence.split() if (not word in stop_words and len(word) > 1))
    # remove just str of 1 length from the summary
    else:
        tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    return tokens

clean_src = []
for s in data['src']:
    clean_src.append(preprocess_sentence(s))

clean_smry = []
for s in data['smry']:
    clean_smry.append(preprocess_sentence(s, 0))

data['src'] = clean_src
data['smry'] = clean_smry

# null check
data.replace('', np.nan, inplace=True)
print(data.isnull().sum())
# src     0
# smry    0

# length dist
src_len = [len(s.split()) for s in data['src']]
smry_len = [len(s.split()) for s in data['smry']]

print(f"length of src\nmin: {np.min(src_len)}, max: {np.max(src_len)}, mean: {np.mean(src_len)}")
# length of src
# min: 1, max: 3093, mean: 346.63057829416425

print(f"length of smry\nmin: {np.min(smry_len)}, max: {np.max(smry_len)}, mean: {np.mean(smry_len)}")
# length of smry
# min: 2, max: 131, mean: 59.418008978082916

plt.title('src')
plt.hist(src_len, bins=40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

plt.title('smry')
plt.hist(smry_len, bins=40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


data.to_csv(path+"/data/cleaned_sample.csv", mode='w')
```

<img src="https://user-images.githubusercontent.com/46865281/80798403-459b6080-8bdf-11ea-8f79-8219c923908e.png" alt="image" style="zoom:25%;" />
<img src="https://user-images.githubusercontent.com/46865281/80798409-48965100-8bdf-11ea-9662-8fd3c4ec9aeb.png" alt="image" style="zoom:25%;" />



### 정수 인코딩



### train, test 데이터셋 분리 및 패딩





## 3) seq2seq 모델, attention 함수 설계 및 학습

* attention 함수 논문 조사해서 선별 후 고를 예정



## 4) 모델 테스트 및 성능 파악



## 5) 모델 개선 방향 조사

* 전처리 과정 개선?
  * 워드 임베딩
  * pre-trained 임베딩 벡터
* 모델 튜닝
* 모델 변경
  * transformer

