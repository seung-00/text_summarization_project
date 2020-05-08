## 데이터 전처리

### 기본적인 전처리

* 정규화

  ```python
  # normalization
  contractions = contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
  ```

* 불용어 처리

  ```python
  from nltk.corpus import stopwords
  stop_words = set(stopwords.words('english'))
  ```

* 전처리 함수

  ```python
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
  ```



### 텍스트 길이 분포 확인 및 시각화

* 모델 패딩을 위해 데이터 길이 분포 확인 및 시각화

  ```python
  src_len = [len(s.split()) for s in data['src']]
  smry_len = [len(s.split()) for s in data['smry']]
  
  print(f"length of src\nmin: {np.min(src_len)}, max: {np.max(src_len)}, mean: {np.mean(src_len)}")
  length of src
  # min: 1, max: 3093, mean: 346.63057829416425
  
  print(f"length of smry\nmin: {np.min(smry_len)}, max: {np.max(smry_len)}, mean: {np.mean(smry_len)}")
  length of smry
  # min: 2, max: 131, mean: 59.418008978082916
  
  plt.subplot(1,2,1)
  plt.boxplot(smry_len)
  plt.title('smry')
  plt.subplot(1,2,2)
  plt.boxplot(src_len)
  plt.title('src')
  plt.tight_layout()
  plt.show()
  
  
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
  ```

  <img src="https://user-images.githubusercontent.com/46865281/81400124-662b6380-9167-11ea-92d0-5e0353bd1605.png" width="400" height="300">

  <img src="https://user-images.githubusercontent.com/46865281/81400215-8d823080-9167-11ea-9703-1ad2bd4cc946.png" width="400" height="300">

  <img src="https://user-images.githubusercontent.com/46865281/81400360-d5a15300-9167-11ea-9a69-92dcc72ccd08.png" width="400" height="300">



### 모델 사이즈에 맞게 데이터 핸들링

```python
src_max_len = 700, smry_max_len = 100

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s.split()) <= max_len):
            cnt = cnt + 1
    print(f'전체 샘플 중 길이가 {max_len} 이하인 샘플의 비율: {cnt / len(nested_list)}')

below_threshold_len(src_max_len, data['src'])
# 전체 샘플 중 길이가 700 이하인 샘플의 비율: 0.9456033799841563
below_threshold_len(smry_max_len, data['smry'])
# 전체 샘플 중 길이가 100 이하인 샘플의 비율: 0.9786110377607605
```

```python
data = data[data['src'].apply(lambda x: len(x.split()) <= src_max_len)]
data = data[data['smry'].apply(lambda x: len(x.split()) <= smry_max_len)]

print(data.shape)
# (3510, 2)
```



### 요약 데이터에 시작, 종료 토큰 추가

```python
data['smry'] = data['smry'].apply(lambda x : 'sostoken '+ x + ' eostoken')
data.head()
```

| src                                               | smry                                              |
| ------------------------------------------------- | ------------------------------------------------- |
| oil market facing uncharted territory drop dem... | sostoken the oil market is facing uncharted te... |
| saudi arabia best positioned weather impact un... | sostoken on monday the may contract for west t... |
| oil futures contract made historic plunge west... | sostoken an oil futures contract went negative... |
| coronavirus pandemic threat airline bankruptci... | sostoken jet fuel prices have fallen faster th... |
| oil industry worst crisis since least great de... | sostoken the oil industry is in crisis like it... |



### partitiotn

```python
X_train, X_test, y_train, y_test = train_test_split(src, smry, test_size=0.2, random_state=0, shuffle=True)
```



### 정수 인코딩

* 토크나이징 한 뒤 출현 빈도가 낮은 단어들 파악

  ```python
  ## src data
  src_tokenizer = Tokenizer()
  # fit_on_texts(corpus) generates a set of words based on fequency
  src_tokenizer.fit_on_texts(X_train)
  
  threshold = 6
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
  
  print(f"set of vocabulary size : {total_src_cnt}")
  # set of vocabulary size : 30087
  print(f"rare words(count<{threshold}) count: {rare_src_cnt}")
  # rare words(count<6) count: 19262
  print(f"set of vocabulary except rare word size: {total_src_cnt - rare_src_cnt}")
  # set of vocabulary except rare word size: 10825
  print(f"percentage of rare words: {(rare_src_cnt / total_src_cnt)*100}")
  # percentage of rare words: 64.02100574999169
  print(f"percentage of rare words frequency from total frequency: {(rare_src_freq / total_src_freq)*100}")
  # percentage of rare words frequency from total frequency: 4.36439810550019
  ```

  ```python
  ## smry data
  smry_tokenizer = Tokenizer()
  smry_tokenizer.fit_on_texts(y_train)
  
  threshold = 4
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
  
  print(f"set of vocabulary size : {total_smry_cnt}")
  # set of vocabulary size : 11769
  print(f"rare words(count<{threshold}) count: {rare_smry_cnt}")
  # rare words(count<6) count: 7531
  print(f"set of vocabulary except rare word size: {total_smry_cnt - rare_smry_cnt}")
  # set of vocabulary except rare word size: 4238
  print(f"percentage of rare words: {(rare_smry_cnt / total_smry_cnt)*100}")
  # percentage of rare words: 63.99014359758688
  print(f"percentage of rare words frequency from total frequency: {(rare_smry_freq / total_smry_freq)*100}")
  # percentage of rare words frequency from total frequency: 6.748086145780584
  ```

* 빈도 낮은 단어들 제거

  ```python
  src_vocab = total_src_cnt - rare_src_cnt
  # 상위 10825 단어만 사용
  src_tokenizer = Tokenizer(num_words = src_vocab+1) 
  src_tokenizer.fit_on_texts(X_train)
  
  # text to int sequences
  X_train = src_tokenizer.texts_to_sequences(X_train) 
  X_test = src_tokenizer.texts_to_sequences(X_test)
  
  smry_vocab = total_smry_cnt - rare_smry_cnt
  # 상위 4238 단어만 사용
  smry_tokenizer = Tokenizer(num_words = smry_vocab+1) 
  smry_tokenizer.fit_on_texts(y_train)
  
  y_train = smry_tokenizer.texts_to_sequences(y_train) 
  y_test = smry_tokenizer.texts_to_sequences(y_test) 
  ```

* 빈 요약 데이터(앞에 추가한 토큰 둘 밖에 안 남았을 경우) 제거

  ```python
  # delete empty samples
  drop_train = [index for index, sentence in enumerate(y_train) if len(sentence) == 2]
  drop_test = [index for index, sentence in enumerate(y_test) if len(sentence) == 2]
  
  X_train = np.delete(X_train, drop_train, axis=0)
  y_train = np.delete(y_train, drop_train, axis=0)
  X_test = np.delete(X_test, drop_test, axis=0)
  y_test = np.delete(y_test, drop_test, axis=0)
  
  # 2808 -> 2807
  # 2808 -> 2807
  # 702 -> 702
  # 702 -> 702
  ```



### 패딩

```python
X_train = pad_sequences(X_train, maxlen = src_max_len, padding='post')
X_test = pad_sequences(X_test, maxlen = src_max_len, padding='post')
y_train = pad_sequences(y_train, maxlen = smry_max_len, padding='post')
y_test = pad_sequences(y_test, maxlen = smry_max_len, padding='post')
```

### 