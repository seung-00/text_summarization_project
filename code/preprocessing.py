import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import tensorflow as tf
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup 
import re
# import matplotlib.pyplot as plt
keras = tf.keras
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

class Preprocessor:
    def __init__(self, data):
        self.data = data
        # normalization
        self.contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
        # stop words
        self.stop_words = set(stopwords.words('english'))

    def preprocess_sentence(self, sentence, remove_stopwords = True):
        """
        brief:
        preprocess sentence
        
        parameters:
        sentence(string): raw data

        returns:
        tokens(string): cleansed data
        """
        sentence = sentence.lower()
        sentence = BeautifulSoup(sentence, "lxml").text # remove html tags
        sentence = re.sub(r'\([^)]*\)', '', sentence) # remove useless str
        sentence = re.sub('"','', sentence) # remove quotation mark
        sentence = ' '.join([self.contractions[t] if t in self.contractions else t for t in sentence.split(" ")]) # normalization, '' : sep
        sentence = re.sub(r"'s\b","",sentence) 
        # remove possessive . ex) roland's -> roland
        # \b means boundary of word
        sentence = re.sub("[^a-zA-Z]", " ", sentence) # convert non-English character to space
        sentence = re.sub('[m]{2,}', 'mm', sentence) #ex) ummmmmmm yeah -> umm yeah
        # remove stopwords, str of 1 length from the key points
        if remove_stopwords:
            tokens = ' '.join(word for word in sentence.split() if (not word in self.stop_words and len(word) > 1))
        # remove just str of 1 length from the summary
        else:
            tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
        return tokens


    def preprocess(self):
        """
        brief:
        preprocess data for auto summarization with seq2seq
        
        parameters:
        data(dataframe): dataframe with 2 level, source and summary data

        returns:
        src_max_len, smry_max_len, src_vocab, smry_vocab(int)
        X_train, X_test, y_train, y_test(numpy.ndarray)
        """

        clean_src = []
        for s in self.data['src']:
            clean_src.append(self.preprocess_sentence(s))

        clean_smry = []
        for s in self.data['smry']:
            clean_smry.append(self.preprocess_sentence(s, 0))

        self.data['src'] = clean_src
        self.data['smry'] = clean_smry

        # check null
        self.data.replace('', np.nan, inplace=True)
        print(self.data.isnull().sum())
        # src     0
        # smry    0

        # 모델 패딩을 위해 각 데이터 길이 분포 시각화
        # read README

        src_max_len = 700
        smry_max_len = 100

        self.data = self.data[self.data['src'].apply(lambda x: len(x.split()) <= src_max_len)]
        self.data = self.data[self.data['smry'].apply(lambda x: len(x.split()) <= smry_max_len)]

        # print(self.data.shape)
        # (3510, 2)

        self.data['smry'] = self.data['smry'].apply(lambda x : 'sostoken '+ x + ' eostoken')
        self.data.head()

        src = list(self.data['src'])
        smry = list(self.data['smry'])

        # partitiotn
        X_train, X_test, y_train, y_test = train_test_split(src, smry, test_size=0.2, random_state=0, shuffle=True)


        ### Integer Encoding

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

        print(f"set of vocabulary except rare word size: {total_src_cnt - rare_src_cnt}")
        # set of vocabulary except rare word size: 10825
        print(f"percentage of rare words frequency from total frequency: {(rare_src_freq / total_src_freq)*100}")
        # percentage of rare words frequency from total frequency: 4.36439810550019

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

        print(f"set of vocabulary except rare word size: {total_smry_cnt - rare_smry_cnt}")
        # set of vocabulary except rare word size: 4238
        print(f"percentage of rare words frequency from total frequency: {(rare_smry_freq / total_smry_freq)*100}")
        # percentage of rare words frequency from total frequency: 6.748086145780584

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

        ### padding
        X_train = pad_sequences(X_train, maxlen = src_max_len, padding='post')
        X_test = pad_sequences(X_test, maxlen = src_max_len, padding='post')
        y_train = pad_sequences(y_train, maxlen = smry_max_len, padding='post')
        y_test = pad_sequences(y_test, maxlen = smry_max_len, padding='post')
        print(type(X_train))
        return src_max_len, smry_max_len, src_vocab, smry_vocab, X_train, X_test, y_train, y_test
