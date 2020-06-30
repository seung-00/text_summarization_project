import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import tensorflow
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup


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
        sentence = str(sentence)
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
        preprocess data and save as csv for auto summarization with seq2seq
        
        parameters:
        data(dataframe): dataframe with 2 level, source and summary data

        returns:
        void 
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
        print(self.data.isnull().sum())        # # src     0
        #
        #  smry    0

        # 모델 패딩을 위해 각 데이터 길이 분포 시각화
        # read README˜

        src_max_len = 700
        smry_max_len = 100

        self.data = self.data[self.data['src'].apply(lambda x: len(x.split()) <= src_max_len)]
        self.data = self.data[self.data['smry'].apply(lambda x: len(x.split()) <= smry_max_len)]

        # print(self.data.shape)
        # (3510, 2)

        self.data['smry'] = self.data['smry'].apply(lambda x : 'sostoken '+ x + ' eostoken')
        self.data.to_csv(path+"/data/cleaned_sample2.csv", mode='w')



if __name__ == "__main__":
    path = "/Users/seungyoungoh/workspace/text_summarization_project"
    data = pd.read_csv(path+"/data/sample2.csv", error_bad_lines = False)
    
    pr = Preprocessor(data)
    pr.preprocess()
    
