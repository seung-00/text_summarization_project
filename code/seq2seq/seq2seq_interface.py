import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[3], 'GPU')


data = pd.read_csv("/home/data/cleaned_sample.csv", error_bad_lines = False)
data = data[['Text','Summary']]
data = data.rename(columns = {"Text":"src","Summary":"smry"})

# 모델 패딩을 위해 각 데이터 길이 분포 시각화
# read README˜

src_max_len = 50
smry_max_len = 8
src = list(data['src'])
smry = list(data['smry'])

# partitiotn
X_train, X_test, y_train, y_test = train_test_split(src, smry, test_size=0.2, random_state=0, shuffle=True)

### Integer Encoding
src_vocab = 8000
src_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = src_vocab)
src_tokenizer.fit_on_texts(X_train)

smry_vocab = 2000
smry_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = smry_vocab)
smry_tokenizer.fit_on_texts(y_train)


reverse_target_word_index = smry_tokenizer.index_word # 요약 단어 집합에서 정수 -> 단어를 얻음
reverse_source_word_index = src_tokenizer.index_word # 원문 단어 집합에서 정수 -> 단어를 얻음
target_word_index = smry_tokenizer.word_index # 요약 단어 집합에서 단어 -> 정수를 얻음


encoder_model= tf.keras.models.load_model('encoder.h5')
decoder_model= tf.keras.models.load_model('decoder.h5')

def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    e_out, e_h, e_c = encoder_model.predict(input_seq)

# 요 부분 공부 필요!
    # Generate empty target sequence of length 1.
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = target_word_index['sostoken']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        generated_token_idx = np.argmax(output_tokens[0, -1, :])
        generated_token_char = reverse_target_word_index[generated_token_idx]

# 조건문 코드 개선 생각!
        if(generated_token_char!='eostoken'):
            decoded_sentence += ' '+generated_token_char

        # Exit condition: either hit max length or find stop word.
        if (generated_token_char == 'eostoken'  or len(decoded_sentence.split()) >= (smry_max_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = generated_token_idx

        # Update internal states
        e_h, e_c = h, c
    return decoded_sentence

input_str = input("enter the review: \t")
tokenized_src = src_tokenizer.texts_to_sequences([input_str])
tokenized_src = tf.keras.preprocessing.sequence.pad_sequences(tokenized_src, maxlen = src_max_len, padding='post')
predicted_line = decode_sequence(tokenized_src.reshape(1,src_max_len))

#"Wow, it's really delicious. It's the best. I want to eat it every day. A restaurant that you want to recommend to people around you!"