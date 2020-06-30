import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[3], 'GPU')

def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    e_out, e_h, e_c = encoder_model.predict(input_seq)
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

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostoken']) and i!=target_word_index['eostoken']):
            newString = newString + reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


data = pd.read_csv("/home/data/cleaned_sample.csv", error_bad_lines = False)
data = data[['Text','Summary']]
data = data.rename(columns = {"Text":"src","Summary":"smry"})
print(data.columns)
print(data.shape)

src_max_len = 50
smry_max_len = 8
src = list(data['src'])
smry = list(data['smry'])

# partitiotn
X_train, X_test, y_train, y_test = train_test_split(src, smry, test_size=0.2, random_state=0, shuffle=True)
X_train_word, X_test_word, y_train_word, y_test_word = X_train, X_test, y_train, y_test


### Integer Encoding
## src data
src_tokenizer = tf.keras.preprocessing.text.Tokenizer()
# fit_on_texts(corpus) generates a set of words based on fequency
src_tokenizer.fit_on_texts(X_train)

threshold = 7
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

## smry data
smry_tokenizer = tf.keras.preprocessing.text.Tokenizer()
smry_tokenizer.fit_on_texts(y_train)

threshold = 6
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


#src_vocab = total_src_cnt - rare_src_cnt
src_vocab = 8000
src_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = src_vocab)
src_tokenizer.fit_on_texts(X_train)

# text to int sequences
X_train = src_tokenizer.texts_to_sequences(X_train)
X_test = src_tokenizer.texts_to_sequences(X_test)

#smry_vocab = total_smry_cnt - rare_smry_cnt
smry_vocab = 2000
smry_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = smry_vocab)
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

### padding
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen = src_max_len, padding='post')
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen = src_max_len, padding='post')
y_train = tf.keras.preprocessing.sequence.pad_sequences(y_train, maxlen = smry_max_len, padding='post')
y_test = tf.keras.preprocessing.sequence.pad_sequences(y_test, maxlen = smry_max_len, padding='post')


reverse_target_word_index = smry_tokenizer.index_word # 요약 단>어 집합에서 정수 -> 단어를 얻음
reverse_source_word_index = src_tokenizer.index_word # 원문 단어
target_word_index = smry_tokenizer.word_index


# encoder1
encoder_model= tf.keras.models.load_model('/.../src/seq2seq/case1/encoder1.h5')
decoder_model= tf.keras.models.load_model('/.../src/seq2seq/case1/decoder1.h5')

referenece_f1 = open("/.../data/no_attention/case1/reference/reference.txt", mode = 'w')
system_f1 = open("/.../data/no_attention/case1/system/system.txt", mode = 'w')

for i in range(0, len(X_test)):
    print("원문 : ",seq2text(X_test[i]))

    y_line = seq2summary(y_test[i])
    print("실제 요약문 :",y_line)
    referenece_f1.write(y_line+'\n')

    predicted_line = decode_sequence(X_test[i].reshape(1,src_max_len))
    print("예측 요약문 :",predicted_line)
    system_f1.write(predicted_line+"\n")
    print("\n")

referenece_f1.close()
system_f1.close()

# encoder2
encoder_model= tf.keras.models.load_model('/.../src/seq2seq/case2/encoder2.h5')
decoder_model= tf.keras.models.load_model('/.../src/seq2seq/case2/decoder2.h5')

referenece_f2 = open("/.../data/no_attention/case2/reference/reference.txt", mode = 'w')
system_f2 = open("/.../data/no_attention/case2/system/system.txt", mode = 'w')

for i in range(0, len(X_test)):
    print("원문 : ",seq2text(X_test[i]))

    y_line = seq2summary(y_test[i])
    print("실제 요약문 :",y_line)
    referenece_f2.write(y_line+'\n')

    predicted_line = decode_sequence(X_test[i].reshape(1,src_max_len))
    print("예측 요약문 :",predicted_line)
    system_f2.write(predicted_line+"\n")
    print("\n")

referenece_f2.close()
system_f2.close()

# encoder3
encoder_model= tf.keras.models.load_model('/.../src/seq2seq/case3/encoder3.h5')
decoder_model= tf.keras.models.load_model('/.../src/seq2seq/case3/decoder3.h5')

referenece_f3 = open("/.../data/no_attention/case3/reference/reference.txt", mode = 'w')
system_f3 = open("/.../data/no_attention/case3/system/system.txt", mode = 'w')

for i in range(0, len(X_test)):
    print("원문 : ",seq2text(X_test[i]))

    y_line = seq2summary(y_test[i])
    print("실제 요약문 :",y_line)
    referenece_f3.write(y_line+'\n')

    predicted_line = decode_sequence(X_test[i].reshape(1,src_max_len))
    print("예측 요약문 :",predicted_line)
    system_f3.write(predicted_line+"\n")
    print("\n")

referenece_f3.close()
system_f3.close()