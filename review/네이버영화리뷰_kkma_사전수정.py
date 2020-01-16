from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from konlpy.tag import Kkma
from konlpy.tag import Okt
import pandas as pd
import numpy as np
import requests
import pickle
import json
import re

train_data = pd.read_table('ratings_train.txt')
test_data= pd.read_table('ratings_test.txt')

train_data=train_data.dropna(how='any')

f = lambda x:" ".join(x for x in x.split())
train_data['document'] = train_data['document'].apply(f)
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

f = lambda x:" ".join(x for x in x.split())
train_data['document'] = train_data['document'].apply(f)
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

blank_index = train_data.loc[train_data['document'] == ''].index
train_data = train_data.drop(blank_index)
print("train data length:", len(train_data))

index_list = train_data['document'].index.tolist()

print(index_list[40223])
print(train_data['document'][40556])
train_data = train_data.drop(40556)

print(index_list[124755])
print(train_data['document'][125801])
train_data = train_data.drop(125801)

print(train_data['document'][index_list[113426]])
train_data = train_data.drop(index_list[113426])

print("train data length:", len(train_data))

test_data=test_data.dropna(how='any')

f = lambda x:" ".join(x for x in x.split())
test_data['document'] = test_data['document'].apply(f)
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

f = lambda x:" ".join(x for x in x.split())
test_data['document'] = test_data['document'].apply(f)
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

blank_index = test_data.loc[test_data['document'] == ''].index
test_data = test_data.drop(blank_index)

print("test data length:", len(test_data))

index_list = test_data['document'].index.tolist()

print(test_data['document'][index_list[33152]])
test_data = test_data.drop(index_list[33152])

print(test_data['document'][index_list[35963]])
test_data = test_data.drop(index_list[35963])

print(test_data['document'][index_list[35965]])
test_data = test_data.drop(index_list[35965])

print(test_data['document'][index_list[34495]])
test_data = test_data.drop(index_list[34495])

print("test data length:", len(test_data))

stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
kkma = Kkma()

X_train=[]
cnt = 0
for sentence in train_data['document']:
    print(sentence, cnt)
    temp_X = []
    temp_X=kkma.morphs(sentence)
    temp_X=[word for word in temp_X if not word in stopwords]
    X_train.append(temp_X)
    cnt += 1

print("***********************************************************")

X_test=[]
cnt = 0
for sentence in test_data['document']:
    print(sentence, cnt)
    temp_X = []
    temp_X=kkma.morphs(sentence) # 토큰화
    temp_X=[word for word in temp_X if not word in stopwords] # 불용어 제거
    X_test.append(temp_X)
    cnt += 1

max_words = 35000
tokenizer = Tokenizer(num_words=max_words) # 상위 35,000개의 단어만 보존
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

max_len=30
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

y_train=np.array(train_data['label'])
y_test=np.array(test_data['label'])

classifier = MultinomialNB()
targets = y_train

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))