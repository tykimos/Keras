---
layout: post
title:  "순환 신경망 레이어 이야기"
author: 김태영
date:   2017-04-09 04:00:00
categories: Lecture
comments: true
image: http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_10.png
---
순환 신경망 모델에서 주로 사용하는 LSTM 레이어에 대해서 알아보겠습니다. 시퀀스 데이터를 다루는 레이어라 설정 파라미터에 따라 다양하게 모델을 구성할 수 있습니다. 그만큼 헷갈리는 부분도 있지만 "나비야" 동요를 학습시켜보면서 차근히 살펴보겠습니다.

---

### 긴 시퀀스를 기억할 수 있는 LSTM (Long Short-Term Memory units)  레이어

케라스에서 제공하는 순환 신경망 레이어는 SimpleRNN, GRU, LSTM이 있으나 주로 사용하는 LSTM에 대해서 알아보겠습니다. 이 LSTM은 아래와 같이 간단히 사용할 수 있습니다.

#### 입력 형태

    LSTM(3, input_dim=1)

기본 인자는 다음과 같습니다.
* 첫번째 인자 : 출력 뉴런의 수 입니다.
* input_dim : 입력 뉴런의 수 입니다.

이는 앞서 살펴본 Dense 레이어 형태와 비슷합니다. input_dim에는 Dense 레이어와 같이 일반적으로 속성의 개수가 들어갑니다. 

    Dense(3, input_dim=1)

LSTM의 한 가지 인자에 대해 더 알아보겠습니다.

    LSTM(3, input_dim=1, input_length=4)

* input_length : 시퀀스 데이터의 입력 길이

Dense와 LSTM을 레고 블록으로 도식화 하면 다음과 같습니다. 왼쪽이 Dense이고, 중앙이 input_length가 1인 LSTM이고 오른쪽이 input_length가 4인 LSTM 입니다. 사실 LSTM의 내부구조는 복잡하지만 간소화하여 외형만 표시한 것입니다. Dense 레이어와 비교한다면 히든 뉴런들이 밖으로 도출되어 있음을 보실 수 있습니다. 그리고 오른쪽 블럭인 경우 input_length가 길다고 해서 각 입력마다 다른 가중치를 사용하는 것이 아니라 중앙에 있는 블럭을 입력 길이 만큼 연결한 것이기 때문에 모두 동일한 가중치를 공유합니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_LSTM1.png)

#### 출력 형태

* return_sequences : 시퀀스 출력 여부

LSTM 레이어는 return_sequences 인자에 따라 마지막 시퀀스에서 한 번만 출력할 수 있고 각 시퀀스에서 출력을 할 수 있습니다. many to many 문제를 풀거나 LSTM 레이어를 여러개로 쌓아올릴 때는 return_sequence=True 옵션을 사용합니다. 자세한 것은 뒤에서 살펴보겠습니다. 아래 그림에서 왼쪽은 return_sequences=False일 때, 오른쪽은 return_sequence=True일 때의 형상입니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_LSTM2.png)

#### 상태유지(stateful) 모드

* stateful : 상태 유지 여부

학습 샘플의 가장 마지막 상태가 다음 샘플 학습 시에 입력으로 전달 여부를 지정하는 것입니다. 하나의 샘플은 4개의 시퀀스 입력이 있고, 총 3개의 샘플이 있을 때, 아래 그림에서 위의 블럭들은 stateful=False일 때의 형상이고, 아래 블럭들은 stateful=True일 때의 형상입니다. 도출된 현재 상태의 가중치가 다음 샘플 학습 시의 초기 상태로 입력됨을 알 수 있습니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_LSTM3.png)

---

### 시퀀스 데이터 준비

순환 신경망은 주로 자연어 처리에 많이 쓰이기 때문에 문장 학습 예제가 일반적이지만 본 강좌에서는 악보 학습을 해보겠습니다. 그 이유는 
- 음계가 문장보다 더 코드화 하기 쉽고, 
- 시계열 자료이며, 
- 나온 결과를 악보로 볼 수 있으며,
- 무엇보다 우리가 학습한 모델이 연주하는 곡을 들어볼 수 있기 때문입니다. 
일단 쉬운 악보인 '나비야'를 준비했습니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_2.png)

음표 밑에 간단한 음표코드를 표시하였습니다. 알파벳은 음계를 나타내며, 숫자는 음의 길이를 나타냅니다.
- c(도), d(레), e(미), f(파), g(솔), a(라), b(시)
- 4(4분음표), 8(8분음표)

---

### 데이터셋 생성

먼저 두 음절만 살펴보겠습니다. 

* g8 e8 e4 | f8 d8 d4 | 

여기서 우리가 정의한 문제대로 4개 음표 입력으로 다음 출력 음표를 예측하려면, 아래와 같이 데이터셋을 구성합니다.

* g8 e8 e4 f8 d8 : 1~4번째 음표, 5번째 음표
* e8 e4 f8 d8 d4 : 2~5번째 음표, 6번째 음표

6개의 음표로는 위와 같이 2개의 샘플이 나옵니다. 각 샘플은 4개의 입력 데이터와 1개의 라벨값으로 구성되어 있습니다. 즉 1~4번째 열은 속성(feature)이고, 5번째 열은 클래스(class)를 나타냅니다. 이렇게 4개씩 구간을 보는 것을 윈도우 크기가 4라고 합니다. 그리고 문자와 숫자로 된 음표(코드)로는 모델 입출력으로 사용할 수 없기 때문에 각 코드를 숫자로 변환할 수 있는 사전을 하나 만들어봅니다. 첫번째 사전은 코드를 숫자로, 두번째 사전은 숫자를 코드로 만드는 코드입니다.


```python
code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}

idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
            7:'c8', 8:'d8', 9:'e8', 10:'f8', 11:'g8', 12:'a8', 13:'b8'}
```

이러한 사전을 이용해서 순차적인 음표를 우리가 지정한 윈도우 크기만큼 잘라 데이터셋을 생성하는 함수를 정의해보겠습니다.


```python
import numpy as np

def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq)-window_size):
        subset = seq[i:(i+window_size+1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)
```

seq라는 변수에 "나비야" 곡 전체 음표를 저장한 다음, seq2dataset() 함수를 하여 dataset를 생성합니다. 데이터셋은 앞서 정의한 사전에 따라 숫자로 변환되어 생성됩니다.


```python
seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
       'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
       'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
       'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

dataset = seq2dataset(seq, window_size = 4)

print(dataset.shape)
print(dataset)
```

    (50, 5)
    [[11  9  2 10  8]
     [ 9  2 10  8  1]
     [ 2 10  8  1  7]
     [10  8  1  7  8]
     [ 8  1  7  8  9]
     [ 1  7  8  9 10]
     [ 7  8  9 10 11]
     [ 8  9 10 11 11]
     [ 9 10 11 11  4]
     [10 11 11  4 11]
     [11 11  4 11  9]
     [11  4 11  9  9]
     [ 4 11  9  9  9]
     [11  9  9  9 10]
     [ 9  9  9 10  8]
     [ 9  9 10  8  1]
     [ 9 10  8  1  7]
     [10  8  1  7  9]
     [ 8  1  7  9 11]
     [ 1  7  9 11 11]
     [ 7  9 11 11  9]
     [ 9 11 11  9  9]
     [11 11  9  9  2]
     [11  9  9  2  8]
     [ 9  9  2  8  8]
     [ 9  2  8  8  8]
     [ 2  8  8  8  8]
     [ 8  8  8  8  8]
     [ 8  8  8  8  9]
     [ 8  8  8  9  3]
     [ 8  8  9  3  9]
     [ 8  9  3  9  9]
     [ 9  3  9  9  9]
     [ 3  9  9  9  9]
     [ 9  9  9  9  9]
     [ 9  9  9  9 10]
     [ 9  9  9 10  4]
     [ 9  9 10  4 11]
     [ 9 10  4 11  9]
     [10  4 11  9  2]
     [ 4 11  9  2 10]
     [11  9  2 10  8]
     [ 9  2 10  8  1]
     [ 2 10  8  1  7]
     [10  8  1  7  9]
     [ 8  1  7  9 11]
     [ 1  7  9 11 11]
     [ 7  9 11 11  9]
     [ 9 11 11  9  9]
     [11 11  9  9  2]]


---

### 학습 과정

"나비야"노래는 우리에게 너무나 익숙한 노래입니다. 만약 옆사람이 "나비야~ 나"까지만 불러도 나머지를 이어서 다 부를 수 있을 정도로 말이죠. 이렇게 첫 4개 음표를 입력하면 나머지를 연주할 수 있는 모델을 만드는 것이 목표입니다. 우리가 정의한 문제를 풀기 위해 먼저 모델을 학습시켜야 합니다. 학습 시키는 방식은 아래와 같습니다.

- 파란색 박스가 입력값이고, 빨간색 박스가 우리가 원하는 출력값입니다. 
- 1~4번째 음표를 데이터로 5번째 음표를 라벨값으로 학습을 시킵니다.
- 다음에는 2~5번째 음표를 데이터로 6번째 음표를 라벨값으로 학습을 시킵니다.
- 이후 한 음표씩 넘어가면서 노래 끝까지 학습시킵니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_5.png)

---
### 예측 과정

예측은 두 가지 방법으로 해보겠습니다. `한 스텝 예측`과 `곡 전체 예측`입니다. 

#### 한 스텝 예측

한 스텝 예측이란 실제 음표 4개를 입력하여 다음 음표 1개를 예측하는 것을 반복하는 것입니다. 이 방법에서는 모델의 입력값으로는 항상 실제 음표가 들어갑니다.
- 모델에 t0, t1, t2, t3를 입력하면 y0 출력이 나옵니다. 
- 모델에 t1, t2, t3, t4를 입력하면 y1 출력이 나옵니다.
- 모델에 t2, t3, t4, t5를 입력하면 y2 출력이 나옵니다.
- 이 과정을 y49 출력까지 반복합니다. 

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_6.png)

#### 곡 전체 예측

곡 전체 예측이란 입력된 초가 4개 음표만을 입력으로 곡 전체를 예측하는 것입니다. 초반부가 지나면, 예측값만으로 모델에 입력되어 다음 예측값이 나오는 식입니다. 그야말로 "나비야~ 나"까지 알려주면 나머지까지 모두 연주를 하는 것이죠. 만약 중간에 틀린 부분이 생긴다면, 이후 음정, 박자는 모두 이상하게 될 가능성이 많습니다. 예측 오류가 누적되는 것이겠죠.

- 모델에 t0, t1, t2, t3를 입력하면 y0 출력이 나옵니다.
- 예측값인 y0를 t4라고 가정하고, 모델에 t1, t2, t3, t4을 입력하면 y1 출력이 나옵니다.
- 예측값인 y1을 t5라고 가정하고, 모델에 t2, t3, t4(예측값), t5(예측값)을 입력하면 y2 출력이 나옵니다.
- 이 과정을 y49 출력까지 반복합니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_7.png)

---

### 다층 퍼셉트론 모델

앞서 생성한 데이터셋으로 먼저 다층 퍼셉트론 모델을 학습시켜보겠습니다. Dense 레이어 3개로 구성하였고, 입력 속성이 4개이고 출력이 12개(one_hot_vec_size=12)으로 설정했습니다.


```python
# 모델 구성하기
model = Sequential()
model.add(Dense(128, input_dim=4, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(one_hot_vec_size, activation='softmax'))
```

"나비야" 악보를 이 모델을 학습할 경우 다음 그림과 같이 수행됩니다. 4개의 음표를 입력으로 받고, 그 다음 음표가 라벨값으로 지정됩니다. 이 과정을 곡이 마칠 때까지 반복하게 됩니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_train_MLP.png)

전체 소스는 다음과 같습니다.


```python
# 코드 사전 정의

code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}

idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
            7:'c8', 8:'d8', 9:'e8', 10:'f8', 11:'g8', 12:'a8', 13:'b8'}

# 데이터셋 생성 함수

import numpy as np

def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq)-window_size):
        subset = seq[i:(i+window_size+1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)

# 시퀀스 데이터 정의

seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
       'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
       'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
       'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

# 데이터셋 생성

dataset = seq2dataset(seq, window_size = 4)

print(dataset.shape)
print(dataset)

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# 랜덤시드 고정시키기
np.random.seed(5)

# 입력(X)과 출력(Y) 변수로 분리하기
train_X = dataset[:,0:4]
train_Y = dataset[:,4]

max_idx_value = 13

# 입력값 정규화 시키기
train_X = train_X / float(max_idx_value)

# 라벨값에 대한 one-hot 인코딩 수행
train_Y = np_utils.to_categorical(train_Y)

one_hot_vec_size = train_Y.shape[1]

print("one hot encoding vector size is ", one_hot_vec_size)

# 모델 구성하기
model = Sequential()
model.add(Dense(128, input_dim=4, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습시키기
model.fit(train_X, train_Y, epochs=2000, batch_size=10, verbose=2)

# 모델 평가하기
scores = model.evaluate(train_X, train_Y)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 예측하기

pred_count = 50 # 최대 예측 개수 정의

# 한 스텝 예측

seq_out = ['g8', 'e8', 'e4', 'f8']
pred_out = model.predict(train_X)

for i in range(pred_count):
    idx = np.argmax(pred_out[i]) # one-hot 인코딩을 인덱스 값으로 변환
    seq_out.append(idx2code[idx]) # seq_out는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장
    
print("one step prediction : ", seq_out)

# 곡 전체 예측

seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in
seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in] # 코드를 인덱스값으로 변환

for i in range(pred_count):
    sample_in = np.array(seq_in)
    sample_in = np.reshape(sample_in, (1, 4)) # batch_size, feature
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx / float(max_idx_value))
    seq_in.pop(0)

print("full song prediction : ", seq_out)
```

    (50, 5)
    [[11  9  2 10  8]
     [ 9  2 10  8  1]
     [ 2 10  8  1  7]
     [10  8  1  7  8]
     [ 8  1  7  8  9]
     [ 1  7  8  9 10]
     [ 7  8  9 10 11]
     [ 8  9 10 11 11]
     [ 9 10 11 11  4]
     [10 11 11  4 11]
     [11 11  4 11  9]
     [11  4 11  9  9]
     [ 4 11  9  9  9]
     [11  9  9  9 10]
     [ 9  9  9 10  8]
     [ 9  9 10  8  1]
     [ 9 10  8  1  7]
     [10  8  1  7  9]
     [ 8  1  7  9 11]
     [ 1  7  9 11 11]
     [ 7  9 11 11  9]
     [ 9 11 11  9  9]
     [11 11  9  9  2]
     [11  9  9  2  8]
     [ 9  9  2  8  8]
     [ 9  2  8  8  8]
     [ 2  8  8  8  8]
     [ 8  8  8  8  8]
     [ 8  8  8  8  9]
     [ 8  8  8  9  3]
     [ 8  8  9  3  9]
     [ 8  9  3  9  9]
     [ 9  3  9  9  9]
     [ 3  9  9  9  9]
     [ 9  9  9  9  9]
     [ 9  9  9  9 10]
     [ 9  9  9 10  4]
     [ 9  9 10  4 11]
     [ 9 10  4 11  9]
     [10  4 11  9  2]
     [ 4 11  9  2 10]
     [11  9  2 10  8]
     [ 9  2 10  8  1]
     [ 2 10  8  1  7]
     [10  8  1  7  9]
     [ 8  1  7  9 11]
     [ 1  7  9 11 11]
     [ 7  9 11 11  9]
     [ 9 11 11  9  9]
     [11 11  9  9  2]]
    ('one hot encoding vector size is ', 12)
    Epoch 1/2000
    2s - loss: 2.4744 - acc: 0.1600
    Epoch 2/2000
    0s - loss: 2.3733 - acc: 0.3400
    Epoch 3/2000
    0s - loss: 2.2874 - acc: 0.3400
    Epoch 4/2000
    0s - loss: 2.2074 - acc: 0.3400
    Epoch 5/2000
    0s - loss: 2.1258 - acc: 0.3400
    Epoch 6/2000
    0s - loss: 2.0627 - acc: 0.3400
    Epoch 7/2000
    0s - loss: 1.9966 - acc: 0.3400
    Epoch 8/2000
    0s - loss: 1.9606 - acc: 0.3400
    Epoch 9/2000
    0s - loss: 1.9286 - acc: 0.3400
    Epoch 10/2000
    0s - loss: 1.9101 - acc: 0.3400
    Epoch 11/2000
    0s - loss: 1.8914 - acc: 0.3400
    Epoch 12/2000
    0s - loss: 1.8800 - acc: 0.3400
    Epoch 13/2000
    0s - loss: 1.8622 - acc: 0.3400
    Epoch 14/2000
    0s - loss: 1.8471 - acc: 0.3400
    Epoch 15/2000
    0s - loss: 1.8319 - acc: 0.3400
    Epoch 16/2000
    0s - loss: 1.8210 - acc: 0.3400
    Epoch 17/2000
    0s - loss: 1.8103 - acc: 0.3400
    Epoch 18/2000
    0s - loss: 1.7987 - acc: 0.3400
    Epoch 19/2000
    0s - loss: 1.7860 - acc: 0.3400
    Epoch 20/2000
    0s - loss: 1.7795 - acc: 0.3400
    Epoch 21/2000
    0s - loss: 1.7673 - acc: 0.3400
    Epoch 22/2000
    0s - loss: 1.7538 - acc: 0.3400
    Epoch 23/2000
    0s - loss: 1.7413 - acc: 0.3400
    Epoch 24/2000
    0s - loss: 1.7320 - acc: 0.3600
    Epoch 25/2000
    0s - loss: 1.7251 - acc: 0.3800
    Epoch 26/2000
    0s - loss: 1.7124 - acc: 0.3800
    Epoch 27/2000
    0s - loss: 1.7033 - acc: 0.3800
    Epoch 28/2000
    0s - loss: 1.6935 - acc: 0.3600
    Epoch 29/2000
    0s - loss: 1.6827 - acc: 0.3800
    Epoch 30/2000
    0s - loss: 1.6747 - acc: 0.4200
    Epoch 31/2000
    0s - loss: 1.6633 - acc: 0.4200
    Epoch 32/2000
    0s - loss: 1.6545 - acc: 0.4000
    Epoch 33/2000
    0s - loss: 1.6524 - acc: 0.3600
    Epoch 34/2000
    0s - loss: 1.6421 - acc: 0.3800
    Epoch 35/2000
    0s - loss: 1.6307 - acc: 0.4000
    Epoch 36/2000
    0s - loss: 1.6231 - acc: 0.4000
    Epoch 37/2000
    0s - loss: 1.6190 - acc: 0.4000
    Epoch 38/2000
    0s - loss: 1.6144 - acc: 0.4200
    Epoch 39/2000
    0s - loss: 1.5997 - acc: 0.4600
    Epoch 40/2000
    0s - loss: 1.5939 - acc: 0.4600
    Epoch 41/2000
    0s - loss: 1.5929 - acc: 0.4200
    Epoch 42/2000
    0s - loss: 1.5860 - acc: 0.4600
    Epoch 43/2000
    0s - loss: 1.5790 - acc: 0.4600
    Epoch 44/2000
    0s - loss: 1.5779 - acc: 0.4600
    Epoch 45/2000
    0s - loss: 1.5683 - acc: 0.4600
    Epoch 46/2000
    0s - loss: 1.5662 - acc: 0.4600
    Epoch 47/2000
    0s - loss: 1.5643 - acc: 0.4600
    Epoch 48/2000
    0s - loss: 1.5549 - acc: 0.4600
    Epoch 49/2000
    0s - loss: 1.5532 - acc: 0.4600
    Epoch 50/2000
    0s - loss: 1.5502 - acc: 0.4600
    Epoch 51/2000
    0s - loss: 1.5440 - acc: 0.5000
    Epoch 52/2000
    0s - loss: 1.5422 - acc: 0.4800
    Epoch 53/2000
    0s - loss: 1.5382 - acc: 0.4800
    Epoch 54/2000
    0s - loss: 1.5378 - acc: 0.4800
    Epoch 55/2000
    0s - loss: 1.5301 - acc: 0.4600
    Epoch 56/2000
    0s - loss: 1.5274 - acc: 0.4400
    Epoch 57/2000
    0s - loss: 1.5222 - acc: 0.4800
    Epoch 58/2000
    0s - loss: 1.5217 - acc: 0.5000
    Epoch 59/2000
    0s - loss: 1.5148 - acc: 0.5000
    Epoch 60/2000
    0s - loss: 1.5113 - acc: 0.4800
    Epoch 61/2000
    0s - loss: 1.5104 - acc: 0.4800
    Epoch 62/2000
    0s - loss: 1.5087 - acc: 0.5000
    Epoch 63/2000
    0s - loss: 1.5063 - acc: 0.4800
    Epoch 64/2000
    0s - loss: 1.5010 - acc: 0.5200
    Epoch 65/2000
    0s - loss: 1.4999 - acc: 0.4800
    Epoch 66/2000
    0s - loss: 1.4990 - acc: 0.4800
    Epoch 67/2000
    0s - loss: 1.4985 - acc: 0.4800
    Epoch 68/2000
    0s - loss: 1.5011 - acc: 0.4000
    Epoch 69/2000
    0s - loss: 1.4961 - acc: 0.5200
    Epoch 70/2000
    0s - loss: 1.4872 - acc: 0.5200
    Epoch 71/2000
    0s - loss: 1.4847 - acc: 0.5200
    Epoch 72/2000
    0s - loss: 1.4791 - acc: 0.4800
    Epoch 73/2000
    0s - loss: 1.4793 - acc: 0.5200
    Epoch 74/2000
    0s - loss: 1.4824 - acc: 0.4800
    Epoch 75/2000
    0s - loss: 1.4775 - acc: 0.4600
    Epoch 76/2000
    0s - loss: 1.4691 - acc: 0.5000
    Epoch 77/2000
    0s - loss: 1.4690 - acc: 0.5600
    Epoch 78/2000
    0s - loss: 1.4740 - acc: 0.4800
    Epoch 79/2000
    0s - loss: 1.4639 - acc: 0.5200
    Epoch 80/2000
    0s - loss: 1.4720 - acc: 0.5400
    Epoch 81/2000
    0s - loss: 1.4608 - acc: 0.5400
    Epoch 82/2000
    0s - loss: 1.4546 - acc: 0.5600
    Epoch 83/2000
    0s - loss: 1.4616 - acc: 0.4600
    Epoch 84/2000
    0s - loss: 1.4557 - acc: 0.5000
    Epoch 85/2000
    0s - loss: 1.4495 - acc: 0.5400
    Epoch 86/2000
    0s - loss: 1.4467 - acc: 0.5000
    Epoch 87/2000
    0s - loss: 1.4454 - acc: 0.5200
    Epoch 88/2000
    0s - loss: 1.4437 - acc: 0.5200
    Epoch 89/2000
    0s - loss: 1.4428 - acc: 0.5600
    Epoch 90/2000
    0s - loss: 1.4406 - acc: 0.5400
    Epoch 91/2000
    0s - loss: 1.4462 - acc: 0.5400
    Epoch 92/2000
    0s - loss: 1.4407 - acc: 0.5400
    Epoch 93/2000
    0s - loss: 1.4347 - acc: 0.5400
    Epoch 94/2000
    0s - loss: 1.4315 - acc: 0.4800
    Epoch 95/2000
    0s - loss: 1.4300 - acc: 0.5000
    Epoch 96/2000
    0s - loss: 1.4235 - acc: 0.5000
    Epoch 97/2000
    0s - loss: 1.4293 - acc: 0.5000
    Epoch 98/2000
    0s - loss: 1.4211 - acc: 0.4800
    Epoch 99/2000
    0s - loss: 1.4159 - acc: 0.5200
    Epoch 100/2000
    0s - loss: 1.4143 - acc: 0.5200
    Epoch 101/2000
    0s - loss: 1.4136 - acc: 0.5200
    Epoch 102/2000
    0s - loss: 1.4160 - acc: 0.5000
    Epoch 103/2000
    0s - loss: 1.4170 - acc: 0.5200
    Epoch 104/2000
    0s - loss: 1.4071 - acc: 0.5200
    Epoch 105/2000
    0s - loss: 1.4032 - acc: 0.5200
    Epoch 106/2000
    0s - loss: 1.4013 - acc: 0.5200
    Epoch 107/2000
    0s - loss: 1.4046 - acc: 0.5200
    Epoch 108/2000
    0s - loss: 1.3997 - acc: 0.5000
    Epoch 109/2000
    0s - loss: 1.4033 - acc: 0.5000
    Epoch 110/2000
    0s - loss: 1.3953 - acc: 0.5200
    Epoch 111/2000
    0s - loss: 1.3953 - acc: 0.5400
    Epoch 112/2000
    0s - loss: 1.3941 - acc: 0.5200
    Epoch 113/2000
    0s - loss: 1.3970 - acc: 0.4800
    Epoch 114/2000
    0s - loss: 1.3903 - acc: 0.5200
    Epoch 115/2000
    0s - loss: 1.3827 - acc: 0.5000
    Epoch 116/2000
    0s - loss: 1.3869 - acc: 0.5400
    Epoch 117/2000
    0s - loss: 1.3765 - acc: 0.5400
    Epoch 118/2000
    0s - loss: 1.3818 - acc: 0.5000
    Epoch 119/2000
    0s - loss: 1.3756 - acc: 0.5000
    Epoch 120/2000
    0s - loss: 1.3712 - acc: 0.5200
    Epoch 121/2000
    0s - loss: 1.3736 - acc: 0.5200
    Epoch 122/2000
    0s - loss: 1.3670 - acc: 0.5000
    Epoch 123/2000
    0s - loss: 1.3811 - acc: 0.4800
    Epoch 124/2000
    0s - loss: 1.3712 - acc: 0.5200
    Epoch 125/2000
    0s - loss: 1.3639 - acc: 0.5400
    Epoch 126/2000
    0s - loss: 1.3556 - acc: 0.5400
    Epoch 127/2000
    0s - loss: 1.3564 - acc: 0.5000
    Epoch 128/2000
    0s - loss: 1.3548 - acc: 0.5200
    Epoch 129/2000
    0s - loss: 1.3528 - acc: 0.5400
    Epoch 130/2000
    0s - loss: 1.3531 - acc: 0.5200
    Epoch 131/2000
    0s - loss: 1.3514 - acc: 0.5600
    Epoch 132/2000
    0s - loss: 1.3539 - acc: 0.5400
    Epoch 133/2000
    0s - loss: 1.3438 - acc: 0.5200
    Epoch 134/2000
    0s - loss: 1.3488 - acc: 0.5200
    Epoch 135/2000
    0s - loss: 1.3403 - acc: 0.5000
    Epoch 136/2000
    0s - loss: 1.3396 - acc: 0.5400
    Epoch 137/2000
    0s - loss: 1.3354 - acc: 0.5200
    Epoch 138/2000
    0s - loss: 1.3337 - acc: 0.5400
    Epoch 139/2000
    0s - loss: 1.3336 - acc: 0.5400
    Epoch 140/2000
    0s - loss: 1.3295 - acc: 0.5400
    Epoch 141/2000
    0s - loss: 1.3292 - acc: 0.5600
    Epoch 142/2000
    0s - loss: 1.3278 - acc: 0.5200
    Epoch 143/2000
    0s - loss: 1.3229 - acc: 0.5400
    Epoch 144/2000
    0s - loss: 1.3227 - acc: 0.5400
    Epoch 145/2000
    0s - loss: 1.3195 - acc: 0.5600
    Epoch 146/2000
    0s - loss: 1.3249 - acc: 0.5400
    Epoch 147/2000
    0s - loss: 1.3145 - acc: 0.5600
    Epoch 148/2000
    0s - loss: 1.3146 - acc: 0.5200
    Epoch 149/2000
    0s - loss: 1.3135 - acc: 0.5200
    Epoch 150/2000
    0s - loss: 1.3100 - acc: 0.5400
    Epoch 151/2000
    0s - loss: 1.3077 - acc: 0.5400
    Epoch 152/2000
    0s - loss: 1.3052 - acc: 0.5600
    Epoch 153/2000
    0s - loss: 1.3046 - acc: 0.5600
    Epoch 154/2000
    0s - loss: 1.3026 - acc: 0.5400
    Epoch 155/2000
    0s - loss: 1.3130 - acc: 0.5600
    Epoch 156/2000
    0s - loss: 1.2950 - acc: 0.5600
    Epoch 157/2000
    0s - loss: 1.3059 - acc: 0.5600
    Epoch 158/2000
    0s - loss: 1.2960 - acc: 0.5400
    Epoch 159/2000
    0s - loss: 1.2932 - acc: 0.5400
    Epoch 160/2000
    0s - loss: 1.2866 - acc: 0.5400
    Epoch 161/2000
    0s - loss: 1.2967 - acc: 0.5400
    Epoch 162/2000
    0s - loss: 1.2878 - acc: 0.5400
    Epoch 163/2000
    0s - loss: 1.2874 - acc: 0.5600
    Epoch 164/2000
    0s - loss: 1.2888 - acc: 0.5600
    Epoch 165/2000
    0s - loss: 1.2850 - acc: 0.5400
    Epoch 166/2000
    0s - loss: 1.2806 - acc: 0.5400
    Epoch 167/2000
    0s - loss: 1.2765 - acc: 0.5400
    Epoch 168/2000
    0s - loss: 1.2794 - acc: 0.5400
    Epoch 169/2000
    0s - loss: 1.2762 - acc: 0.5600
    Epoch 170/2000
    0s - loss: 1.2783 - acc: 0.5400
    Epoch 171/2000
    0s - loss: 1.2739 - acc: 0.5400
    Epoch 172/2000
    0s - loss: 1.2740 - acc: 0.5400
    Epoch 173/2000
    0s - loss: 1.2640 - acc: 0.5600
    Epoch 174/2000
    0s - loss: 1.2629 - acc: 0.5600
    Epoch 175/2000
    0s - loss: 1.2645 - acc: 0.5600
    Epoch 176/2000
    0s - loss: 1.2595 - acc: 0.5600
    Epoch 177/2000
    0s - loss: 1.2572 - acc: 0.5600
    Epoch 178/2000
    0s - loss: 1.2587 - acc: 0.5400
    Epoch 179/2000
    0s - loss: 1.2583 - acc: 0.5600
    Epoch 180/2000
    0s - loss: 1.2498 - acc: 0.5600
    Epoch 181/2000
    0s - loss: 1.2504 - acc: 0.5600
    Epoch 182/2000
    0s - loss: 1.2507 - acc: 0.5400
    Epoch 183/2000
    0s - loss: 1.2487 - acc: 0.5600
    Epoch 184/2000
    0s - loss: 1.2439 - acc: 0.5600
    Epoch 185/2000
    0s - loss: 1.2439 - acc: 0.5600
    Epoch 186/2000
    0s - loss: 1.2414 - acc: 0.5400
    Epoch 187/2000
    0s - loss: 1.2372 - acc: 0.5400
    Epoch 188/2000
    0s - loss: 1.2403 - acc: 0.5400
    Epoch 189/2000
    0s - loss: 1.2416 - acc: 0.5400
    Epoch 190/2000
    0s - loss: 1.2319 - acc: 0.5400
    Epoch 191/2000
    0s - loss: 1.2308 - acc: 0.5600
    Epoch 192/2000
    0s - loss: 1.2273 - acc: 0.5600
    Epoch 193/2000
    0s - loss: 1.2347 - acc: 0.5400
    Epoch 194/2000
    0s - loss: 1.2286 - acc: 0.5400
    Epoch 195/2000
    0s - loss: 1.2293 - acc: 0.5400
    Epoch 196/2000
    0s - loss: 1.2288 - acc: 0.5600
    Epoch 197/2000
    0s - loss: 1.2293 - acc: 0.5600
    Epoch 198/2000
    0s - loss: 1.2246 - acc: 0.5400
    Epoch 199/2000
    0s - loss: 1.2168 - acc: 0.5600
    Epoch 200/2000
    0s - loss: 1.2113 - acc: 0.5400
    Epoch 201/2000
    0s - loss: 1.2245 - acc: 0.5400
    Epoch 202/2000
    0s - loss: 1.2174 - acc: 0.5400
    Epoch 203/2000
    0s - loss: 1.2127 - acc: 0.5600
    Epoch 204/2000
    0s - loss: 1.2058 - acc: 0.6000
    Epoch 205/2000
    0s - loss: 1.2103 - acc: 0.5600
    Epoch 206/2000
    0s - loss: 1.2096 - acc: 0.5400
    Epoch 207/2000
    0s - loss: 1.1998 - acc: 0.5800
    Epoch 208/2000
    0s - loss: 1.2015 - acc: 0.5800
    Epoch 209/2000
    0s - loss: 1.1999 - acc: 0.6000
    Epoch 210/2000
    0s - loss: 1.2051 - acc: 0.5600
    Epoch 211/2000
    0s - loss: 1.1976 - acc: 0.5400
    Epoch 212/2000
    0s - loss: 1.1953 - acc: 0.6000
    Epoch 213/2000
    0s - loss: 1.2017 - acc: 0.5600
    Epoch 214/2000
    0s - loss: 1.1934 - acc: 0.5800
    Epoch 215/2000
    0s - loss: 1.1911 - acc: 0.5800
    Epoch 216/2000
    0s - loss: 1.1967 - acc: 0.5400
    Epoch 217/2000
    0s - loss: 1.1929 - acc: 0.5600
    Epoch 218/2000
    0s - loss: 1.1908 - acc: 0.5800
    Epoch 219/2000
    0s - loss: 1.1881 - acc: 0.6000
    Epoch 220/2000
    0s - loss: 1.1819 - acc: 0.5800
    Epoch 221/2000
    0s - loss: 1.1853 - acc: 0.5400
    Epoch 222/2000
    0s - loss: 1.1817 - acc: 0.5400
    Epoch 223/2000
    0s - loss: 1.1734 - acc: 0.5800
    Epoch 224/2000
    0s - loss: 1.1786 - acc: 0.5800
    Epoch 225/2000
    0s - loss: 1.1720 - acc: 0.5800
    Epoch 226/2000
    0s - loss: 1.1725 - acc: 0.5400
    Epoch 227/2000
    0s - loss: 1.1733 - acc: 0.5400
    Epoch 228/2000
    0s - loss: 1.1710 - acc: 0.5800
    Epoch 229/2000
    0s - loss: 1.1723 - acc: 0.6000
    Epoch 230/2000
    0s - loss: 1.1601 - acc: 0.5800
    Epoch 231/2000
    0s - loss: 1.1610 - acc: 0.5800
    Epoch 232/2000
    0s - loss: 1.1651 - acc: 0.5800
    Epoch 233/2000
    0s - loss: 1.1593 - acc: 0.5800
    Epoch 234/2000
    0s - loss: 1.1574 - acc: 0.5600
    Epoch 235/2000
    0s - loss: 1.1575 - acc: 0.5600
    Epoch 236/2000
    0s - loss: 1.1519 - acc: 0.5800
    Epoch 237/2000
    0s - loss: 1.1505 - acc: 0.6000
    Epoch 238/2000
    0s - loss: 1.1513 - acc: 0.5800
    Epoch 239/2000
    0s - loss: 1.1485 - acc: 0.5800
    Epoch 240/2000
    0s - loss: 1.1463 - acc: 0.5800
    Epoch 241/2000
    0s - loss: 1.1476 - acc: 0.5800
    Epoch 242/2000
    0s - loss: 1.1458 - acc: 0.5800
    Epoch 243/2000
    0s - loss: 1.1479 - acc: 0.5800
    Epoch 244/2000
    0s - loss: 1.1443 - acc: 0.6000
    Epoch 245/2000
    0s - loss: 1.1511 - acc: 0.5800
    Epoch 246/2000
    0s - loss: 1.1343 - acc: 0.5800
    Epoch 247/2000
    0s - loss: 1.1348 - acc: 0.5800
    Epoch 248/2000
    0s - loss: 1.1343 - acc: 0.6000
    Epoch 249/2000
    0s - loss: 1.1340 - acc: 0.5800
    Epoch 250/2000
    0s - loss: 1.1354 - acc: 0.5800
    Epoch 251/2000
    0s - loss: 1.1283 - acc: 0.5800
    Epoch 252/2000
    0s - loss: 1.1298 - acc: 0.5800
    Epoch 253/2000
    0s - loss: 1.1233 - acc: 0.5800
    Epoch 254/2000
    0s - loss: 1.1290 - acc: 0.5800
    Epoch 255/2000
    0s - loss: 1.1229 - acc: 0.5800
    Epoch 256/2000
    0s - loss: 1.1304 - acc: 0.6000
    Epoch 257/2000
    0s - loss: 1.1266 - acc: 0.6000
    Epoch 258/2000
    0s - loss: 1.1258 - acc: 0.5800
    Epoch 259/2000
    0s - loss: 1.1198 - acc: 0.5600
    Epoch 260/2000
    0s - loss: 1.1139 - acc: 0.5800
    Epoch 261/2000
    0s - loss: 1.1138 - acc: 0.5800
    Epoch 262/2000
    0s - loss: 1.1122 - acc: 0.5800
    Epoch 263/2000
    0s - loss: 1.1119 - acc: 0.5800
    Epoch 264/2000
    0s - loss: 1.1073 - acc: 0.5800
    Epoch 265/2000
    0s - loss: 1.1067 - acc: 0.5800
    Epoch 266/2000
    0s - loss: 1.1160 - acc: 0.5800
    Epoch 267/2000
    0s - loss: 1.1028 - acc: 0.5800
    Epoch 268/2000
    0s - loss: 1.1029 - acc: 0.5800
    Epoch 269/2000
    0s - loss: 1.1030 - acc: 0.5800
    Epoch 270/2000
    0s - loss: 1.1038 - acc: 0.5800
    Epoch 271/2000
    0s - loss: 1.0989 - acc: 0.5800
    Epoch 272/2000
    0s - loss: 1.1044 - acc: 0.5800
    Epoch 273/2000
    0s - loss: 1.1007 - acc: 0.6000
    Epoch 274/2000
    0s - loss: 1.1016 - acc: 0.5800
    Epoch 275/2000
    0s - loss: 1.1014 - acc: 0.5600
    Epoch 276/2000
    0s - loss: 1.1015 - acc: 0.5600
    Epoch 277/2000
    0s - loss: 1.0926 - acc: 0.5800
    Epoch 278/2000
    0s - loss: 1.0874 - acc: 0.5800
    Epoch 279/2000
    0s - loss: 1.0889 - acc: 0.5800
    Epoch 280/2000
    0s - loss: 1.0864 - acc: 0.5800
    Epoch 281/2000
    0s - loss: 1.0829 - acc: 0.5800
    Epoch 282/2000
    0s - loss: 1.0825 - acc: 0.5800
    Epoch 283/2000
    0s - loss: 1.0830 - acc: 0.5800
    Epoch 284/2000
    0s - loss: 1.0824 - acc: 0.5800
    Epoch 285/2000
    0s - loss: 1.0813 - acc: 0.5800
    Epoch 286/2000
    0s - loss: 1.0761 - acc: 0.5800
    Epoch 287/2000
    0s - loss: 1.0764 - acc: 0.5800
    Epoch 288/2000
    0s - loss: 1.0743 - acc: 0.5800
    Epoch 289/2000
    0s - loss: 1.0717 - acc: 0.5800
    Epoch 290/2000
    0s - loss: 1.0705 - acc: 0.5800
    Epoch 291/2000
    0s - loss: 1.0678 - acc: 0.5800
    Epoch 292/2000
    0s - loss: 1.0742 - acc: 0.5800
    Epoch 293/2000
    0s - loss: 1.0708 - acc: 0.5400
    Epoch 294/2000
    0s - loss: 1.0721 - acc: 0.5600
    Epoch 295/2000
    0s - loss: 1.0634 - acc: 0.5800
    Epoch 296/2000
    0s - loss: 1.0721 - acc: 0.5800
    Epoch 297/2000
    0s - loss: 1.0595 - acc: 0.5800
    Epoch 298/2000
    0s - loss: 1.0594 - acc: 0.5800
    Epoch 299/2000
    0s - loss: 1.0705 - acc: 0.5600
    Epoch 300/2000
    0s - loss: 1.0669 - acc: 0.5800
    Epoch 301/2000
    0s - loss: 1.0578 - acc: 0.5800
    Epoch 302/2000
    0s - loss: 1.0641 - acc: 0.6000
    Epoch 303/2000
    0s - loss: 1.0559 - acc: 0.5800
    Epoch 304/2000
    0s - loss: 1.0596 - acc: 0.5400
    Epoch 305/2000
    0s - loss: 1.0549 - acc: 0.5400
    Epoch 306/2000
    0s - loss: 1.0499 - acc: 0.6000
    Epoch 307/2000
    0s - loss: 1.0460 - acc: 0.5400
    Epoch 308/2000
    0s - loss: 1.0477 - acc: 0.5600
    Epoch 309/2000
    0s - loss: 1.0444 - acc: 0.5800
    Epoch 310/2000
    0s - loss: 1.0413 - acc: 0.5600
    Epoch 311/2000
    0s - loss: 1.0433 - acc: 0.5400
    Epoch 312/2000
    0s - loss: 1.0386 - acc: 0.5800
    Epoch 313/2000
    0s - loss: 1.0396 - acc: 0.5600
    Epoch 314/2000
    0s - loss: 1.0356 - acc: 0.5400
    Epoch 315/2000
    0s - loss: 1.0349 - acc: 0.5800
    Epoch 316/2000
    0s - loss: 1.0349 - acc: 0.5800
    Epoch 317/2000
    0s - loss: 1.0340 - acc: 0.6000
    Epoch 318/2000
    0s - loss: 1.0317 - acc: 0.5800
    Epoch 319/2000
    0s - loss: 1.0360 - acc: 0.6000
    Epoch 320/2000
    0s - loss: 1.0322 - acc: 0.5800
    Epoch 321/2000
    0s - loss: 1.0252 - acc: 0.5800
    Epoch 322/2000
    0s - loss: 1.0242 - acc: 0.5400
    Epoch 323/2000
    0s - loss: 1.0248 - acc: 0.5400
    Epoch 324/2000
    0s - loss: 1.0267 - acc: 0.5600
    Epoch 325/2000
    0s - loss: 1.0218 - acc: 0.6000
    Epoch 326/2000
    0s - loss: 1.0192 - acc: 0.5600
    Epoch 327/2000
    0s - loss: 1.0204 - acc: 0.5400
    Epoch 328/2000
    0s - loss: 1.0223 - acc: 0.5400
    Epoch 329/2000
    0s - loss: 1.0163 - acc: 0.5600
    Epoch 330/2000
    0s - loss: 1.0219 - acc: 0.5800
    Epoch 331/2000
    0s - loss: 1.0210 - acc: 0.5600
    Epoch 332/2000
    0s - loss: 1.0178 - acc: 0.5800
    Epoch 333/2000
    0s - loss: 1.0115 - acc: 0.6000
    Epoch 334/2000
    0s - loss: 1.0174 - acc: 0.5800
    Epoch 335/2000
    0s - loss: 1.0209 - acc: 0.5600
    Epoch 336/2000
    0s - loss: 1.0134 - acc: 0.5400
    Epoch 337/2000
    0s - loss: 1.0138 - acc: 0.6000
    Epoch 338/2000
    0s - loss: 1.0158 - acc: 0.5600
    Epoch 339/2000
    0s - loss: 1.0129 - acc: 0.5400
    Epoch 340/2000
    0s - loss: 1.0017 - acc: 0.5600
    Epoch 341/2000
    0s - loss: 1.0064 - acc: 0.5800
    Epoch 342/2000
    0s - loss: 1.0089 - acc: 0.5600
    Epoch 343/2000
    0s - loss: 0.9992 - acc: 0.6000
    Epoch 344/2000
    0s - loss: 1.0001 - acc: 0.5800
    Epoch 345/2000
    0s - loss: 1.0053 - acc: 0.5600
    Epoch 346/2000
    0s - loss: 1.0001 - acc: 0.5600
    Epoch 347/2000
    0s - loss: 1.0007 - acc: 0.5600
    Epoch 348/2000
    0s - loss: 1.0076 - acc: 0.5600
    Epoch 349/2000
    0s - loss: 0.9900 - acc: 0.5800
    Epoch 350/2000
    0s - loss: 0.9912 - acc: 0.5600
    Epoch 351/2000
    0s - loss: 0.9926 - acc: 0.5600
    Epoch 352/2000
    0s - loss: 0.9998 - acc: 0.5600
    Epoch 353/2000
    0s - loss: 0.9916 - acc: 0.5600
    Epoch 354/2000
    0s - loss: 0.9950 - acc: 0.6000
    Epoch 355/2000
    0s - loss: 0.9931 - acc: 0.5800
    Epoch 356/2000
    0s - loss: 0.9948 - acc: 0.5800
    Epoch 357/2000
    0s - loss: 0.9867 - acc: 0.5600
    Epoch 358/2000
    0s - loss: 0.9823 - acc: 0.5600
    Epoch 359/2000
    0s - loss: 0.9820 - acc: 0.5600
    Epoch 360/2000
    0s - loss: 0.9821 - acc: 0.5600
    Epoch 361/2000
    0s - loss: 0.9800 - acc: 0.5600
    Epoch 362/2000
    0s - loss: 0.9788 - acc: 0.5800
    Epoch 363/2000
    0s - loss: 0.9757 - acc: 0.5600
    Epoch 364/2000
    0s - loss: 0.9759 - acc: 0.5600
    Epoch 365/2000
    0s - loss: 0.9734 - acc: 0.5800
    Epoch 366/2000
    0s - loss: 0.9697 - acc: 0.6000
    Epoch 367/2000
    0s - loss: 0.9689 - acc: 0.5600
    Epoch 368/2000
    0s - loss: 0.9666 - acc: 0.5600
    Epoch 369/2000
    0s - loss: 0.9636 - acc: 0.5800
    Epoch 370/2000
    0s - loss: 0.9667 - acc: 0.5600
    Epoch 371/2000
    0s - loss: 0.9649 - acc: 0.5600
    Epoch 372/2000
    0s - loss: 0.9626 - acc: 0.5600
    Epoch 373/2000
    0s - loss: 0.9634 - acc: 0.5600
    Epoch 374/2000
    0s - loss: 0.9542 - acc: 0.5800
    Epoch 375/2000
    0s - loss: 0.9636 - acc: 0.5800
    Epoch 376/2000
    0s - loss: 0.9623 - acc: 0.6000
    Epoch 377/2000
    0s - loss: 0.9589 - acc: 0.6000
    Epoch 378/2000
    0s - loss: 0.9557 - acc: 0.5800
    Epoch 379/2000
    0s - loss: 0.9502 - acc: 0.5800
    Epoch 380/2000
    0s - loss: 0.9541 - acc: 0.5600
    Epoch 381/2000
    0s - loss: 0.9541 - acc: 0.5600
    Epoch 382/2000
    0s - loss: 0.9489 - acc: 0.6000
    Epoch 383/2000
    0s - loss: 0.9511 - acc: 0.6000
    Epoch 384/2000
    0s - loss: 0.9588 - acc: 0.5600
    Epoch 385/2000
    0s - loss: 0.9517 - acc: 0.5600
    Epoch 386/2000
    0s - loss: 0.9506 - acc: 0.5800
    Epoch 387/2000
    0s - loss: 0.9464 - acc: 0.6200
    Epoch 388/2000
    0s - loss: 0.9430 - acc: 0.6000
    Epoch 389/2000
    0s - loss: 0.9442 - acc: 0.5600
    Epoch 390/2000
    0s - loss: 0.9524 - acc: 0.6000
    Epoch 391/2000
    0s - loss: 0.9465 - acc: 0.6400
    Epoch 392/2000
    0s - loss: 0.9399 - acc: 0.5600
    Epoch 393/2000
    0s - loss: 0.9353 - acc: 0.5600
    Epoch 394/2000
    0s - loss: 0.9353 - acc: 0.5800
    Epoch 395/2000
    0s - loss: 0.9366 - acc: 0.6000
    Epoch 396/2000
    0s - loss: 0.9329 - acc: 0.6000
    Epoch 397/2000
    0s - loss: 0.9350 - acc: 0.5800
    Epoch 398/2000
    0s - loss: 0.9320 - acc: 0.6000
    Epoch 399/2000
    0s - loss: 0.9312 - acc: 0.5600
    Epoch 400/2000
    0s - loss: 0.9316 - acc: 0.5600
    Epoch 401/2000
    0s - loss: 0.9362 - acc: 0.6200
    Epoch 402/2000
    0s - loss: 0.9277 - acc: 0.6000
    Epoch 403/2000
    0s - loss: 0.9285 - acc: 0.5800
    Epoch 404/2000
    0s - loss: 0.9236 - acc: 0.6000
    Epoch 405/2000
    0s - loss: 0.9321 - acc: 0.6000
    Epoch 406/2000
    0s - loss: 0.9280 - acc: 0.5800
    Epoch 407/2000
    0s - loss: 0.9200 - acc: 0.5800
    Epoch 408/2000
    0s - loss: 0.9313 - acc: 0.6000
    Epoch 409/2000
    0s - loss: 0.9307 - acc: 0.6000
    Epoch 410/2000
    0s - loss: 0.9273 - acc: 0.5600
    Epoch 411/2000
    0s - loss: 0.9172 - acc: 0.5800
    Epoch 412/2000
    0s - loss: 0.9138 - acc: 0.5800
    Epoch 413/2000
    0s - loss: 0.9219 - acc: 0.5800
    Epoch 414/2000
    0s - loss: 0.9198 - acc: 0.5800
    Epoch 415/2000
    0s - loss: 0.9130 - acc: 0.6000
    Epoch 416/2000
    0s - loss: 0.9142 - acc: 0.5800
    Epoch 417/2000
    0s - loss: 0.9105 - acc: 0.5800
    Epoch 418/2000
    0s - loss: 0.9085 - acc: 0.5800
    Epoch 419/2000
    0s - loss: 0.9111 - acc: 0.5800
    Epoch 420/2000
    0s - loss: 0.9124 - acc: 0.6200
    Epoch 421/2000
    0s - loss: 0.9057 - acc: 0.6000
    Epoch 422/2000
    0s - loss: 0.9049 - acc: 0.5800
    Epoch 423/2000
    0s - loss: 0.9020 - acc: 0.5800
    Epoch 424/2000
    0s - loss: 0.9027 - acc: 0.6000
    Epoch 425/2000
    0s - loss: 0.9062 - acc: 0.5800
    Epoch 426/2000
    0s - loss: 0.9010 - acc: 0.5800
    Epoch 427/2000
    0s - loss: 0.9059 - acc: 0.5600
    Epoch 428/2000
    0s - loss: 0.9017 - acc: 0.6000
    Epoch 429/2000
    0s - loss: 0.8993 - acc: 0.5800
    Epoch 430/2000
    0s - loss: 0.8944 - acc: 0.5800
    Epoch 431/2000
    0s - loss: 0.8933 - acc: 0.5600
    Epoch 432/2000
    0s - loss: 0.8977 - acc: 0.5800
    Epoch 433/2000
    0s - loss: 0.8937 - acc: 0.6400
    Epoch 434/2000
    0s - loss: 0.8975 - acc: 0.6000
    Epoch 435/2000
    0s - loss: 0.9066 - acc: 0.5800
    Epoch 436/2000
    0s - loss: 0.8904 - acc: 0.6000
    Epoch 437/2000
    0s - loss: 0.8881 - acc: 0.6200
    Epoch 438/2000
    0s - loss: 0.8933 - acc: 0.6400
    Epoch 439/2000
    0s - loss: 0.8905 - acc: 0.6000
    Epoch 440/2000
    0s - loss: 0.8875 - acc: 0.6000
    Epoch 441/2000
    0s - loss: 0.8890 - acc: 0.5800
    Epoch 442/2000
    0s - loss: 0.8851 - acc: 0.6200
    Epoch 443/2000
    0s - loss: 0.8889 - acc: 0.6000
    Epoch 444/2000
    0s - loss: 0.8821 - acc: 0.6000
    Epoch 445/2000
    0s - loss: 0.8862 - acc: 0.5800
    Epoch 446/2000
    0s - loss: 0.8828 - acc: 0.6000
    Epoch 447/2000
    0s - loss: 0.8831 - acc: 0.5800
    Epoch 448/2000
    0s - loss: 0.8833 - acc: 0.5800
    Epoch 449/2000
    0s - loss: 0.8945 - acc: 0.5800
    Epoch 450/2000
    0s - loss: 0.8762 - acc: 0.6000
    Epoch 451/2000
    0s - loss: 0.8851 - acc: 0.6600
    Epoch 452/2000
    0s - loss: 0.8822 - acc: 0.6400
    Epoch 453/2000
    0s - loss: 0.8853 - acc: 0.6200
    Epoch 454/2000
    0s - loss: 0.8807 - acc: 0.6200
    Epoch 455/2000
    0s - loss: 0.8784 - acc: 0.6400
    Epoch 456/2000
    0s - loss: 0.8765 - acc: 0.6000
    Epoch 457/2000
    0s - loss: 0.8821 - acc: 0.6000
    Epoch 458/2000
    0s - loss: 0.8707 - acc: 0.6000
    Epoch 459/2000
    0s - loss: 0.8727 - acc: 0.6200
    Epoch 460/2000
    0s - loss: 0.8714 - acc: 0.6400
    Epoch 461/2000
    0s - loss: 0.8739 - acc: 0.6200
    Epoch 462/2000
    0s - loss: 0.8659 - acc: 0.6200
    Epoch 463/2000
    0s - loss: 0.8708 - acc: 0.6200
    Epoch 464/2000
    0s - loss: 0.8672 - acc: 0.6200
    Epoch 465/2000
    0s - loss: 0.8636 - acc: 0.6200
    Epoch 466/2000
    0s - loss: 0.8624 - acc: 0.6600
    Epoch 467/2000
    0s - loss: 0.8628 - acc: 0.6400
    Epoch 468/2000
    0s - loss: 0.8585 - acc: 0.6200
    Epoch 469/2000
    0s - loss: 0.8593 - acc: 0.6400
    Epoch 470/2000
    0s - loss: 0.8584 - acc: 0.6600
    Epoch 471/2000
    0s - loss: 0.8572 - acc: 0.6000
    Epoch 472/2000
    0s - loss: 0.8562 - acc: 0.6200
    Epoch 473/2000
    0s - loss: 0.8519 - acc: 0.6600
    Epoch 474/2000
    0s - loss: 0.8536 - acc: 0.6600
    Epoch 475/2000
    0s - loss: 0.8541 - acc: 0.6400
    Epoch 476/2000
    0s - loss: 0.8499 - acc: 0.6200
    Epoch 477/2000
    0s - loss: 0.8616 - acc: 0.6200
    Epoch 478/2000
    0s - loss: 0.8498 - acc: 0.6400
    Epoch 479/2000
    0s - loss: 0.8505 - acc: 0.6400
    Epoch 480/2000
    0s - loss: 0.8467 - acc: 0.6200
    Epoch 481/2000
    0s - loss: 0.8458 - acc: 0.6600
    Epoch 482/2000
    0s - loss: 0.8518 - acc: 0.6200
    Epoch 483/2000
    0s - loss: 0.8475 - acc: 0.6200
    Epoch 484/2000
    0s - loss: 0.8490 - acc: 0.6400
    Epoch 485/2000
    0s - loss: 0.8576 - acc: 0.6800
    Epoch 486/2000
    0s - loss: 0.8435 - acc: 0.6800
    Epoch 487/2000
    0s - loss: 0.8503 - acc: 0.6200
    Epoch 488/2000
    0s - loss: 0.8453 - acc: 0.6400
    Epoch 489/2000
    0s - loss: 0.8430 - acc: 0.6400
    Epoch 490/2000
    0s - loss: 0.8448 - acc: 0.6200
    Epoch 491/2000
    0s - loss: 0.8414 - acc: 0.6400
    Epoch 492/2000
    0s - loss: 0.8376 - acc: 0.6400
    Epoch 493/2000
    0s - loss: 0.8335 - acc: 0.6400
    Epoch 494/2000
    0s - loss: 0.8326 - acc: 0.6800
    Epoch 495/2000
    0s - loss: 0.8393 - acc: 0.6600
    Epoch 496/2000
    0s - loss: 0.8354 - acc: 0.6200
    Epoch 497/2000
    0s - loss: 0.8309 - acc: 0.6400
    Epoch 498/2000
    0s - loss: 0.8292 - acc: 0.6800
    Epoch 499/2000
    0s - loss: 0.8316 - acc: 0.6800
    Epoch 500/2000
    0s - loss: 0.8312 - acc: 0.6600
    Epoch 501/2000
    0s - loss: 0.8287 - acc: 0.6600
    Epoch 502/2000
    0s - loss: 0.8329 - acc: 0.6600
    Epoch 503/2000
    0s - loss: 0.8269 - acc: 0.6600
    Epoch 504/2000
    0s - loss: 0.8290 - acc: 0.6400
    Epoch 505/2000
    0s - loss: 0.8265 - acc: 0.6400
    Epoch 506/2000
    0s - loss: 0.8337 - acc: 0.6400
    Epoch 507/2000
    0s - loss: 0.8263 - acc: 0.6800
    Epoch 508/2000
    0s - loss: 0.8282 - acc: 0.6800
    Epoch 509/2000
    0s - loss: 0.8209 - acc: 0.6600
    Epoch 510/2000
    0s - loss: 0.8189 - acc: 0.6800
    Epoch 511/2000
    0s - loss: 0.8329 - acc: 0.6200
    Epoch 512/2000
    0s - loss: 0.8164 - acc: 0.6800
    Epoch 513/2000
    0s - loss: 0.8236 - acc: 0.6600
    Epoch 514/2000
    0s - loss: 0.8197 - acc: 0.6800
    Epoch 515/2000
    0s - loss: 0.8159 - acc: 0.6800
    Epoch 516/2000
    0s - loss: 0.8134 - acc: 0.6800
    Epoch 517/2000
    0s - loss: 0.8154 - acc: 0.6600
    Epoch 518/2000
    0s - loss: 0.8350 - acc: 0.5800
    Epoch 519/2000
    0s - loss: 0.8168 - acc: 0.6600
    Epoch 520/2000
    0s - loss: 0.8247 - acc: 0.6400
    Epoch 521/2000
    0s - loss: 0.8172 - acc: 0.6600
    Epoch 522/2000
    0s - loss: 0.8081 - acc: 0.6800
    Epoch 523/2000
    0s - loss: 0.8186 - acc: 0.6200
    Epoch 524/2000
    0s - loss: 0.8060 - acc: 0.6400
    Epoch 525/2000
    0s - loss: 0.8078 - acc: 0.6600
    Epoch 526/2000
    0s - loss: 0.8101 - acc: 0.6800
    Epoch 527/2000
    0s - loss: 0.8041 - acc: 0.6800
    Epoch 528/2000
    0s - loss: 0.8086 - acc: 0.6600
    Epoch 529/2000
    0s - loss: 0.8048 - acc: 0.6400
    Epoch 530/2000
    0s - loss: 0.8095 - acc: 0.6600
    Epoch 531/2000
    0s - loss: 0.8031 - acc: 0.6800
    Epoch 532/2000
    0s - loss: 0.8037 - acc: 0.6800
    Epoch 533/2000
    0s - loss: 0.8065 - acc: 0.6000
    Epoch 534/2000
    0s - loss: 0.8000 - acc: 0.6600
    Epoch 535/2000
    0s - loss: 0.8005 - acc: 0.6600
    Epoch 536/2000
    0s - loss: 0.8003 - acc: 0.6800
    Epoch 537/2000
    0s - loss: 0.8042 - acc: 0.6600
    Epoch 538/2000
    0s - loss: 0.8118 - acc: 0.6400
    Epoch 539/2000
    0s - loss: 0.7936 - acc: 0.6800
    Epoch 540/2000
    0s - loss: 0.8030 - acc: 0.6800
    Epoch 541/2000
    0s - loss: 0.8016 - acc: 0.6800
    Epoch 542/2000
    0s - loss: 0.7951 - acc: 0.6800
    Epoch 543/2000
    0s - loss: 0.8027 - acc: 0.6800
    Epoch 544/2000
    0s - loss: 0.7946 - acc: 0.6800
    Epoch 545/2000
    0s - loss: 0.7957 - acc: 0.6600
    Epoch 546/2000
    0s - loss: 0.7937 - acc: 0.6400
    Epoch 547/2000
    0s - loss: 0.7867 - acc: 0.6600
    Epoch 548/2000
    0s - loss: 0.7967 - acc: 0.6800
    Epoch 549/2000
    0s - loss: 0.7900 - acc: 0.6600
    Epoch 550/2000
    0s - loss: 0.7849 - acc: 0.6800
    Epoch 551/2000
    0s - loss: 0.7884 - acc: 0.6800
    Epoch 552/2000
    0s - loss: 0.7855 - acc: 0.6800
    Epoch 553/2000
    0s - loss: 0.7872 - acc: 0.6800
    Epoch 554/2000
    0s - loss: 0.7829 - acc: 0.6800
    Epoch 555/2000
    0s - loss: 0.7804 - acc: 0.6800
    Epoch 556/2000
    0s - loss: 0.7853 - acc: 0.6800
    Epoch 557/2000
    0s - loss: 0.7954 - acc: 0.6800
    Epoch 558/2000
    0s - loss: 0.7798 - acc: 0.6600
    Epoch 559/2000
    0s - loss: 0.7797 - acc: 0.6800
    Epoch 560/2000
    0s - loss: 0.7744 - acc: 0.6800
    Epoch 561/2000
    0s - loss: 0.7771 - acc: 0.6800
    Epoch 562/2000
    0s - loss: 0.7742 - acc: 0.6800
    Epoch 563/2000
    0s - loss: 0.7700 - acc: 0.6800
    Epoch 564/2000
    0s - loss: 0.7763 - acc: 0.6800
    Epoch 565/2000
    0s - loss: 0.7693 - acc: 0.6800
    Epoch 566/2000
    0s - loss: 0.7851 - acc: 0.6800
    Epoch 567/2000
    0s - loss: 0.7786 - acc: 0.7000
    Epoch 568/2000
    0s - loss: 0.7682 - acc: 0.6800
    Epoch 569/2000
    0s - loss: 0.7762 - acc: 0.6600
    Epoch 570/2000
    0s - loss: 0.7681 - acc: 0.6800
    Epoch 571/2000
    0s - loss: 0.7761 - acc: 0.6800
    Epoch 572/2000
    0s - loss: 0.7694 - acc: 0.6800
    Epoch 573/2000
    0s - loss: 0.7730 - acc: 0.6400
    Epoch 574/2000
    0s - loss: 0.7715 - acc: 0.7000
    Epoch 575/2000
    0s - loss: 0.7694 - acc: 0.6800
    Epoch 576/2000
    0s - loss: 0.7634 - acc: 0.6800
    Epoch 577/2000
    0s - loss: 0.7688 - acc: 0.6800
    Epoch 578/2000
    0s - loss: 0.7666 - acc: 0.6800
    Epoch 579/2000
    0s - loss: 0.7721 - acc: 0.6800
    Epoch 580/2000
    0s - loss: 0.7654 - acc: 0.6800
    Epoch 581/2000
    0s - loss: 0.7618 - acc: 0.6800
    Epoch 582/2000
    0s - loss: 0.7582 - acc: 0.6800
    Epoch 583/2000
    0s - loss: 0.7564 - acc: 0.6800
    Epoch 584/2000
    0s - loss: 0.7622 - acc: 0.6800
    Epoch 585/2000
    0s - loss: 0.7586 - acc: 0.6800
    Epoch 586/2000
    0s - loss: 0.7602 - acc: 0.7000
    Epoch 587/2000
    0s - loss: 0.7565 - acc: 0.6800
    Epoch 588/2000
    0s - loss: 0.7534 - acc: 0.6800
    Epoch 589/2000
    0s - loss: 0.7555 - acc: 0.6800
    Epoch 590/2000
    0s - loss: 0.7535 - acc: 0.6800
    Epoch 591/2000
    0s - loss: 0.7473 - acc: 0.6800
    Epoch 592/2000
    0s - loss: 0.7504 - acc: 0.6800
    Epoch 593/2000
    0s - loss: 0.7520 - acc: 0.7200
    Epoch 594/2000
    0s - loss: 0.7495 - acc: 0.7000
    Epoch 595/2000
    0s - loss: 0.7481 - acc: 0.6800
    Epoch 596/2000
    0s - loss: 0.7496 - acc: 0.6800
    Epoch 597/2000
    0s - loss: 0.7454 - acc: 0.6800
    Epoch 598/2000
    0s - loss: 0.7479 - acc: 0.6800
    Epoch 599/2000
    0s - loss: 0.7517 - acc: 0.6800
    Epoch 600/2000
    0s - loss: 0.7420 - acc: 0.6800
    Epoch 601/2000
    0s - loss: 0.7438 - acc: 0.6800
    Epoch 602/2000
    0s - loss: 0.7461 - acc: 0.6800
    Epoch 603/2000
    0s - loss: 0.7405 - acc: 0.6800
    Epoch 604/2000
    0s - loss: 0.7461 - acc: 0.7000
    Epoch 605/2000
    0s - loss: 0.7406 - acc: 0.7000
    Epoch 606/2000
    0s - loss: 0.7402 - acc: 0.6600
    Epoch 607/2000
    0s - loss: 0.7505 - acc: 0.6800
    Epoch 608/2000
    0s - loss: 0.7409 - acc: 0.6800
    Epoch 609/2000
    0s - loss: 0.7396 - acc: 0.6800
    Epoch 610/2000
    0s - loss: 0.7367 - acc: 0.6800
    Epoch 611/2000
    0s - loss: 0.7376 - acc: 0.6800
    Epoch 612/2000
    0s - loss: 0.7337 - acc: 0.6800
    Epoch 613/2000
    0s - loss: 0.7432 - acc: 0.7200
    Epoch 614/2000
    0s - loss: 0.7313 - acc: 0.6800
    Epoch 615/2000
    0s - loss: 0.7370 - acc: 0.6800
    Epoch 616/2000
    0s - loss: 0.7405 - acc: 0.6800
    Epoch 617/2000
    0s - loss: 0.7303 - acc: 0.7000
    Epoch 618/2000
    0s - loss: 0.7345 - acc: 0.6800
    Epoch 619/2000
    0s - loss: 0.7279 - acc: 0.6800
    Epoch 620/2000
    0s - loss: 0.7308 - acc: 0.7000
    Epoch 621/2000
    0s - loss: 0.7309 - acc: 0.7200
    Epoch 622/2000
    0s - loss: 0.7208 - acc: 0.7200
    Epoch 623/2000
    0s - loss: 0.7271 - acc: 0.6800
    Epoch 624/2000
    0s - loss: 0.7217 - acc: 0.7000
    Epoch 625/2000
    0s - loss: 0.7325 - acc: 0.6800
    Epoch 626/2000
    0s - loss: 0.7393 - acc: 0.6800
    Epoch 627/2000
    0s - loss: 0.7285 - acc: 0.6800
    Epoch 628/2000
    0s - loss: 0.7209 - acc: 0.6800
    Epoch 629/2000
    0s - loss: 0.7272 - acc: 0.7000
    Epoch 630/2000
    0s - loss: 0.7284 - acc: 0.7000
    Epoch 631/2000
    0s - loss: 0.7242 - acc: 0.6800
    Epoch 632/2000
    0s - loss: 0.7220 - acc: 0.7200
    Epoch 633/2000
    0s - loss: 0.7215 - acc: 0.7000
    Epoch 634/2000
    0s - loss: 0.7181 - acc: 0.6800
    Epoch 635/2000
    0s - loss: 0.7147 - acc: 0.6800
    Epoch 636/2000
    0s - loss: 0.7207 - acc: 0.7200
    Epoch 637/2000
    0s - loss: 0.7138 - acc: 0.7200
    Epoch 638/2000
    0s - loss: 0.7239 - acc: 0.7000
    Epoch 639/2000
    0s - loss: 0.7148 - acc: 0.6800
    Epoch 640/2000
    0s - loss: 0.7266 - acc: 0.7000
    Epoch 641/2000
    0s - loss: 0.7127 - acc: 0.7200
    Epoch 642/2000
    0s - loss: 0.7165 - acc: 0.7200
    Epoch 643/2000
    0s - loss: 0.7161 - acc: 0.7200
    Epoch 644/2000
    0s - loss: 0.7079 - acc: 0.7200
    Epoch 645/2000
    0s - loss: 0.7071 - acc: 0.7400
    Epoch 646/2000
    0s - loss: 0.7041 - acc: 0.7000
    Epoch 647/2000
    0s - loss: 0.7041 - acc: 0.6800
    Epoch 648/2000
    0s - loss: 0.7112 - acc: 0.7000
    Epoch 649/2000
    0s - loss: 0.7020 - acc: 0.7400
    Epoch 650/2000
    0s - loss: 0.7063 - acc: 0.7600
    Epoch 651/2000
    0s - loss: 0.7030 - acc: 0.7200
    Epoch 652/2000
    0s - loss: 0.7000 - acc: 0.7000
    Epoch 653/2000
    0s - loss: 0.6977 - acc: 0.6800
    Epoch 654/2000
    0s - loss: 0.7001 - acc: 0.6800
    Epoch 655/2000
    0s - loss: 0.6985 - acc: 0.6800
    Epoch 656/2000
    0s - loss: 0.7078 - acc: 0.7000
    Epoch 657/2000
    0s - loss: 0.7006 - acc: 0.7000
    Epoch 658/2000
    0s - loss: 0.7020 - acc: 0.6800
    Epoch 659/2000
    0s - loss: 0.7094 - acc: 0.7200
    Epoch 660/2000
    0s - loss: 0.6968 - acc: 0.7400
    Epoch 661/2000
    0s - loss: 0.7020 - acc: 0.7000
    Epoch 662/2000
    0s - loss: 0.6980 - acc: 0.6800
    Epoch 663/2000
    0s - loss: 0.6951 - acc: 0.7200
    Epoch 664/2000
    0s - loss: 0.7048 - acc: 0.7400
    Epoch 665/2000
    0s - loss: 0.6865 - acc: 0.7200
    Epoch 666/2000
    0s - loss: 0.7002 - acc: 0.7200
    Epoch 667/2000
    0s - loss: 0.6890 - acc: 0.7200
    Epoch 668/2000
    0s - loss: 0.6902 - acc: 0.7000
    Epoch 669/2000
    0s - loss: 0.6950 - acc: 0.7200
    Epoch 670/2000
    0s - loss: 0.6929 - acc: 0.7000
    Epoch 671/2000
    0s - loss: 0.6939 - acc: 0.7200
    Epoch 672/2000
    0s - loss: 0.6891 - acc: 0.7400
    Epoch 673/2000
    0s - loss: 0.6900 - acc: 0.7200
    Epoch 674/2000
    0s - loss: 0.6868 - acc: 0.7000
    Epoch 675/2000
    0s - loss: 0.6943 - acc: 0.7600
    Epoch 676/2000
    0s - loss: 0.6808 - acc: 0.7600
    Epoch 677/2000
    0s - loss: 0.6905 - acc: 0.7000
    Epoch 678/2000
    0s - loss: 0.6824 - acc: 0.7400
    Epoch 679/2000
    0s - loss: 0.6888 - acc: 0.7600
    Epoch 680/2000
    0s - loss: 0.6817 - acc: 0.7400
    Epoch 681/2000
    0s - loss: 0.6907 - acc: 0.7000
    Epoch 682/2000
    0s - loss: 0.6731 - acc: 0.7200
    Epoch 683/2000
    0s - loss: 0.6849 - acc: 0.7400
    Epoch 684/2000
    0s - loss: 0.6862 - acc: 0.7200
    Epoch 685/2000
    0s - loss: 0.6776 - acc: 0.7400
    Epoch 686/2000
    0s - loss: 0.6778 - acc: 0.7400
    Epoch 687/2000
    0s - loss: 0.6734 - acc: 0.7000
    Epoch 688/2000
    0s - loss: 0.6791 - acc: 0.7800
    Epoch 689/2000
    0s - loss: 0.6760 - acc: 0.7800
    Epoch 690/2000
    0s - loss: 0.6727 - acc: 0.7400
    Epoch 691/2000
    0s - loss: 0.6874 - acc: 0.7000
    Epoch 692/2000
    0s - loss: 0.6976 - acc: 0.7200
    Epoch 693/2000
    0s - loss: 0.6745 - acc: 0.7600
    Epoch 694/2000
    0s - loss: 0.6785 - acc: 0.7800
    Epoch 695/2000
    0s - loss: 0.6645 - acc: 0.7600
    Epoch 696/2000
    0s - loss: 0.6706 - acc: 0.7400
    Epoch 697/2000
    0s - loss: 0.6687 - acc: 0.7400
    Epoch 698/2000
    0s - loss: 0.6763 - acc: 0.7600
    Epoch 699/2000
    0s - loss: 0.6708 - acc: 0.7600
    Epoch 700/2000
    0s - loss: 0.6698 - acc: 0.7400
    Epoch 701/2000
    0s - loss: 0.6648 - acc: 0.7000
    Epoch 702/2000
    0s - loss: 0.6663 - acc: 0.7400
    Epoch 703/2000
    0s - loss: 0.6656 - acc: 0.7400
    Epoch 704/2000
    0s - loss: 0.6648 - acc: 0.7600
    Epoch 705/2000
    0s - loss: 0.6644 - acc: 0.7600
    Epoch 706/2000
    0s - loss: 0.6629 - acc: 0.7400
    Epoch 707/2000
    0s - loss: 0.6610 - acc: 0.7600
    Epoch 708/2000
    0s - loss: 0.6610 - acc: 0.7200
    Epoch 709/2000
    0s - loss: 0.6574 - acc: 0.7600
    Epoch 710/2000
    0s - loss: 0.6607 - acc: 0.7400
    Epoch 711/2000
    0s - loss: 0.6647 - acc: 0.7200
    Epoch 712/2000
    0s - loss: 0.6564 - acc: 0.7400
    Epoch 713/2000
    0s - loss: 0.6640 - acc: 0.7000
    Epoch 714/2000
    0s - loss: 0.6650 - acc: 0.7200
    Epoch 715/2000
    0s - loss: 0.6562 - acc: 0.7600
    Epoch 716/2000
    0s - loss: 0.6540 - acc: 0.7600
    Epoch 717/2000
    0s - loss: 0.6566 - acc: 0.7000
    Epoch 718/2000
    0s - loss: 0.6574 - acc: 0.7400
    Epoch 719/2000
    0s - loss: 0.6549 - acc: 0.7600
    Epoch 720/2000
    0s - loss: 0.6548 - acc: 0.8000
    Epoch 721/2000
    0s - loss: 0.6478 - acc: 0.7800
    Epoch 722/2000
    0s - loss: 0.6536 - acc: 0.7400
    Epoch 723/2000
    0s - loss: 0.6500 - acc: 0.7600
    Epoch 724/2000
    0s - loss: 0.6507 - acc: 0.7600
    Epoch 725/2000
    0s - loss: 0.6475 - acc: 0.7800
    Epoch 726/2000
    0s - loss: 0.6405 - acc: 0.7800
    Epoch 727/2000
    0s - loss: 0.6519 - acc: 0.7000
    Epoch 728/2000
    0s - loss: 0.6498 - acc: 0.7600
    Epoch 729/2000
    0s - loss: 0.6545 - acc: 0.7600
    Epoch 730/2000
    0s - loss: 0.6445 - acc: 0.7400
    Epoch 731/2000
    0s - loss: 0.6424 - acc: 0.7800
    Epoch 732/2000
    0s - loss: 0.6488 - acc: 0.7400
    Epoch 733/2000
    0s - loss: 0.6398 - acc: 0.7600
    Epoch 734/2000
    0s - loss: 0.6431 - acc: 0.7600
    Epoch 735/2000
    0s - loss: 0.6402 - acc: 0.8000
    Epoch 736/2000
    0s - loss: 0.6353 - acc: 0.7800
    Epoch 737/2000
    0s - loss: 0.6363 - acc: 0.7600
    Epoch 738/2000
    0s - loss: 0.6442 - acc: 0.7400
    Epoch 739/2000
    0s - loss: 0.6414 - acc: 0.8000
    Epoch 740/2000
    0s - loss: 0.6388 - acc: 0.7200
    Epoch 741/2000
    0s - loss: 0.6360 - acc: 0.7400
    Epoch 742/2000
    0s - loss: 0.6296 - acc: 0.7800
    Epoch 743/2000
    0s - loss: 0.6379 - acc: 0.7600
    Epoch 744/2000
    0s - loss: 0.6283 - acc: 0.7600
    Epoch 745/2000
    0s - loss: 0.6317 - acc: 0.8000
    Epoch 746/2000
    0s - loss: 0.6296 - acc: 0.8000
    Epoch 747/2000
    0s - loss: 0.6241 - acc: 0.7800
    Epoch 748/2000
    0s - loss: 0.6279 - acc: 0.7600
    Epoch 749/2000
    0s - loss: 0.6316 - acc: 0.7200
    Epoch 750/2000
    0s - loss: 0.6263 - acc: 0.7600
    Epoch 751/2000
    0s - loss: 0.6297 - acc: 0.7800
    Epoch 752/2000
    0s - loss: 0.6226 - acc: 0.7600
    Epoch 753/2000
    0s - loss: 0.6269 - acc: 0.7200
    Epoch 754/2000
    0s - loss: 0.6207 - acc: 0.7600
    Epoch 755/2000
    0s - loss: 0.6274 - acc: 0.7800
    Epoch 756/2000
    0s - loss: 0.6311 - acc: 0.7400
    Epoch 757/2000
    0s - loss: 0.6309 - acc: 0.7000
    Epoch 758/2000
    0s - loss: 0.6224 - acc: 0.7600
    Epoch 759/2000
    0s - loss: 0.6236 - acc: 0.7800
    Epoch 760/2000
    0s - loss: 0.6231 - acc: 0.8000
    Epoch 761/2000
    0s - loss: 0.6172 - acc: 0.7800
    Epoch 762/2000
    0s - loss: 0.6229 - acc: 0.7600
    Epoch 763/2000
    0s - loss: 0.6141 - acc: 0.8000
    Epoch 764/2000
    0s - loss: 0.6167 - acc: 0.7800
    Epoch 765/2000
    0s - loss: 0.6144 - acc: 0.7600
    Epoch 766/2000
    0s - loss: 0.6173 - acc: 0.7800
    Epoch 767/2000
    0s - loss: 0.6159 - acc: 0.7600
    Epoch 768/2000
    0s - loss: 0.6116 - acc: 0.7600
    Epoch 769/2000
    0s - loss: 0.6134 - acc: 0.7800
    Epoch 770/2000
    0s - loss: 0.6192 - acc: 0.7800
    Epoch 771/2000
    0s - loss: 0.6108 - acc: 0.7800
    Epoch 772/2000
    0s - loss: 0.6103 - acc: 0.8000
    Epoch 773/2000
    0s - loss: 0.6090 - acc: 0.8000
    Epoch 774/2000
    0s - loss: 0.6071 - acc: 0.7800
    Epoch 775/2000
    0s - loss: 0.6129 - acc: 0.7800
    Epoch 776/2000
    0s - loss: 0.6091 - acc: 0.8000
    Epoch 777/2000
    0s - loss: 0.6140 - acc: 0.7600
    Epoch 778/2000
    0s - loss: 0.6011 - acc: 0.8200
    Epoch 779/2000
    0s - loss: 0.6072 - acc: 0.8200
    Epoch 780/2000
    0s - loss: 0.6109 - acc: 0.8000
    Epoch 781/2000
    0s - loss: 0.6011 - acc: 0.7600
    Epoch 782/2000
    0s - loss: 0.6081 - acc: 0.7600
    Epoch 783/2000
    0s - loss: 0.6031 - acc: 0.7800
    Epoch 784/2000
    0s - loss: 0.6083 - acc: 0.8000
    Epoch 785/2000
    0s - loss: 0.6041 - acc: 0.8000
    Epoch 786/2000
    0s - loss: 0.6023 - acc: 0.8000
    Epoch 787/2000
    0s - loss: 0.6000 - acc: 0.7800
    Epoch 788/2000
    0s - loss: 0.6043 - acc: 0.7800
    Epoch 789/2000
    0s - loss: 0.6007 - acc: 0.8000
    Epoch 790/2000
    0s - loss: 0.5940 - acc: 0.7800
    Epoch 791/2000
    0s - loss: 0.5958 - acc: 0.8000
    Epoch 792/2000
    0s - loss: 0.5958 - acc: 0.8000
    Epoch 793/2000
    0s - loss: 0.5943 - acc: 0.7800
    Epoch 794/2000
    0s - loss: 0.5977 - acc: 0.8200
    Epoch 795/2000
    0s - loss: 0.5976 - acc: 0.8000
    Epoch 796/2000
    0s - loss: 0.6041 - acc: 0.7800
    Epoch 797/2000
    0s - loss: 0.5902 - acc: 0.8000
    Epoch 798/2000
    0s - loss: 0.5917 - acc: 0.8000
    Epoch 799/2000
    0s - loss: 0.6029 - acc: 0.7400
    Epoch 800/2000
    0s - loss: 0.5958 - acc: 0.7800
    Epoch 801/2000
    0s - loss: 0.5979 - acc: 0.8200
    Epoch 802/2000
    0s - loss: 0.5858 - acc: 0.7800
    Epoch 803/2000
    0s - loss: 0.5921 - acc: 0.8200
    Epoch 804/2000
    0s - loss: 0.5882 - acc: 0.8000
    Epoch 805/2000
    0s - loss: 0.5872 - acc: 0.7800
    Epoch 806/2000
    0s - loss: 0.5913 - acc: 0.7800
    Epoch 807/2000
    0s - loss: 0.5867 - acc: 0.8200
    Epoch 808/2000
    0s - loss: 0.5888 - acc: 0.8000
    Epoch 809/2000
    0s - loss: 0.5905 - acc: 0.8000
    Epoch 810/2000
    0s - loss: 0.5883 - acc: 0.8000
    Epoch 811/2000
    0s - loss: 0.5850 - acc: 0.8000
    Epoch 812/2000
    0s - loss: 0.5832 - acc: 0.7800
    Epoch 813/2000
    0s - loss: 0.5933 - acc: 0.8200
    Epoch 814/2000
    0s - loss: 0.5811 - acc: 0.7800
    Epoch 815/2000
    0s - loss: 0.5785 - acc: 0.8200
    Epoch 816/2000
    0s - loss: 0.5893 - acc: 0.8000
    Epoch 817/2000
    0s - loss: 0.5889 - acc: 0.7800
    Epoch 818/2000
    0s - loss: 0.5799 - acc: 0.8000
    Epoch 819/2000
    0s - loss: 0.5875 - acc: 0.8200
    Epoch 820/2000
    0s - loss: 0.5755 - acc: 0.8200
    Epoch 821/2000
    0s - loss: 0.5807 - acc: 0.8000
    Epoch 822/2000
    0s - loss: 0.5752 - acc: 0.8400
    Epoch 823/2000
    0s - loss: 0.5776 - acc: 0.8200
    Epoch 824/2000
    0s - loss: 0.5776 - acc: 0.8000
    Epoch 825/2000
    0s - loss: 0.5787 - acc: 0.7600
    Epoch 826/2000
    0s - loss: 0.5735 - acc: 0.8000
    Epoch 827/2000
    0s - loss: 0.5706 - acc: 0.8200
    Epoch 828/2000
    0s - loss: 0.5711 - acc: 0.8000
    Epoch 829/2000
    0s - loss: 0.5725 - acc: 0.8200
    Epoch 830/2000
    0s - loss: 0.5757 - acc: 0.8200
    Epoch 831/2000
    0s - loss: 0.5683 - acc: 0.8200
    Epoch 832/2000
    0s - loss: 0.5785 - acc: 0.8200
    Epoch 833/2000
    0s - loss: 0.5661 - acc: 0.8200
    Epoch 834/2000
    0s - loss: 0.5704 - acc: 0.8400
    Epoch 835/2000
    0s - loss: 0.5640 - acc: 0.8600
    Epoch 836/2000
    0s - loss: 0.5662 - acc: 0.8200
    Epoch 837/2000
    0s - loss: 0.5703 - acc: 0.8000
    Epoch 838/2000
    0s - loss: 0.5644 - acc: 0.8200
    Epoch 839/2000
    0s - loss: 0.5690 - acc: 0.8200
    Epoch 840/2000
    0s - loss: 0.5660 - acc: 0.8200
    Epoch 841/2000
    0s - loss: 0.5635 - acc: 0.8000
    Epoch 842/2000
    0s - loss: 0.5691 - acc: 0.8200
    Epoch 843/2000
    0s - loss: 0.5599 - acc: 0.8400
    Epoch 844/2000
    0s - loss: 0.5604 - acc: 0.8600
    Epoch 845/2000
    0s - loss: 0.5634 - acc: 0.8400
    Epoch 846/2000
    0s - loss: 0.5639 - acc: 0.8000
    Epoch 847/2000
    0s - loss: 0.5710 - acc: 0.8000
    Epoch 848/2000
    0s - loss: 0.5560 - acc: 0.8200
    Epoch 849/2000
    0s - loss: 0.5688 - acc: 0.8400
    Epoch 850/2000
    0s - loss: 0.5528 - acc: 0.8400
    Epoch 851/2000
    0s - loss: 0.5594 - acc: 0.8000
    Epoch 852/2000
    0s - loss: 0.5594 - acc: 0.8200
    Epoch 853/2000
    0s - loss: 0.5509 - acc: 0.8400
    Epoch 854/2000
    0s - loss: 0.5666 - acc: 0.8400
    Epoch 855/2000
    0s - loss: 0.5597 - acc: 0.8600
    Epoch 856/2000
    0s - loss: 0.5625 - acc: 0.8200
    Epoch 857/2000
    0s - loss: 0.5529 - acc: 0.8200
    Epoch 858/2000
    0s - loss: 0.5548 - acc: 0.8400
    Epoch 859/2000
    0s - loss: 0.5546 - acc: 0.8400
    Epoch 860/2000
    0s - loss: 0.5480 - acc: 0.8400
    Epoch 861/2000
    0s - loss: 0.5588 - acc: 0.8400
    Epoch 862/2000
    0s - loss: 0.5546 - acc: 0.8000
    Epoch 863/2000
    0s - loss: 0.5541 - acc: 0.8400
    Epoch 864/2000
    0s - loss: 0.5448 - acc: 0.8600
    Epoch 865/2000
    0s - loss: 0.5523 - acc: 0.8200
    Epoch 866/2000
    0s - loss: 0.5499 - acc: 0.8200
    Epoch 867/2000
    0s - loss: 0.5397 - acc: 0.8200
    Epoch 868/2000
    0s - loss: 0.5596 - acc: 0.8400
    Epoch 869/2000
    0s - loss: 0.5504 - acc: 0.8400
    Epoch 870/2000
    0s - loss: 0.5434 - acc: 0.8200
    Epoch 871/2000
    0s - loss: 0.5462 - acc: 0.8000
    Epoch 872/2000
    0s - loss: 0.5415 - acc: 0.8000
    Epoch 873/2000
    0s - loss: 0.5387 - acc: 0.8200
    Epoch 874/2000
    0s - loss: 0.5421 - acc: 0.8400
    Epoch 875/2000
    0s - loss: 0.5437 - acc: 0.8600
    Epoch 876/2000
    0s - loss: 0.5403 - acc: 0.7800
    Epoch 877/2000
    0s - loss: 0.5338 - acc: 0.8400
    Epoch 878/2000
    0s - loss: 0.5435 - acc: 0.8600
    Epoch 879/2000
    0s - loss: 0.5432 - acc: 0.8400
    Epoch 880/2000
    0s - loss: 0.5425 - acc: 0.8600
    Epoch 881/2000
    0s - loss: 0.5382 - acc: 0.8000
    Epoch 882/2000
    0s - loss: 0.5329 - acc: 0.8400
    Epoch 883/2000
    0s - loss: 0.5312 - acc: 0.8600
    Epoch 884/2000
    0s - loss: 0.5335 - acc: 0.8400
    Epoch 885/2000
    0s - loss: 0.5337 - acc: 0.8600
    Epoch 886/2000
    0s - loss: 0.5327 - acc: 0.8200
    Epoch 887/2000
    0s - loss: 0.5353 - acc: 0.8200
    Epoch 888/2000
    0s - loss: 0.5272 - acc: 0.8400
    Epoch 889/2000
    0s - loss: 0.5274 - acc: 0.8600
    Epoch 890/2000
    0s - loss: 0.5235 - acc: 0.8600
    Epoch 891/2000
    0s - loss: 0.5264 - acc: 0.8200
    Epoch 892/2000
    0s - loss: 0.5282 - acc: 0.8600
    Epoch 893/2000
    0s - loss: 0.5234 - acc: 0.8600
    Epoch 894/2000
    0s - loss: 0.5229 - acc: 0.8600
    Epoch 895/2000
    0s - loss: 0.5282 - acc: 0.8200
    Epoch 896/2000
    0s - loss: 0.5277 - acc: 0.8600
    Epoch 897/2000
    0s - loss: 0.5217 - acc: 0.8600
    Epoch 898/2000
    0s - loss: 0.5218 - acc: 0.8800
    Epoch 899/2000
    0s - loss: 0.5229 - acc: 0.8600
    Epoch 900/2000
    0s - loss: 0.5184 - acc: 0.8400
    Epoch 901/2000
    0s - loss: 0.5247 - acc: 0.8600
    Epoch 902/2000
    0s - loss: 0.5140 - acc: 0.8600
    Epoch 903/2000
    0s - loss: 0.5323 - acc: 0.8200
    Epoch 904/2000
    0s - loss: 0.5156 - acc: 0.8400
    Epoch 905/2000
    0s - loss: 0.5197 - acc: 0.8600
    Epoch 906/2000
    0s - loss: 0.5174 - acc: 0.8600
    Epoch 907/2000
    0s - loss: 0.5245 - acc: 0.8400
    Epoch 908/2000
    0s - loss: 0.5210 - acc: 0.8000
    Epoch 909/2000
    0s - loss: 0.5119 - acc: 0.8600
    Epoch 910/2000
    0s - loss: 0.5146 - acc: 0.8600
    Epoch 911/2000
    0s - loss: 0.5191 - acc: 0.8400
    Epoch 912/2000
    0s - loss: 0.5057 - acc: 0.8400
    Epoch 913/2000
    0s - loss: 0.5365 - acc: 0.8400
    Epoch 914/2000
    0s - loss: 0.5135 - acc: 0.8400
    Epoch 915/2000
    0s - loss: 0.5108 - acc: 0.8600
    Epoch 916/2000
    0s - loss: 0.5153 - acc: 0.8600
    Epoch 917/2000
    0s - loss: 0.5062 - acc: 0.8600
    Epoch 918/2000
    0s - loss: 0.5115 - acc: 0.8600
    Epoch 919/2000
    0s - loss: 0.5101 - acc: 0.8600
    Epoch 920/2000
    0s - loss: 0.5069 - acc: 0.8600
    Epoch 921/2000
    0s - loss: 0.5088 - acc: 0.8400
    Epoch 922/2000
    0s - loss: 0.5056 - acc: 0.8600
    Epoch 923/2000
    0s - loss: 0.5037 - acc: 0.8400
    Epoch 924/2000
    0s - loss: 0.5051 - acc: 0.8600
    Epoch 925/2000
    0s - loss: 0.5093 - acc: 0.8600
    Epoch 926/2000
    0s - loss: 0.5166 - acc: 0.8800
    Epoch 927/2000
    0s - loss: 0.5046 - acc: 0.8400
    Epoch 928/2000
    0s - loss: 0.5042 - acc: 0.8200
    Epoch 929/2000
    0s - loss: 0.4995 - acc: 0.8600
    Epoch 930/2000
    0s - loss: 0.4977 - acc: 0.8600
    Epoch 931/2000
    0s - loss: 0.4972 - acc: 0.8600
    Epoch 932/2000
    0s - loss: 0.5023 - acc: 0.8600
    Epoch 933/2000
    0s - loss: 0.4957 - acc: 0.8400
    Epoch 934/2000
    0s - loss: 0.4933 - acc: 0.8600
    Epoch 935/2000
    0s - loss: 0.4973 - acc: 0.8600
    Epoch 936/2000
    0s - loss: 0.4937 - acc: 0.8600
    Epoch 937/2000
    0s - loss: 0.4950 - acc: 0.8600
    Epoch 938/2000
    0s - loss: 0.4927 - acc: 0.8600
    Epoch 939/2000
    0s - loss: 0.4958 - acc: 0.8600
    Epoch 940/2000
    0s - loss: 0.4921 - acc: 0.8600
    Epoch 941/2000
    0s - loss: 0.4969 - acc: 0.8400
    Epoch 942/2000
    0s - loss: 0.4980 - acc: 0.8600
    Epoch 943/2000
    0s - loss: 0.5110 - acc: 0.8600
    Epoch 944/2000
    0s - loss: 0.4896 - acc: 0.8400
    Epoch 945/2000
    0s - loss: 0.4923 - acc: 0.8600
    Epoch 946/2000
    0s - loss: 0.4971 - acc: 0.8400
    Epoch 947/2000
    0s - loss: 0.4856 - acc: 0.8800
    Epoch 948/2000
    0s - loss: 0.4906 - acc: 0.8600
    Epoch 949/2000
    0s - loss: 0.4917 - acc: 0.8600
    Epoch 950/2000
    0s - loss: 0.5145 - acc: 0.8400
    Epoch 951/2000
    0s - loss: 0.4949 - acc: 0.8400
    Epoch 952/2000
    0s - loss: 0.4927 - acc: 0.8600
    Epoch 953/2000
    0s - loss: 0.4862 - acc: 0.8600
    Epoch 954/2000
    0s - loss: 0.4846 - acc: 0.8600
    Epoch 955/2000
    0s - loss: 0.4903 - acc: 0.8600
    Epoch 956/2000
    0s - loss: 0.4799 - acc: 0.8600
    Epoch 957/2000
    0s - loss: 0.4828 - acc: 0.8600
    Epoch 958/2000
    0s - loss: 0.4908 - acc: 0.8600
    Epoch 959/2000
    0s - loss: 0.4783 - acc: 0.8800
    Epoch 960/2000
    0s - loss: 0.4844 - acc: 0.8600
    Epoch 961/2000
    0s - loss: 0.4813 - acc: 0.8600
    Epoch 962/2000
    0s - loss: 0.4853 - acc: 0.8600
    Epoch 963/2000
    0s - loss: 0.4835 - acc: 0.8800
    Epoch 964/2000
    0s - loss: 0.4797 - acc: 0.8600
    Epoch 965/2000
    0s - loss: 0.4784 - acc: 0.8400
    Epoch 966/2000
    0s - loss: 0.4749 - acc: 0.8600
    Epoch 967/2000
    0s - loss: 0.4785 - acc: 0.8600
    Epoch 968/2000
    0s - loss: 0.4749 - acc: 0.8200
    Epoch 969/2000
    0s - loss: 0.4736 - acc: 0.8600
    Epoch 970/2000
    0s - loss: 0.4735 - acc: 0.8400
    Epoch 971/2000
    0s - loss: 0.4692 - acc: 0.8600
    Epoch 972/2000
    0s - loss: 0.4759 - acc: 0.8600
    Epoch 973/2000
    0s - loss: 0.4761 - acc: 0.8600
    Epoch 974/2000
    0s - loss: 0.4717 - acc: 0.8600
    Epoch 975/2000
    0s - loss: 0.4717 - acc: 0.8600
    Epoch 976/2000
    0s - loss: 0.4885 - acc: 0.8600
    Epoch 977/2000
    0s - loss: 0.4773 - acc: 0.8600
    Epoch 978/2000
    0s - loss: 0.4702 - acc: 0.8600
    Epoch 979/2000
    0s - loss: 0.4813 - acc: 0.9000
    Epoch 980/2000
    0s - loss: 0.4715 - acc: 0.8400
    Epoch 981/2000
    0s - loss: 0.4664 - acc: 0.8600
    Epoch 982/2000
    0s - loss: 0.4679 - acc: 0.8600
    Epoch 983/2000
    0s - loss: 0.4645 - acc: 0.8800
    Epoch 984/2000
    0s - loss: 0.4691 - acc: 0.8800
    Epoch 985/2000
    0s - loss: 0.4710 - acc: 0.8400
    Epoch 986/2000
    0s - loss: 0.4679 - acc: 0.8600
    Epoch 987/2000
    0s - loss: 0.4673 - acc: 0.8600
    Epoch 988/2000
    0s - loss: 0.4702 - acc: 0.8800
    Epoch 989/2000
    0s - loss: 0.4704 - acc: 0.8600
    Epoch 990/2000
    0s - loss: 0.4671 - acc: 0.8400
    Epoch 991/2000
    0s - loss: 0.4645 - acc: 0.8400
    Epoch 992/2000
    0s - loss: 0.4632 - acc: 0.8400
    Epoch 993/2000
    0s - loss: 0.4623 - acc: 0.8600
    Epoch 994/2000
    0s - loss: 0.4645 - acc: 0.8600
    Epoch 995/2000
    0s - loss: 0.4557 - acc: 0.8800
    Epoch 996/2000
    0s - loss: 0.4583 - acc: 0.8600
    Epoch 997/2000
    0s - loss: 0.4644 - acc: 0.8600
    Epoch 998/2000
    0s - loss: 0.4564 - acc: 0.8800
    Epoch 999/2000
    0s - loss: 0.4536 - acc: 0.8400
    Epoch 1000/2000
    0s - loss: 0.4597 - acc: 0.8600
    Epoch 1001/2000
    0s - loss: 0.4518 - acc: 0.8800
    Epoch 1002/2000
    0s - loss: 0.4580 - acc: 0.8800
    Epoch 1003/2000
    0s - loss: 0.4513 - acc: 0.8800
    Epoch 1004/2000
    0s - loss: 0.4554 - acc: 0.8600
    Epoch 1005/2000
    0s - loss: 0.4464 - acc: 0.8800
    Epoch 1006/2000
    0s - loss: 0.4600 - acc: 0.8600
    Epoch 1007/2000
    0s - loss: 0.4550 - acc: 0.8800
    Epoch 1008/2000
    0s - loss: 0.4491 - acc: 0.8600
    Epoch 1009/2000
    0s - loss: 0.4595 - acc: 0.8800
    Epoch 1010/2000
    0s - loss: 0.4545 - acc: 0.8200
    Epoch 1011/2000
    0s - loss: 0.4537 - acc: 0.8600
    Epoch 1012/2000
    0s - loss: 0.4485 - acc: 0.8800
    Epoch 1013/2000
    0s - loss: 0.4425 - acc: 0.8400
    Epoch 1014/2000
    0s - loss: 0.4571 - acc: 0.8600
    Epoch 1015/2000
    0s - loss: 0.4438 - acc: 0.9000
    Epoch 1016/2000
    0s - loss: 0.4455 - acc: 0.9000
    Epoch 1017/2000
    0s - loss: 0.4490 - acc: 0.8600
    Epoch 1018/2000
    0s - loss: 0.4496 - acc: 0.8800
    Epoch 1019/2000
    0s - loss: 0.4446 - acc: 0.9000
    Epoch 1020/2000
    0s - loss: 0.4419 - acc: 0.9000
    Epoch 1021/2000
    0s - loss: 0.4496 - acc: 0.8600
    Epoch 1022/2000
    0s - loss: 0.4418 - acc: 0.8600
    Epoch 1023/2000
    0s - loss: 0.4383 - acc: 0.8800
    Epoch 1024/2000
    0s - loss: 0.4414 - acc: 0.9000
    Epoch 1025/2000
    0s - loss: 0.4420 - acc: 0.8800
    Epoch 1026/2000
    0s - loss: 0.4454 - acc: 0.8800
    Epoch 1027/2000
    0s - loss: 0.4404 - acc: 0.9000
    Epoch 1028/2000
    0s - loss: 0.4408 - acc: 0.9000
    Epoch 1029/2000
    0s - loss: 0.4491 - acc: 0.8200
    Epoch 1030/2000
    0s - loss: 0.4459 - acc: 0.8800
    Epoch 1031/2000
    0s - loss: 0.4371 - acc: 0.8600
    Epoch 1032/2000
    0s - loss: 0.4403 - acc: 0.8800
    Epoch 1033/2000
    0s - loss: 0.4346 - acc: 0.8800
    Epoch 1034/2000
    0s - loss: 0.4344 - acc: 0.8600
    Epoch 1035/2000
    0s - loss: 0.4380 - acc: 0.8800
    Epoch 1036/2000
    0s - loss: 0.4396 - acc: 0.8400
    Epoch 1037/2000
    0s - loss: 0.4344 - acc: 0.8800
    Epoch 1038/2000
    0s - loss: 0.4321 - acc: 0.8800
    Epoch 1039/2000
    0s - loss: 0.4370 - acc: 0.9000
    Epoch 1040/2000
    0s - loss: 0.4346 - acc: 0.9000
    Epoch 1041/2000
    0s - loss: 0.4303 - acc: 0.9000
    Epoch 1042/2000
    0s - loss: 0.4301 - acc: 0.8400
    Epoch 1043/2000
    0s - loss: 0.4344 - acc: 0.8800
    Epoch 1044/2000
    0s - loss: 0.4273 - acc: 0.9000
    Epoch 1045/2000
    0s - loss: 0.4279 - acc: 0.9200
    Epoch 1046/2000
    0s - loss: 0.4266 - acc: 0.9200
    Epoch 1047/2000
    0s - loss: 0.4289 - acc: 0.8800
    Epoch 1048/2000
    0s - loss: 0.4259 - acc: 0.9000
    Epoch 1049/2000
    0s - loss: 0.4292 - acc: 0.8800
    Epoch 1050/2000
    0s - loss: 0.4265 - acc: 0.8400
    Epoch 1051/2000
    0s - loss: 0.4336 - acc: 0.8400
    Epoch 1052/2000
    0s - loss: 0.4225 - acc: 0.9000
    Epoch 1053/2000
    0s - loss: 0.4276 - acc: 0.8400
    Epoch 1054/2000
    0s - loss: 0.4249 - acc: 0.8800
    Epoch 1055/2000
    0s - loss: 0.4341 - acc: 0.8800
    Epoch 1056/2000
    0s - loss: 0.4230 - acc: 0.9000
    Epoch 1057/2000
    0s - loss: 0.4197 - acc: 0.9200
    Epoch 1058/2000
    0s - loss: 0.4288 - acc: 0.9200
    Epoch 1059/2000
    0s - loss: 0.4253 - acc: 0.9000
    Epoch 1060/2000
    0s - loss: 0.4226 - acc: 0.8600
    Epoch 1061/2000
    0s - loss: 0.4239 - acc: 0.9000
    Epoch 1062/2000
    0s - loss: 0.4237 - acc: 0.9000
    Epoch 1063/2000
    0s - loss: 0.4213 - acc: 0.8800
    Epoch 1064/2000
    0s - loss: 0.4185 - acc: 0.9000
    Epoch 1065/2000
    0s - loss: 0.4213 - acc: 0.9000
    Epoch 1066/2000
    0s - loss: 0.4195 - acc: 0.9000
    Epoch 1067/2000
    0s - loss: 0.4175 - acc: 0.8600
    Epoch 1068/2000
    0s - loss: 0.4188 - acc: 0.9000
    Epoch 1069/2000
    0s - loss: 0.4141 - acc: 0.9000
    Epoch 1070/2000
    0s - loss: 0.4168 - acc: 0.9000
    Epoch 1071/2000
    0s - loss: 0.4234 - acc: 0.9000
    Epoch 1072/2000
    0s - loss: 0.4163 - acc: 0.9000
    Epoch 1073/2000
    0s - loss: 0.4102 - acc: 0.9000
    Epoch 1074/2000
    0s - loss: 0.4103 - acc: 0.9000
    Epoch 1075/2000
    0s - loss: 0.4151 - acc: 0.9000
    Epoch 1076/2000
    0s - loss: 0.4082 - acc: 0.9000
    Epoch 1077/2000
    0s - loss: 0.4178 - acc: 0.8800
    Epoch 1078/2000
    0s - loss: 0.4070 - acc: 0.9000
    Epoch 1079/2000
    0s - loss: 0.4175 - acc: 0.9200
    Epoch 1080/2000
    0s - loss: 0.4150 - acc: 0.8800
    Epoch 1081/2000
    0s - loss: 0.4195 - acc: 0.8800
    Epoch 1082/2000
    0s - loss: 0.4133 - acc: 0.9000
    Epoch 1083/2000
    0s - loss: 0.4128 - acc: 0.8800
    Epoch 1084/2000
    0s - loss: 0.4033 - acc: 0.9000
    Epoch 1085/2000
    0s - loss: 0.4332 - acc: 0.8600
    Epoch 1086/2000
    0s - loss: 0.4092 - acc: 0.9000
    Epoch 1087/2000
    0s - loss: 0.4147 - acc: 0.9000
    Epoch 1088/2000
    0s - loss: 0.4170 - acc: 0.8800
    Epoch 1089/2000
    0s - loss: 0.4149 - acc: 0.9000
    Epoch 1090/2000
    0s - loss: 0.4066 - acc: 0.9000
    Epoch 1091/2000
    0s - loss: 0.4045 - acc: 0.9200
    Epoch 1092/2000
    0s - loss: 0.4073 - acc: 0.8600
    Epoch 1093/2000
    0s - loss: 0.4049 - acc: 0.9000
    Epoch 1094/2000
    0s - loss: 0.4036 - acc: 0.9000
    Epoch 1095/2000
    0s - loss: 0.3972 - acc: 0.9000
    Epoch 1096/2000
    0s - loss: 0.4025 - acc: 0.9200
    Epoch 1097/2000
    0s - loss: 0.3987 - acc: 0.9200
    Epoch 1098/2000
    0s - loss: 0.4009 - acc: 0.9200
    Epoch 1099/2000
    0s - loss: 0.4038 - acc: 0.9200
    Epoch 1100/2000
    0s - loss: 0.3994 - acc: 0.9200
    Epoch 1101/2000
    0s - loss: 0.3982 - acc: 0.9000
    Epoch 1102/2000
    0s - loss: 0.3954 - acc: 0.9200
    Epoch 1103/2000
    0s - loss: 0.3986 - acc: 0.9200
    Epoch 1104/2000
    0s - loss: 0.4012 - acc: 0.9200
    Epoch 1105/2000
    0s - loss: 0.3978 - acc: 0.8800
    Epoch 1106/2000
    0s - loss: 0.3950 - acc: 0.9200
    Epoch 1107/2000
    0s - loss: 0.3947 - acc: 0.9000
    Epoch 1108/2000
    0s - loss: 0.4028 - acc: 0.9200
    Epoch 1109/2000
    0s - loss: 0.4080 - acc: 0.9200
    Epoch 1110/2000
    0s - loss: 0.4006 - acc: 0.9000
    Epoch 1111/2000
    0s - loss: 0.3975 - acc: 0.9000
    Epoch 1112/2000
    0s - loss: 0.3953 - acc: 0.8800
    Epoch 1113/2000
    0s - loss: 0.4058 - acc: 0.9000
    Epoch 1114/2000
    0s - loss: 0.3975 - acc: 0.9200
    Epoch 1115/2000
    0s - loss: 0.3910 - acc: 0.9000
    Epoch 1116/2000
    0s - loss: 0.3916 - acc: 0.9000
    Epoch 1117/2000
    0s - loss: 0.3904 - acc: 0.9200
    Epoch 1118/2000
    0s - loss: 0.3890 - acc: 0.9000
    Epoch 1119/2000
    0s - loss: 0.3876 - acc: 0.9200
    Epoch 1120/2000
    0s - loss: 0.3905 - acc: 0.9000
    Epoch 1121/2000
    0s - loss: 0.3855 - acc: 0.9200
    Epoch 1122/2000
    0s - loss: 0.3962 - acc: 0.9000
    Epoch 1123/2000
    0s - loss: 0.3897 - acc: 0.9000
    Epoch 1124/2000
    0s - loss: 0.3853 - acc: 0.9000
    Epoch 1125/2000
    0s - loss: 0.3843 - acc: 0.9200
    Epoch 1126/2000
    0s - loss: 0.3917 - acc: 0.9000
    Epoch 1127/2000
    0s - loss: 0.3847 - acc: 0.9200
    Epoch 1128/2000
    0s - loss: 0.3852 - acc: 0.9200
    Epoch 1129/2000
    0s - loss: 0.3854 - acc: 0.9200
    Epoch 1130/2000
    0s - loss: 0.3865 - acc: 0.9200
    Epoch 1131/2000
    0s - loss: 0.3825 - acc: 0.9200
    Epoch 1132/2000
    0s - loss: 0.3910 - acc: 0.9200
    Epoch 1133/2000
    0s - loss: 0.3830 - acc: 0.9200
    Epoch 1134/2000
    0s - loss: 0.3868 - acc: 0.9200
    Epoch 1135/2000
    0s - loss: 0.3931 - acc: 0.9000
    Epoch 1136/2000
    0s - loss: 0.3859 - acc: 0.9200
    Epoch 1137/2000
    0s - loss: 0.3859 - acc: 0.9000
    Epoch 1138/2000
    0s - loss: 0.3933 - acc: 0.9000
    Epoch 1139/2000
    0s - loss: 0.3873 - acc: 0.8800
    Epoch 1140/2000
    0s - loss: 0.4030 - acc: 0.9000
    Epoch 1141/2000
    0s - loss: 0.3856 - acc: 0.9000
    Epoch 1142/2000
    0s - loss: 0.3704 - acc: 0.9200
    Epoch 1143/2000
    0s - loss: 0.3809 - acc: 0.9200
    Epoch 1144/2000
    0s - loss: 0.3823 - acc: 0.9000
    Epoch 1145/2000
    0s - loss: 0.3812 - acc: 0.9200
    Epoch 1146/2000
    0s - loss: 0.3782 - acc: 0.9200
    Epoch 1147/2000
    0s - loss: 0.3830 - acc: 0.9200
    Epoch 1148/2000
    0s - loss: 0.3789 - acc: 0.9200
    Epoch 1149/2000
    0s - loss: 0.3755 - acc: 0.9200
    Epoch 1150/2000
    0s - loss: 0.3721 - acc: 0.9200
    Epoch 1151/2000
    0s - loss: 0.3746 - acc: 0.9200
    Epoch 1152/2000
    0s - loss: 0.3983 - acc: 0.8800
    Epoch 1153/2000
    0s - loss: 0.3853 - acc: 0.9200
    Epoch 1154/2000
    0s - loss: 0.3829 - acc: 0.9200
    Epoch 1155/2000
    0s - loss: 0.3696 - acc: 0.9200
    Epoch 1156/2000
    0s - loss: 0.3707 - acc: 0.9200
    Epoch 1157/2000
    0s - loss: 0.3702 - acc: 0.9200
    Epoch 1158/2000
    0s - loss: 0.3697 - acc: 0.9200
    Epoch 1159/2000
    0s - loss: 0.3696 - acc: 0.9200
    Epoch 1160/2000
    0s - loss: 0.3753 - acc: 0.9200
    Epoch 1161/2000
    0s - loss: 0.3760 - acc: 0.9200
    Epoch 1162/2000
    0s - loss: 0.3753 - acc: 0.9000
    Epoch 1163/2000
    0s - loss: 0.3713 - acc: 0.9200
    Epoch 1164/2000
    0s - loss: 0.3858 - acc: 0.9200
    Epoch 1165/2000
    0s - loss: 0.3672 - acc: 0.9000
    Epoch 1166/2000
    0s - loss: 0.3612 - acc: 0.9200
    Epoch 1167/2000
    0s - loss: 0.3686 - acc: 0.9200
    Epoch 1168/2000
    0s - loss: 0.3707 - acc: 0.9000
    Epoch 1169/2000
    0s - loss: 0.3688 - acc: 0.9200
    Epoch 1170/2000
    0s - loss: 0.3732 - acc: 0.9200
    Epoch 1171/2000
    0s - loss: 0.3649 - acc: 0.9000
    Epoch 1172/2000
    0s - loss: 0.3790 - acc: 0.9200
    Epoch 1173/2000
    0s - loss: 0.3678 - acc: 0.9000
    Epoch 1174/2000
    0s - loss: 0.3618 - acc: 0.9000
    Epoch 1175/2000
    0s - loss: 0.3690 - acc: 0.9200
    Epoch 1176/2000
    0s - loss: 0.3680 - acc: 0.9200
    Epoch 1177/2000
    0s - loss: 0.3622 - acc: 0.9200
    Epoch 1178/2000
    0s - loss: 0.3624 - acc: 0.9200
    Epoch 1179/2000
    0s - loss: 0.3630 - acc: 0.9200
    Epoch 1180/2000
    0s - loss: 0.3601 - acc: 0.9200
    Epoch 1181/2000
    0s - loss: 0.3598 - acc: 0.9200
    Epoch 1182/2000
    0s - loss: 0.3698 - acc: 0.8800
    Epoch 1183/2000
    0s - loss: 0.3589 - acc: 0.9200
    Epoch 1184/2000
    0s - loss: 0.3720 - acc: 0.9200
    Epoch 1185/2000
    0s - loss: 0.3649 - acc: 0.9000
    Epoch 1186/2000
    0s - loss: 0.3632 - acc: 0.8800
    Epoch 1187/2000
    0s - loss: 0.3609 - acc: 0.9200
    Epoch 1188/2000
    0s - loss: 0.3637 - acc: 0.9200
    Epoch 1189/2000
    0s - loss: 0.3638 - acc: 0.9200
    Epoch 1190/2000
    0s - loss: 0.3631 - acc: 0.8800
    Epoch 1191/2000
    0s - loss: 0.3575 - acc: 0.8800
    Epoch 1192/2000
    0s - loss: 0.3526 - acc: 0.9200
    Epoch 1193/2000
    0s - loss: 0.3761 - acc: 0.8800
    Epoch 1194/2000
    0s - loss: 0.3574 - acc: 0.9200
    Epoch 1195/2000
    0s - loss: 0.3681 - acc: 0.8800
    Epoch 1196/2000
    0s - loss: 0.3572 - acc: 0.9000
    Epoch 1197/2000
    0s - loss: 0.3487 - acc: 0.9200
    Epoch 1198/2000
    0s - loss: 0.3579 - acc: 0.9200
    Epoch 1199/2000
    0s - loss: 0.3512 - acc: 0.9200
    Epoch 1200/2000
    0s - loss: 0.3588 - acc: 0.9200
    Epoch 1201/2000
    0s - loss: 0.3539 - acc: 0.9200
    Epoch 1202/2000
    0s - loss: 0.3505 - acc: 0.9200
    Epoch 1203/2000
    0s - loss: 0.3597 - acc: 0.9200
    Epoch 1204/2000
    0s - loss: 0.3466 - acc: 0.9200
    Epoch 1205/2000
    0s - loss: 0.3566 - acc: 0.9200
    Epoch 1206/2000
    0s - loss: 0.3428 - acc: 0.9200
    Epoch 1207/2000
    0s - loss: 0.3574 - acc: 0.9200
    Epoch 1208/2000
    0s - loss: 0.3477 - acc: 0.9000
    Epoch 1209/2000
    0s - loss: 0.3579 - acc: 0.9000
    Epoch 1210/2000
    0s - loss: 0.3609 - acc: 0.9000
    Epoch 1211/2000
    0s - loss: 0.3507 - acc: 0.9200
    Epoch 1212/2000
    0s - loss: 0.3462 - acc: 0.8800
    Epoch 1213/2000
    0s - loss: 0.3484 - acc: 0.9000
    Epoch 1214/2000
    0s - loss: 0.3438 - acc: 0.9200
    Epoch 1215/2000
    0s - loss: 0.3463 - acc: 0.9200
    Epoch 1216/2000
    0s - loss: 0.3467 - acc: 0.9200
    Epoch 1217/2000
    0s - loss: 0.3496 - acc: 0.9000
    Epoch 1218/2000
    0s - loss: 0.3410 - acc: 0.9200
    Epoch 1219/2000
    0s - loss: 0.3424 - acc: 0.9200
    Epoch 1220/2000
    0s - loss: 0.3415 - acc: 0.9200
    Epoch 1221/2000
    0s - loss: 0.3463 - acc: 0.9200
    Epoch 1222/2000
    0s - loss: 0.3417 - acc: 0.9200
    Epoch 1223/2000
    0s - loss: 0.3407 - acc: 0.9200
    Epoch 1224/2000
    0s - loss: 0.3437 - acc: 0.9200
    Epoch 1225/2000
    0s - loss: 0.3369 - acc: 0.9200
    Epoch 1226/2000
    0s - loss: 0.3430 - acc: 0.9200
    Epoch 1227/2000
    0s - loss: 0.3489 - acc: 0.9200
    Epoch 1228/2000
    0s - loss: 0.3519 - acc: 0.9400
    Epoch 1229/2000
    0s - loss: 0.3372 - acc: 0.9000
    Epoch 1230/2000
    0s - loss: 0.3433 - acc: 0.9200
    Epoch 1231/2000
    0s - loss: 0.3453 - acc: 0.9200
    Epoch 1232/2000
    0s - loss: 0.3411 - acc: 0.9200
    Epoch 1233/2000
    0s - loss: 0.3386 - acc: 0.9200
    Epoch 1234/2000
    0s - loss: 0.3408 - acc: 0.9200
    Epoch 1235/2000
    0s - loss: 0.3344 - acc: 0.9200
    Epoch 1236/2000
    0s - loss: 0.3550 - acc: 0.9200
    Epoch 1237/2000
    0s - loss: 0.3366 - acc: 0.9200
    Epoch 1238/2000
    0s - loss: 0.3506 - acc: 0.9200
    Epoch 1239/2000
    0s - loss: 0.3437 - acc: 0.9200
    Epoch 1240/2000
    0s - loss: 0.3431 - acc: 0.8800
    Epoch 1241/2000
    0s - loss: 0.3335 - acc: 0.9000
    Epoch 1242/2000
    0s - loss: 0.3417 - acc: 0.9200
    Epoch 1243/2000
    0s - loss: 0.3334 - acc: 0.9200
    Epoch 1244/2000
    0s - loss: 0.3329 - acc: 0.9200
    Epoch 1245/2000
    0s - loss: 0.3289 - acc: 0.9200
    Epoch 1246/2000
    0s - loss: 0.3354 - acc: 0.9200
    Epoch 1247/2000
    0s - loss: 0.3321 - acc: 0.9200
    Epoch 1248/2000
    0s - loss: 0.3346 - acc: 0.9200
    Epoch 1249/2000
    0s - loss: 0.3349 - acc: 0.9200
    Epoch 1250/2000
    0s - loss: 0.3310 - acc: 0.9200
    Epoch 1251/2000
    0s - loss: 0.3369 - acc: 0.9200
    Epoch 1252/2000
    0s - loss: 0.3376 - acc: 0.9200
    Epoch 1253/2000
    0s - loss: 0.3289 - acc: 0.9200
    Epoch 1254/2000
    0s - loss: 0.3275 - acc: 0.9200
    Epoch 1255/2000
    0s - loss: 0.3371 - acc: 0.9000
    Epoch 1256/2000
    0s - loss: 0.3306 - acc: 0.9200
    Epoch 1257/2000
    0s - loss: 0.3347 - acc: 0.9200
    Epoch 1258/2000
    0s - loss: 0.3307 - acc: 0.9200
    Epoch 1259/2000
    0s - loss: 0.3278 - acc: 0.9200
    Epoch 1260/2000
    0s - loss: 0.3261 - acc: 0.9200
    Epoch 1261/2000
    0s - loss: 0.3367 - acc: 0.9200
    Epoch 1262/2000
    0s - loss: 0.3346 - acc: 0.9000
    Epoch 1263/2000
    0s - loss: 0.3321 - acc: 0.9200
    Epoch 1264/2000
    0s - loss: 0.3212 - acc: 0.9200
    Epoch 1265/2000
    0s - loss: 0.3291 - acc: 0.9200
    Epoch 1266/2000
    0s - loss: 0.3364 - acc: 0.9000
    Epoch 1267/2000
    0s - loss: 0.3200 - acc: 0.9400
    Epoch 1268/2000
    0s - loss: 0.3318 - acc: 0.9200
    Epoch 1269/2000
    0s - loss: 0.3232 - acc: 0.9200
    Epoch 1270/2000
    0s - loss: 0.3221 - acc: 0.9200
    Epoch 1271/2000
    0s - loss: 0.3244 - acc: 0.9200
    Epoch 1272/2000
    0s - loss: 0.3295 - acc: 0.9000
    Epoch 1273/2000
    0s - loss: 0.3193 - acc: 0.9200
    Epoch 1274/2000
    0s - loss: 0.3244 - acc: 0.9200
    Epoch 1275/2000
    0s - loss: 0.3220 - acc: 0.9200
    Epoch 1276/2000
    0s - loss: 0.3284 - acc: 0.9200
    Epoch 1277/2000
    0s - loss: 0.3255 - acc: 0.9000
    Epoch 1278/2000
    0s - loss: 0.3237 - acc: 0.9000
    Epoch 1279/2000
    0s - loss: 0.3369 - acc: 0.9200
    Epoch 1280/2000
    0s - loss: 0.3308 - acc: 0.9000
    Epoch 1281/2000
    0s - loss: 0.3208 - acc: 0.9400
    Epoch 1282/2000
    0s - loss: 0.3323 - acc: 0.9000
    Epoch 1283/2000
    0s - loss: 0.3263 - acc: 0.9200
    Epoch 1284/2000
    0s - loss: 0.3303 - acc: 0.9200
    Epoch 1285/2000
    0s - loss: 0.3205 - acc: 0.9000
    Epoch 1286/2000
    0s - loss: 0.3162 - acc: 0.9200
    Epoch 1287/2000
    0s - loss: 0.3225 - acc: 0.9200
    Epoch 1288/2000
    0s - loss: 0.3294 - acc: 0.9000
    Epoch 1289/2000
    0s - loss: 0.3237 - acc: 0.9200
    Epoch 1290/2000
    0s - loss: 0.3219 - acc: 0.9200
    Epoch 1291/2000
    0s - loss: 0.3283 - acc: 0.9200
    Epoch 1292/2000
    0s - loss: 0.3171 - acc: 0.9200
    Epoch 1293/2000
    0s - loss: 0.3297 - acc: 0.9000
    Epoch 1294/2000
    0s - loss: 0.3215 - acc: 0.9000
    Epoch 1295/2000
    0s - loss: 0.3179 - acc: 0.9200
    Epoch 1296/2000
    0s - loss: 0.3183 - acc: 0.9200
    Epoch 1297/2000
    0s - loss: 0.3218 - acc: 0.9200
    Epoch 1298/2000
    0s - loss: 0.3244 - acc: 0.9200
    Epoch 1299/2000
    0s - loss: 0.3168 - acc: 0.9200
    Epoch 1300/2000
    0s - loss: 0.3056 - acc: 0.9200
    Epoch 1301/2000
    0s - loss: 0.3163 - acc: 0.9000
    Epoch 1302/2000
    0s - loss: 0.3204 - acc: 0.9200
    Epoch 1303/2000
    0s - loss: 0.3189 - acc: 0.9200
    Epoch 1304/2000
    0s - loss: 0.3119 - acc: 0.9200
    Epoch 1305/2000
    0s - loss: 0.3147 - acc: 0.9200
    Epoch 1306/2000
    0s - loss: 0.3146 - acc: 0.8800
    Epoch 1307/2000
    0s - loss: 0.3097 - acc: 0.9200
    Epoch 1308/2000
    0s - loss: 0.3114 - acc: 0.9200
    Epoch 1309/2000
    0s - loss: 0.3098 - acc: 0.9200
    Epoch 1310/2000
    0s - loss: 0.3090 - acc: 0.9200
    Epoch 1311/2000
    0s - loss: 0.3233 - acc: 0.8800
    Epoch 1312/2000
    0s - loss: 0.3074 - acc: 0.9200
    Epoch 1313/2000
    0s - loss: 0.3174 - acc: 0.9200
    Epoch 1314/2000
    0s - loss: 0.3089 - acc: 0.9000
    Epoch 1315/2000
    0s - loss: 0.3194 - acc: 0.9200
    Epoch 1316/2000
    0s - loss: 0.3062 - acc: 0.9200
    Epoch 1317/2000
    0s - loss: 0.3105 - acc: 0.9200
    Epoch 1318/2000
    0s - loss: 0.3150 - acc: 0.9200
    Epoch 1319/2000
    0s - loss: 0.3103 - acc: 0.9000
    Epoch 1320/2000
    0s - loss: 0.3139 - acc: 0.9200
    Epoch 1321/2000
    0s - loss: 0.3127 - acc: 0.9200
    Epoch 1322/2000
    0s - loss: 0.3090 - acc: 0.9200
    Epoch 1323/2000
    0s - loss: 0.3069 - acc: 0.9200
    Epoch 1324/2000
    0s - loss: 0.3087 - acc: 0.9200
    Epoch 1325/2000
    0s - loss: 0.3230 - acc: 0.9000
    Epoch 1326/2000
    0s - loss: 0.3003 - acc: 0.9000
    Epoch 1327/2000
    0s - loss: 0.3068 - acc: 0.9200
    Epoch 1328/2000
    0s - loss: 0.3092 - acc: 0.9200
    Epoch 1329/2000
    0s - loss: 0.3084 - acc: 0.9200
    Epoch 1330/2000
    0s - loss: 0.3048 - acc: 0.9200
    Epoch 1331/2000
    0s - loss: 0.2982 - acc: 0.9200
    Epoch 1332/2000
    0s - loss: 0.3088 - acc: 0.9200
    Epoch 1333/2000
    0s - loss: 0.3035 - acc: 0.9200
    Epoch 1334/2000
    0s - loss: 0.2992 - acc: 0.9000
    Epoch 1335/2000
    0s - loss: 0.3021 - acc: 0.9200
    Epoch 1336/2000
    0s - loss: 0.3048 - acc: 0.9200
    Epoch 1337/2000
    0s - loss: 0.3036 - acc: 0.9200
    Epoch 1338/2000
    0s - loss: 0.2966 - acc: 0.9200
    Epoch 1339/2000
    0s - loss: 0.3193 - acc: 0.9000
    Epoch 1340/2000
    0s - loss: 0.2986 - acc: 0.9200
    Epoch 1341/2000
    0s - loss: 0.3101 - acc: 0.9200
    Epoch 1342/2000
    0s - loss: 0.2981 - acc: 0.9200
    Epoch 1343/2000
    0s - loss: 0.3090 - acc: 0.9000
    Epoch 1344/2000
    0s - loss: 0.3198 - acc: 0.9000
    Epoch 1345/2000
    0s - loss: 0.3019 - acc: 0.9200
    Epoch 1346/2000
    0s - loss: 0.2970 - acc: 0.9200
    Epoch 1347/2000
    0s - loss: 0.3035 - acc: 0.9200
    Epoch 1348/2000
    0s - loss: 0.2963 - acc: 0.9000
    Epoch 1349/2000
    0s - loss: 0.2951 - acc: 0.9200
    Epoch 1350/2000
    0s - loss: 0.2936 - acc: 0.9200
    Epoch 1351/2000
    0s - loss: 0.2951 - acc: 0.9200
    Epoch 1352/2000
    0s - loss: 0.2961 - acc: 0.9200
    Epoch 1353/2000
    0s - loss: 0.2946 - acc: 0.9200
    Epoch 1354/2000
    0s - loss: 0.2954 - acc: 0.9200
    Epoch 1355/2000
    0s - loss: 0.3051 - acc: 0.9200
    Epoch 1356/2000
    0s - loss: 0.2951 - acc: 0.9200
    Epoch 1357/2000
    0s - loss: 0.3042 - acc: 0.9000
    Epoch 1358/2000
    0s - loss: 0.2964 - acc: 0.9200
    Epoch 1359/2000
    0s - loss: 0.2939 - acc: 0.9200
    Epoch 1360/2000
    0s - loss: 0.2948 - acc: 0.9200
    Epoch 1361/2000
    0s - loss: 0.2991 - acc: 0.9200
    Epoch 1362/2000
    0s - loss: 0.2948 - acc: 0.9000
    Epoch 1363/2000
    0s - loss: 0.2995 - acc: 0.9000
    Epoch 1364/2000
    0s - loss: 0.3033 - acc: 0.9200
    Epoch 1365/2000
    0s - loss: 0.3077 - acc: 0.9200
    Epoch 1366/2000
    0s - loss: 0.3169 - acc: 0.9200
    Epoch 1367/2000
    0s - loss: 0.3107 - acc: 0.9200
    Epoch 1368/2000
    0s - loss: 0.3083 - acc: 0.9200
    Epoch 1369/2000
    0s - loss: 0.3023 - acc: 0.9000
    Epoch 1370/2000
    0s - loss: 0.2957 - acc: 0.9200
    Epoch 1371/2000
    0s - loss: 0.2996 - acc: 0.9000
    Epoch 1372/2000
    0s - loss: 0.2963 - acc: 0.9200
    Epoch 1373/2000
    0s - loss: 0.2862 - acc: 0.9200
    Epoch 1374/2000
    0s - loss: 0.2930 - acc: 0.9200
    Epoch 1375/2000
    0s - loss: 0.2912 - acc: 0.9200
    Epoch 1376/2000
    0s - loss: 0.3018 - acc: 0.9200
    Epoch 1377/2000
    0s - loss: 0.2889 - acc: 0.9200
    Epoch 1378/2000
    0s - loss: 0.2926 - acc: 0.9200
    Epoch 1379/2000
    0s - loss: 0.3028 - acc: 0.9000
    Epoch 1380/2000
    0s - loss: 0.2909 - acc: 0.9200
    Epoch 1381/2000
    0s - loss: 0.2829 - acc: 0.9200
    Epoch 1382/2000
    0s - loss: 0.2851 - acc: 0.9200
    Epoch 1383/2000
    0s - loss: 0.2839 - acc: 0.9200
    Epoch 1384/2000
    0s - loss: 0.2835 - acc: 0.9200
    Epoch 1385/2000
    0s - loss: 0.2873 - acc: 0.9200
    Epoch 1386/2000
    0s - loss: 0.2828 - acc: 0.9200
    Epoch 1387/2000
    0s - loss: 0.2815 - acc: 0.9200
    Epoch 1388/2000
    0s - loss: 0.2826 - acc: 0.9200
    Epoch 1389/2000
    0s - loss: 0.2835 - acc: 0.9200
    Epoch 1390/2000
    0s - loss: 0.2817 - acc: 0.9200
    Epoch 1391/2000
    0s - loss: 0.2851 - acc: 0.9200
    Epoch 1392/2000
    0s - loss: 0.2914 - acc: 0.9200
    Epoch 1393/2000
    0s - loss: 0.2853 - acc: 0.9200
    Epoch 1394/2000
    0s - loss: 0.2886 - acc: 0.9200
    Epoch 1395/2000
    0s - loss: 0.2895 - acc: 0.9200
    Epoch 1396/2000
    0s - loss: 0.2969 - acc: 0.9200
    Epoch 1397/2000
    0s - loss: 0.2857 - acc: 0.9200
    Epoch 1398/2000
    0s - loss: 0.2902 - acc: 0.9200
    Epoch 1399/2000
    0s - loss: 0.2892 - acc: 0.9000
    Epoch 1400/2000
    0s - loss: 0.2889 - acc: 0.9200
    Epoch 1401/2000
    0s - loss: 0.2810 - acc: 0.9200
    Epoch 1402/2000
    0s - loss: 0.2859 - acc: 0.9200
    Epoch 1403/2000
    0s - loss: 0.2897 - acc: 0.9200
    Epoch 1404/2000
    0s - loss: 0.2831 - acc: 0.9000
    Epoch 1405/2000
    0s - loss: 0.2913 - acc: 0.9200
    Epoch 1406/2000
    0s - loss: 0.2789 - acc: 0.9200
    Epoch 1407/2000
    0s - loss: 0.2928 - acc: 0.9200
    Epoch 1408/2000
    0s - loss: 0.2792 - acc: 0.9400
    Epoch 1409/2000
    0s - loss: 0.2809 - acc: 0.9200
    Epoch 1410/2000
    0s - loss: 0.2802 - acc: 0.9200
    Epoch 1411/2000
    0s - loss: 0.2918 - acc: 0.9200
    Epoch 1412/2000
    0s - loss: 0.3179 - acc: 0.9000
    Epoch 1413/2000
    0s - loss: 0.2799 - acc: 0.9200
    Epoch 1414/2000
    0s - loss: 0.2867 - acc: 0.9200
    Epoch 1415/2000
    0s - loss: 0.2828 - acc: 0.9200
    Epoch 1416/2000
    0s - loss: 0.2783 - acc: 0.9200
    Epoch 1417/2000
    0s - loss: 0.2774 - acc: 0.9200
    Epoch 1418/2000
    0s - loss: 0.2846 - acc: 0.9200
    Epoch 1419/2000
    0s - loss: 0.2731 - acc: 0.9200
    Epoch 1420/2000
    0s - loss: 0.2781 - acc: 0.9000
    Epoch 1421/2000
    0s - loss: 0.2831 - acc: 0.9000
    Epoch 1422/2000
    0s - loss: 0.2787 - acc: 0.9200
    Epoch 1423/2000
    0s - loss: 0.2752 - acc: 0.9000
    Epoch 1424/2000
    0s - loss: 0.2740 - acc: 0.9200
    Epoch 1425/2000
    0s - loss: 0.2823 - acc: 0.9200
    Epoch 1426/2000
    0s - loss: 0.2775 - acc: 0.9200
    Epoch 1427/2000
    0s - loss: 0.2841 - acc: 0.9000
    Epoch 1428/2000
    0s - loss: 0.2794 - acc: 0.9000
    Epoch 1429/2000
    0s - loss: 0.2771 - acc: 0.9200
    Epoch 1430/2000
    0s - loss: 0.2759 - acc: 0.9200
    Epoch 1431/2000
    0s - loss: 0.2712 - acc: 0.9200
    Epoch 1432/2000
    0s - loss: 0.2801 - acc: 0.9000
    Epoch 1433/2000
    0s - loss: 0.2752 - acc: 0.9000
    Epoch 1434/2000
    0s - loss: 0.2714 - acc: 0.9200
    Epoch 1435/2000
    0s - loss: 0.2741 - acc: 0.9200
    Epoch 1436/2000
    0s - loss: 0.2685 - acc: 0.9000
    Epoch 1437/2000
    0s - loss: 0.2730 - acc: 0.9000
    Epoch 1438/2000
    0s - loss: 0.2673 - acc: 0.9200
    Epoch 1439/2000
    0s - loss: 0.2800 - acc: 0.9200
    Epoch 1440/2000
    0s - loss: 0.2687 - acc: 0.9400
    Epoch 1441/2000
    0s - loss: 0.2719 - acc: 0.9200
    Epoch 1442/2000
    0s - loss: 0.2727 - acc: 0.9200
    Epoch 1443/2000
    0s - loss: 0.2817 - acc: 0.9200
    Epoch 1444/2000
    0s - loss: 0.2760 - acc: 0.9000
    Epoch 1445/2000
    0s - loss: 0.2719 - acc: 0.9200
    Epoch 1446/2000
    0s - loss: 0.2746 - acc: 0.9200
    Epoch 1447/2000
    0s - loss: 0.2678 - acc: 0.9200
    Epoch 1448/2000
    0s - loss: 0.2741 - acc: 0.9000
    Epoch 1449/2000
    0s - loss: 0.2730 - acc: 0.9200
    Epoch 1450/2000
    0s - loss: 0.2777 - acc: 0.9200
    Epoch 1451/2000
    0s - loss: 0.2790 - acc: 0.9000
    Epoch 1452/2000
    0s - loss: 0.2670 - acc: 0.9000
    Epoch 1453/2000
    0s - loss: 0.2729 - acc: 0.9200
    Epoch 1454/2000
    0s - loss: 0.2727 - acc: 0.9200
    Epoch 1455/2000
    0s - loss: 0.2756 - acc: 0.9200
    Epoch 1456/2000
    0s - loss: 0.2666 - acc: 0.9200
    Epoch 1457/2000
    0s - loss: 0.2687 - acc: 0.9200
    Epoch 1458/2000
    0s - loss: 0.2697 - acc: 0.9200
    Epoch 1459/2000
    0s - loss: 0.2647 - acc: 0.9200
    Epoch 1460/2000
    0s - loss: 0.2692 - acc: 0.9200
    Epoch 1461/2000
    0s - loss: 0.2685 - acc: 0.9200
    Epoch 1462/2000
    0s - loss: 0.2789 - acc: 0.9000
    Epoch 1463/2000
    0s - loss: 0.2745 - acc: 0.9200
    Epoch 1464/2000
    0s - loss: 0.2779 - acc: 0.9200
    Epoch 1465/2000
    0s - loss: 0.2727 - acc: 0.9200
    Epoch 1466/2000
    0s - loss: 0.2639 - acc: 0.9200
    Epoch 1467/2000
    0s - loss: 0.2621 - acc: 0.9200
    Epoch 1468/2000
    0s - loss: 0.2688 - acc: 0.9000
    Epoch 1469/2000
    0s - loss: 0.2732 - acc: 0.9200
    Epoch 1470/2000
    0s - loss: 0.2855 - acc: 0.9000
    Epoch 1471/2000
    0s - loss: 0.2710 - acc: 0.9200
    Epoch 1472/2000
    0s - loss: 0.2831 - acc: 0.9200
    Epoch 1473/2000
    0s - loss: 0.2873 - acc: 0.9000
    Epoch 1474/2000
    0s - loss: 0.2823 - acc: 0.9200
    Epoch 1475/2000
    0s - loss: 0.2669 - acc: 0.9200
    Epoch 1476/2000
    0s - loss: 0.2935 - acc: 0.9200
    Epoch 1477/2000
    0s - loss: 0.2750 - acc: 0.9200
    Epoch 1478/2000
    0s - loss: 0.2689 - acc: 0.9200
    Epoch 1479/2000
    0s - loss: 0.2762 - acc: 0.9200
    Epoch 1480/2000
    0s - loss: 0.2593 - acc: 0.9200
    Epoch 1481/2000
    0s - loss: 0.2679 - acc: 0.9000
    Epoch 1482/2000
    0s - loss: 0.2641 - acc: 0.9200
    Epoch 1483/2000
    0s - loss: 0.2613 - acc: 0.9200
    Epoch 1484/2000
    0s - loss: 0.2598 - acc: 0.9200
    Epoch 1485/2000
    0s - loss: 0.2617 - acc: 0.9000
    Epoch 1486/2000
    0s - loss: 0.2723 - acc: 0.9200
    Epoch 1487/2000
    0s - loss: 0.2783 - acc: 0.9400
    Epoch 1488/2000
    0s - loss: 0.2718 - acc: 0.9200
    Epoch 1489/2000
    0s - loss: 0.2836 - acc: 0.9000
    Epoch 1490/2000
    0s - loss: 0.2827 - acc: 0.9000
    Epoch 1491/2000
    0s - loss: 0.2676 - acc: 0.9200
    Epoch 1492/2000
    0s - loss: 0.2793 - acc: 0.9000
    Epoch 1493/2000
    0s - loss: 0.2675 - acc: 0.9200
    Epoch 1494/2000
    0s - loss: 0.2737 - acc: 0.9200
    Epoch 1495/2000
    0s - loss: 0.2654 - acc: 0.9200
    Epoch 1496/2000
    0s - loss: 0.2659 - acc: 0.9200
    Epoch 1497/2000
    0s - loss: 0.2576 - acc: 0.9200
    Epoch 1498/2000
    0s - loss: 0.2608 - acc: 0.9200
    Epoch 1499/2000
    0s - loss: 0.2626 - acc: 0.9000
    Epoch 1500/2000
    0s - loss: 0.2596 - acc: 0.9000
    Epoch 1501/2000
    0s - loss: 0.2595 - acc: 0.9000
    Epoch 1502/2000
    0s - loss: 0.2584 - acc: 0.9200
    Epoch 1503/2000
    0s - loss: 0.2537 - acc: 0.9200
    Epoch 1504/2000
    0s - loss: 0.2601 - acc: 0.9200
    Epoch 1505/2000
    0s - loss: 0.2575 - acc: 0.9200
    Epoch 1506/2000
    0s - loss: 0.2714 - acc: 0.9000
    Epoch 1507/2000
    0s - loss: 0.2673 - acc: 0.9200
    Epoch 1508/2000
    0s - loss: 0.2701 - acc: 0.9200
    Epoch 1509/2000
    0s - loss: 0.2517 - acc: 0.9200
    Epoch 1510/2000
    0s - loss: 0.2564 - acc: 0.9000
    Epoch 1511/2000
    0s - loss: 0.2534 - acc: 0.9200
    Epoch 1512/2000
    0s - loss: 0.2584 - acc: 0.9200
    Epoch 1513/2000
    0s - loss: 0.2560 - acc: 0.9000
    Epoch 1514/2000
    0s - loss: 0.2569 - acc: 0.9000
    Epoch 1515/2000
    0s - loss: 0.2524 - acc: 0.9200
    Epoch 1516/2000
    0s - loss: 0.2527 - acc: 0.9200
    Epoch 1517/2000
    0s - loss: 0.2622 - acc: 0.9200
    Epoch 1518/2000
    0s - loss: 0.2544 - acc: 0.9000
    Epoch 1519/2000
    0s - loss: 0.2670 - acc: 0.9000
    Epoch 1520/2000
    0s - loss: 0.2535 - acc: 0.9600
    Epoch 1521/2000
    0s - loss: 0.2597 - acc: 0.9200
    Epoch 1522/2000
    0s - loss: 0.2575 - acc: 0.9200
    Epoch 1523/2000
    0s - loss: 0.2682 - acc: 0.9200
    Epoch 1524/2000
    0s - loss: 0.2582 - acc: 0.9000
    Epoch 1525/2000
    0s - loss: 0.2668 - acc: 0.9000
    Epoch 1526/2000
    0s - loss: 0.2615 - acc: 0.9200
    Epoch 1527/2000
    0s - loss: 0.2571 - acc: 0.9200
    Epoch 1528/2000
    0s - loss: 0.2506 - acc: 0.9400
    Epoch 1529/2000
    0s - loss: 0.2610 - acc: 0.9000
    Epoch 1530/2000
    0s - loss: 0.2564 - acc: 0.9200
    Epoch 1531/2000
    0s - loss: 0.2552 - acc: 0.9200
    Epoch 1532/2000
    0s - loss: 0.2566 - acc: 0.9200
    Epoch 1533/2000
    0s - loss: 0.2510 - acc: 0.9200
    Epoch 1534/2000
    0s - loss: 0.2543 - acc: 0.9200
    Epoch 1535/2000
    0s - loss: 0.2632 - acc: 0.9200
    Epoch 1536/2000
    0s - loss: 0.2519 - acc: 0.9200
    Epoch 1537/2000
    0s - loss: 0.2583 - acc: 0.9200
    Epoch 1538/2000
    0s - loss: 0.2607 - acc: 0.9000
    Epoch 1539/2000
    0s - loss: 0.2492 - acc: 0.9200
    Epoch 1540/2000
    0s - loss: 0.2463 - acc: 0.9200
    Epoch 1541/2000
    0s - loss: 0.2453 - acc: 0.9200
    Epoch 1542/2000
    0s - loss: 0.2446 - acc: 0.9200
    Epoch 1543/2000
    0s - loss: 0.2504 - acc: 0.9000
    Epoch 1544/2000
    0s - loss: 0.2605 - acc: 0.9200
    Epoch 1545/2000
    0s - loss: 0.2525 - acc: 0.9200
    Epoch 1546/2000
    0s - loss: 0.2502 - acc: 0.9000
    Epoch 1547/2000
    0s - loss: 0.2510 - acc: 0.9200
    Epoch 1548/2000
    0s - loss: 0.2604 - acc: 0.9000
    Epoch 1549/2000
    0s - loss: 0.2464 - acc: 0.9200
    Epoch 1550/2000
    0s - loss: 0.2686 - acc: 0.9000
    Epoch 1551/2000
    0s - loss: 0.2540 - acc: 0.9200
    Epoch 1552/2000
    0s - loss: 0.2565 - acc: 0.9200
    Epoch 1553/2000
    0s - loss: 0.2614 - acc: 0.9200
    Epoch 1554/2000
    0s - loss: 0.2535 - acc: 0.9200
    Epoch 1555/2000
    0s - loss: 0.2430 - acc: 0.9200
    Epoch 1556/2000
    0s - loss: 0.2524 - acc: 0.9200
    Epoch 1557/2000
    0s - loss: 0.2609 - acc: 0.9000
    Epoch 1558/2000
    0s - loss: 0.2500 - acc: 0.9200
    Epoch 1559/2000
    0s - loss: 0.2545 - acc: 0.9200
    Epoch 1560/2000
    0s - loss: 0.2500 - acc: 0.9200
    Epoch 1561/2000
    0s - loss: 0.2579 - acc: 0.9200
    Epoch 1562/2000
    0s - loss: 0.2666 - acc: 0.9000
    Epoch 1563/2000
    0s - loss: 0.2537 - acc: 0.9200
    Epoch 1564/2000
    0s - loss: 0.2470 - acc: 0.9400
    Epoch 1565/2000
    0s - loss: 0.2458 - acc: 0.9200
    Epoch 1566/2000
    0s - loss: 0.2414 - acc: 0.9000
    Epoch 1567/2000
    0s - loss: 0.2431 - acc: 0.9000
    Epoch 1568/2000
    0s - loss: 0.2478 - acc: 0.9200
    Epoch 1569/2000
    0s - loss: 0.2428 - acc: 0.9200
    Epoch 1570/2000
    0s - loss: 0.2453 - acc: 0.9200
    Epoch 1571/2000
    0s - loss: 0.2455 - acc: 0.9000
    Epoch 1572/2000
    0s - loss: 0.2422 - acc: 0.9000
    Epoch 1573/2000
    0s - loss: 0.2402 - acc: 0.9200
    Epoch 1574/2000
    0s - loss: 0.2514 - acc: 0.9000
    Epoch 1575/2000
    0s - loss: 0.2451 - acc: 0.9000
    Epoch 1576/2000
    0s - loss: 0.2444 - acc: 0.9200
    Epoch 1577/2000
    0s - loss: 0.2391 - acc: 0.9200
    Epoch 1578/2000
    0s - loss: 0.2451 - acc: 0.9200
    Epoch 1579/2000
    0s - loss: 0.2426 - acc: 0.9200
    Epoch 1580/2000
    0s - loss: 0.2428 - acc: 0.9200
    Epoch 1581/2000
    0s - loss: 0.2383 - acc: 0.9200
    Epoch 1582/2000
    0s - loss: 0.2437 - acc: 0.9200
    Epoch 1583/2000
    0s - loss: 0.2405 - acc: 0.9200
    Epoch 1584/2000
    0s - loss: 0.2394 - acc: 0.9200
    Epoch 1585/2000
    0s - loss: 0.2389 - acc: 0.9000
    Epoch 1586/2000
    0s - loss: 0.2396 - acc: 0.9200
    Epoch 1587/2000
    0s - loss: 0.2384 - acc: 0.9200
    Epoch 1588/2000
    0s - loss: 0.2427 - acc: 0.9200
    Epoch 1589/2000
    0s - loss: 0.2436 - acc: 0.9000
    Epoch 1590/2000
    0s - loss: 0.2521 - acc: 0.9200
    Epoch 1591/2000
    0s - loss: 0.2547 - acc: 0.9400
    Epoch 1592/2000
    0s - loss: 0.2477 - acc: 0.9000
    Epoch 1593/2000
    0s - loss: 0.2505 - acc: 0.9200
    Epoch 1594/2000
    0s - loss: 0.2429 - acc: 0.9200
    Epoch 1595/2000
    0s - loss: 0.2413 - acc: 0.9000
    Epoch 1596/2000
    0s - loss: 0.2480 - acc: 0.9200
    Epoch 1597/2000
    0s - loss: 0.2504 - acc: 0.9200
    Epoch 1598/2000
    0s - loss: 0.2465 - acc: 0.9200
    Epoch 1599/2000
    0s - loss: 0.2314 - acc: 0.9200
    Epoch 1600/2000
    0s - loss: 0.2474 - acc: 0.9200
    Epoch 1601/2000
    0s - loss: 0.2438 - acc: 0.9000
    Epoch 1602/2000
    0s - loss: 0.2377 - acc: 0.9200
    Epoch 1603/2000
    0s - loss: 0.2384 - acc: 0.9200
    Epoch 1604/2000
    0s - loss: 0.2318 - acc: 0.9200
    Epoch 1605/2000
    0s - loss: 0.2411 - acc: 0.9200
    Epoch 1606/2000
    0s - loss: 0.2357 - acc: 0.9200
    Epoch 1607/2000
    0s - loss: 0.2409 - acc: 0.9200
    Epoch 1608/2000
    0s - loss: 0.2361 - acc: 0.9200
    Epoch 1609/2000
    0s - loss: 0.2375 - acc: 0.9200
    Epoch 1610/2000
    0s - loss: 0.2362 - acc: 0.9200
    Epoch 1611/2000
    0s - loss: 0.2391 - acc: 0.9200
    Epoch 1612/2000
    0s - loss: 0.2376 - acc: 0.9200
    Epoch 1613/2000
    0s - loss: 0.2312 - acc: 0.9000
    Epoch 1614/2000
    0s - loss: 0.2385 - acc: 0.9200
    Epoch 1615/2000
    0s - loss: 0.2351 - acc: 0.9200
    Epoch 1616/2000
    0s - loss: 0.2364 - acc: 0.9200
    Epoch 1617/2000
    0s - loss: 0.2346 - acc: 0.9200
    Epoch 1618/2000
    0s - loss: 0.2443 - acc: 0.9000
    Epoch 1619/2000
    0s - loss: 0.2362 - acc: 0.9200
    Epoch 1620/2000
    0s - loss: 0.2363 - acc: 0.9200
    Epoch 1621/2000
    0s - loss: 0.2386 - acc: 0.9000
    Epoch 1622/2000
    0s - loss: 0.2394 - acc: 0.9200
    Epoch 1623/2000
    0s - loss: 0.2323 - acc: 0.9200
    Epoch 1624/2000
    0s - loss: 0.2484 - acc: 0.9200
    Epoch 1625/2000
    0s - loss: 0.2345 - acc: 0.9000
    Epoch 1626/2000
    0s - loss: 0.2373 - acc: 0.9200
    Epoch 1627/2000
    0s - loss: 0.2354 - acc: 0.9200
    Epoch 1628/2000
    0s - loss: 0.2342 - acc: 0.9200
    Epoch 1629/2000
    0s - loss: 0.2661 - acc: 0.8800
    Epoch 1630/2000
    0s - loss: 0.2388 - acc: 0.9200
    Epoch 1631/2000
    0s - loss: 0.2437 - acc: 0.9200
    Epoch 1632/2000
    0s - loss: 0.2406 - acc: 0.9000
    Epoch 1633/2000
    0s - loss: 0.2361 - acc: 0.9000
    Epoch 1634/2000
    0s - loss: 0.2422 - acc: 0.9000
    Epoch 1635/2000
    0s - loss: 0.2339 - acc: 0.9200
    Epoch 1636/2000
    0s - loss: 0.2341 - acc: 0.9200
    Epoch 1637/2000
    0s - loss: 0.2316 - acc: 0.9200
    Epoch 1638/2000
    0s - loss: 0.2344 - acc: 0.9200
    Epoch 1639/2000
    0s - loss: 0.2321 - acc: 0.9200
    Epoch 1640/2000
    0s - loss: 0.2332 - acc: 0.9200
    Epoch 1641/2000
    0s - loss: 0.2342 - acc: 0.9200
    Epoch 1642/2000
    0s - loss: 0.2285 - acc: 0.9200
    Epoch 1643/2000
    0s - loss: 0.2346 - acc: 0.9200
    Epoch 1644/2000
    0s - loss: 0.2394 - acc: 0.9200
    Epoch 1645/2000
    0s - loss: 0.2291 - acc: 0.9200
    Epoch 1646/2000
    0s - loss: 0.2431 - acc: 0.9000
    Epoch 1647/2000
    0s - loss: 0.2382 - acc: 0.9200
    Epoch 1648/2000
    0s - loss: 0.2470 - acc: 0.9200
    Epoch 1649/2000
    0s - loss: 0.2603 - acc: 0.8800
    Epoch 1650/2000
    0s - loss: 0.2363 - acc: 0.9000
    Epoch 1651/2000
    0s - loss: 0.2304 - acc: 0.9200
    Epoch 1652/2000
    0s - loss: 0.2422 - acc: 0.9000
    Epoch 1653/2000
    0s - loss: 0.2245 - acc: 0.9200
    Epoch 1654/2000
    0s - loss: 0.2313 - acc: 0.9200
    Epoch 1655/2000
    0s - loss: 0.2370 - acc: 0.8800
    Epoch 1656/2000
    0s - loss: 0.2296 - acc: 0.9200
    Epoch 1657/2000
    0s - loss: 0.2329 - acc: 0.9000
    Epoch 1658/2000
    0s - loss: 0.2243 - acc: 0.9200
    Epoch 1659/2000
    0s - loss: 0.2310 - acc: 0.9200
    Epoch 1660/2000
    0s - loss: 0.2296 - acc: 0.9200
    Epoch 1661/2000
    0s - loss: 0.2318 - acc: 0.9000
    Epoch 1662/2000
    0s - loss: 0.2257 - acc: 0.9000
    Epoch 1663/2000
    0s - loss: 0.2257 - acc: 0.9200
    Epoch 1664/2000
    0s - loss: 0.2294 - acc: 0.9000
    Epoch 1665/2000
    0s - loss: 0.2273 - acc: 0.9200
    Epoch 1666/2000
    0s - loss: 0.2324 - acc: 0.9400
    Epoch 1667/2000
    0s - loss: 0.2255 - acc: 0.9000
    Epoch 1668/2000
    0s - loss: 0.2279 - acc: 0.9200
    Epoch 1669/2000
    0s - loss: 0.2237 - acc: 0.9200
    Epoch 1670/2000
    0s - loss: 0.2272 - acc: 0.9000
    Epoch 1671/2000
    0s - loss: 0.2315 - acc: 0.9000
    Epoch 1672/2000
    0s - loss: 0.2241 - acc: 0.9000
    Epoch 1673/2000
    0s - loss: 0.2271 - acc: 0.9200
    Epoch 1674/2000
    0s - loss: 0.2317 - acc: 0.9200
    Epoch 1675/2000
    0s - loss: 0.2296 - acc: 0.9200
    Epoch 1676/2000
    0s - loss: 0.2393 - acc: 0.8800
    Epoch 1677/2000
    0s - loss: 0.2306 - acc: 0.9000
    Epoch 1678/2000
    0s - loss: 0.2447 - acc: 0.9200
    Epoch 1679/2000
    0s - loss: 0.2290 - acc: 0.9200
    Epoch 1680/2000
    0s - loss: 0.2299 - acc: 0.9200
    Epoch 1681/2000
    0s - loss: 0.2401 - acc: 0.9200
    Epoch 1682/2000
    0s - loss: 0.2323 - acc: 0.9200
    Epoch 1683/2000
    0s - loss: 0.2227 - acc: 0.9200
    Epoch 1684/2000
    0s - loss: 0.2219 - acc: 0.9200
    Epoch 1685/2000
    0s - loss: 0.2384 - acc: 0.9000
    Epoch 1686/2000
    0s - loss: 0.2275 - acc: 0.9000
    Epoch 1687/2000
    0s - loss: 0.2269 - acc: 0.9200
    Epoch 1688/2000
    0s - loss: 0.2209 - acc: 0.9200
    Epoch 1689/2000
    0s - loss: 0.2202 - acc: 0.9200
    Epoch 1690/2000
    0s - loss: 0.2192 - acc: 0.9200
    Epoch 1691/2000
    0s - loss: 0.2220 - acc: 0.9200
    Epoch 1692/2000
    0s - loss: 0.2236 - acc: 0.9200
    Epoch 1693/2000
    0s - loss: 0.2215 - acc: 0.9200
    Epoch 1694/2000
    0s - loss: 0.2200 - acc: 0.9200
    Epoch 1695/2000
    0s - loss: 0.2216 - acc: 0.9200
    Epoch 1696/2000
    0s - loss: 0.2262 - acc: 0.9000
    Epoch 1697/2000
    0s - loss: 0.2234 - acc: 0.9200
    Epoch 1698/2000
    0s - loss: 0.2270 - acc: 0.9200
    Epoch 1699/2000
    0s - loss: 0.2209 - acc: 0.9200
    Epoch 1700/2000
    0s - loss: 0.2231 - acc: 0.9000
    Epoch 1701/2000
    0s - loss: 0.2306 - acc: 0.9200
    Epoch 1702/2000
    0s - loss: 0.2169 - acc: 0.9200
    Epoch 1703/2000
    0s - loss: 0.2271 - acc: 0.9200
    Epoch 1704/2000
    0s - loss: 0.2307 - acc: 0.9200
    Epoch 1705/2000
    0s - loss: 0.2244 - acc: 0.9200
    Epoch 1706/2000
    0s - loss: 0.2137 - acc: 0.9200
    Epoch 1707/2000
    0s - loss: 0.2341 - acc: 0.9000
    Epoch 1708/2000
    0s - loss: 0.2339 - acc: 0.9200
    Epoch 1709/2000
    0s - loss: 0.2368 - acc: 0.9200
    Epoch 1710/2000
    0s - loss: 0.2223 - acc: 0.9200
    Epoch 1711/2000
    0s - loss: 0.2245 - acc: 0.9000
    Epoch 1712/2000
    0s - loss: 0.2169 - acc: 0.9200
    Epoch 1713/2000
    0s - loss: 0.2212 - acc: 0.9200
    Epoch 1714/2000
    0s - loss: 0.2219 - acc: 0.9200
    Epoch 1715/2000
    0s - loss: 0.2235 - acc: 0.9200
    Epoch 1716/2000
    0s - loss: 0.2225 - acc: 0.9000
    Epoch 1717/2000
    0s - loss: 0.2262 - acc: 0.9000
    Epoch 1718/2000
    0s - loss: 0.2275 - acc: 0.9200
    Epoch 1719/2000
    0s - loss: 0.2312 - acc: 0.9000
    Epoch 1720/2000
    0s - loss: 0.2206 - acc: 0.9200
    Epoch 1721/2000
    0s - loss: 0.2153 - acc: 0.9200
    Epoch 1722/2000
    0s - loss: 0.2283 - acc: 0.8800
    Epoch 1723/2000
    0s - loss: 0.2175 - acc: 0.9000
    Epoch 1724/2000
    0s - loss: 0.2182 - acc: 0.9200
    Epoch 1725/2000
    0s - loss: 0.2278 - acc: 0.9200
    Epoch 1726/2000
    0s - loss: 0.2329 - acc: 0.8800
    Epoch 1727/2000
    0s - loss: 0.2280 - acc: 0.9000
    Epoch 1728/2000
    0s - loss: 0.2214 - acc: 0.9200
    Epoch 1729/2000
    0s - loss: 0.2215 - acc: 0.9200
    Epoch 1730/2000
    0s - loss: 0.2229 - acc: 0.9400
    Epoch 1731/2000
    0s - loss: 0.2304 - acc: 0.9000
    Epoch 1732/2000
    0s - loss: 0.2198 - acc: 0.9000
    Epoch 1733/2000
    0s - loss: 0.2232 - acc: 0.9200
    Epoch 1734/2000
    0s - loss: 0.2285 - acc: 0.9200
    Epoch 1735/2000
    0s - loss: 0.2200 - acc: 0.9200
    Epoch 1736/2000
    0s - loss: 0.2334 - acc: 0.9200
    Epoch 1737/2000
    0s - loss: 0.2130 - acc: 0.9200
    Epoch 1738/2000
    0s - loss: 0.2112 - acc: 0.9200
    Epoch 1739/2000
    0s - loss: 0.2218 - acc: 0.9000
    Epoch 1740/2000
    0s - loss: 0.2129 - acc: 0.9200
    Epoch 1741/2000
    0s - loss: 0.2236 - acc: 0.9000
    Epoch 1742/2000
    0s - loss: 0.2222 - acc: 0.8800
    Epoch 1743/2000
    0s - loss: 0.2293 - acc: 0.9200
    Epoch 1744/2000
    0s - loss: 0.2231 - acc: 0.9200
    Epoch 1745/2000
    0s - loss: 0.2175 - acc: 0.9000
    Epoch 1746/2000
    0s - loss: 0.2229 - acc: 0.9000
    Epoch 1747/2000
    0s - loss: 0.2147 - acc: 0.9200
    Epoch 1748/2000
    0s - loss: 0.2136 - acc: 0.9200
    Epoch 1749/2000
    0s - loss: 0.2166 - acc: 0.9200
    Epoch 1750/2000
    0s - loss: 0.2186 - acc: 0.9200
    Epoch 1751/2000
    0s - loss: 0.2257 - acc: 0.9200
    Epoch 1752/2000
    0s - loss: 0.2187 - acc: 0.9400
    Epoch 1753/2000
    0s - loss: 0.2115 - acc: 0.9200
    Epoch 1754/2000
    0s - loss: 0.2118 - acc: 0.9000
    Epoch 1755/2000
    0s - loss: 0.2256 - acc: 0.9000
    Epoch 1756/2000
    0s - loss: 0.2106 - acc: 0.9200
    Epoch 1757/2000
    0s - loss: 0.2234 - acc: 0.9200
    Epoch 1758/2000
    0s - loss: 0.2098 - acc: 0.9200
    Epoch 1759/2000
    0s - loss: 0.2155 - acc: 0.9200
    Epoch 1760/2000
    0s - loss: 0.2232 - acc: 0.9200
    Epoch 1761/2000
    0s - loss: 0.2115 - acc: 0.9200
    Epoch 1762/2000
    0s - loss: 0.2146 - acc: 0.9200
    Epoch 1763/2000
    0s - loss: 0.2139 - acc: 0.9000
    Epoch 1764/2000
    0s - loss: 0.2208 - acc: 0.9200
    Epoch 1765/2000
    0s - loss: 0.2081 - acc: 0.9200
    Epoch 1766/2000
    0s - loss: 0.2113 - acc: 0.9200
    Epoch 1767/2000
    0s - loss: 0.2173 - acc: 0.9000
    Epoch 1768/2000
    0s - loss: 0.2098 - acc: 0.9200
    Epoch 1769/2000
    0s - loss: 0.2169 - acc: 0.9200
    Epoch 1770/2000
    0s - loss: 0.2075 - acc: 0.9200
    Epoch 1771/2000
    0s - loss: 0.2116 - acc: 0.9000
    Epoch 1772/2000
    0s - loss: 0.2067 - acc: 0.9200
    Epoch 1773/2000
    0s - loss: 0.2157 - acc: 0.9000
    Epoch 1774/2000
    0s - loss: 0.2190 - acc: 0.9000
    Epoch 1775/2000
    0s - loss: 0.2162 - acc: 0.9000
    Epoch 1776/2000
    0s - loss: 0.2127 - acc: 0.9200
    Epoch 1777/2000
    0s - loss: 0.2069 - acc: 0.9200
    Epoch 1778/2000
    0s - loss: 0.2113 - acc: 0.9200
    Epoch 1779/2000
    0s - loss: 0.2090 - acc: 0.9200
    Epoch 1780/2000
    0s - loss: 0.2101 - acc: 0.9200
    Epoch 1781/2000
    0s - loss: 0.2087 - acc: 0.9200
    Epoch 1782/2000
    0s - loss: 0.2085 - acc: 0.9200
    Epoch 1783/2000
    0s - loss: 0.2089 - acc: 0.9200
    Epoch 1784/2000
    0s - loss: 0.2075 - acc: 0.9000
    Epoch 1785/2000
    0s - loss: 0.2140 - acc: 0.9200
    Epoch 1786/2000
    0s - loss: 0.2077 - acc: 0.9000
    Epoch 1787/2000
    0s - loss: 0.2144 - acc: 0.9200
    Epoch 1788/2000
    0s - loss: 0.2113 - acc: 0.9200
    Epoch 1789/2000
    0s - loss: 0.2094 - acc: 0.9000
    Epoch 1790/2000
    0s - loss: 0.2172 - acc: 0.9200
    Epoch 1791/2000
    0s - loss: 0.2239 - acc: 0.8800
    Epoch 1792/2000
    0s - loss: 0.2107 - acc: 0.9200
    Epoch 1793/2000
    0s - loss: 0.2080 - acc: 0.9200
    Epoch 1794/2000
    0s - loss: 0.2167 - acc: 0.9200
    Epoch 1795/2000
    0s - loss: 0.2129 - acc: 0.9200
    Epoch 1796/2000
    0s - loss: 0.2113 - acc: 0.9000
    Epoch 1797/2000
    0s - loss: 0.2083 - acc: 0.9200
    Epoch 1798/2000
    0s - loss: 0.2080 - acc: 0.9000
    Epoch 1799/2000
    0s - loss: 0.2113 - acc: 0.9200
    Epoch 1800/2000
    0s - loss: 0.2118 - acc: 0.9200
    Epoch 1801/2000
    0s - loss: 0.2061 - acc: 0.9200
    Epoch 1802/2000
    0s - loss: 0.2113 - acc: 0.9000
    Epoch 1803/2000
    0s - loss: 0.2117 - acc: 0.9200
    Epoch 1804/2000
    0s - loss: 0.2144 - acc: 0.9200
    Epoch 1805/2000
    0s - loss: 0.2064 - acc: 0.9200
    Epoch 1806/2000
    0s - loss: 0.2060 - acc: 0.9000
    Epoch 1807/2000
    0s - loss: 0.2209 - acc: 0.9000
    Epoch 1808/2000
    0s - loss: 0.2100 - acc: 0.9000
    Epoch 1809/2000
    0s - loss: 0.2049 - acc: 0.9200
    Epoch 1810/2000
    0s - loss: 0.2070 - acc: 0.9400
    Epoch 1811/2000
    0s - loss: 0.2049 - acc: 0.9200
    Epoch 1812/2000
    0s - loss: 0.2143 - acc: 0.9000
    Epoch 1813/2000
    0s - loss: 0.2087 - acc: 0.9200
    Epoch 1814/2000
    0s - loss: 0.2045 - acc: 0.9200
    Epoch 1815/2000
    0s - loss: 0.2290 - acc: 0.9200
    Epoch 1816/2000
    0s - loss: 0.2230 - acc: 0.9200
    Epoch 1817/2000
    0s - loss: 0.2113 - acc: 0.9200
    Epoch 1818/2000
    0s - loss: 0.1977 - acc: 0.9200
    Epoch 1819/2000
    0s - loss: 0.2040 - acc: 0.9200
    Epoch 1820/2000
    0s - loss: 0.2128 - acc: 0.9200
    Epoch 1821/2000
    0s - loss: 0.2078 - acc: 0.9000
    Epoch 1822/2000
    0s - loss: 0.2044 - acc: 0.9200
    Epoch 1823/2000
    0s - loss: 0.2056 - acc: 0.9200
    Epoch 1824/2000
    0s - loss: 0.2051 - acc: 0.9200
    Epoch 1825/2000
    0s - loss: 0.2128 - acc: 0.9000
    Epoch 1826/2000
    0s - loss: 0.2063 - acc: 0.9000
    Epoch 1827/2000
    0s - loss: 0.2181 - acc: 0.9200
    Epoch 1828/2000
    0s - loss: 0.2045 - acc: 0.9200
    Epoch 1829/2000
    0s - loss: 0.2141 - acc: 0.9200
    Epoch 1830/2000
    0s - loss: 0.1994 - acc: 0.9200
    Epoch 1831/2000
    0s - loss: 0.2032 - acc: 0.9200
    Epoch 1832/2000
    0s - loss: 0.2076 - acc: 0.9200
    Epoch 1833/2000
    0s - loss: 0.2039 - acc: 0.9000
    Epoch 1834/2000
    0s - loss: 0.2068 - acc: 0.9000
    Epoch 1835/2000
    0s - loss: 0.2041 - acc: 0.9200
    Epoch 1836/2000
    0s - loss: 0.2086 - acc: 0.9000
    Epoch 1837/2000
    0s - loss: 0.2076 - acc: 0.9200
    Epoch 1838/2000
    0s - loss: 0.2076 - acc: 0.9200
    Epoch 1839/2000
    0s - loss: 0.2048 - acc: 0.9200
    Epoch 1840/2000
    0s - loss: 0.2054 - acc: 0.9000
    Epoch 1841/2000
    0s - loss: 0.2042 - acc: 0.9200
    Epoch 1842/2000
    0s - loss: 0.2073 - acc: 0.9200
    Epoch 1843/2000
    0s - loss: 0.1999 - acc: 0.9200
    Epoch 1844/2000
    0s - loss: 0.2035 - acc: 0.9200
    Epoch 1845/2000
    0s - loss: 0.2099 - acc: 0.9200
    Epoch 1846/2000
    0s - loss: 0.1981 - acc: 0.9000
    Epoch 1847/2000
    0s - loss: 0.2034 - acc: 0.9200
    Epoch 1848/2000
    0s - loss: 0.2040 - acc: 0.9200
    Epoch 1849/2000
    0s - loss: 0.1983 - acc: 0.9200
    Epoch 1850/2000
    0s - loss: 0.2158 - acc: 0.9200
    Epoch 1851/2000
    0s - loss: 0.2026 - acc: 0.9200
    Epoch 1852/2000
    0s - loss: 0.2245 - acc: 0.9200
    Epoch 1853/2000
    0s - loss: 0.2017 - acc: 0.9200
    Epoch 1854/2000
    0s - loss: 0.2040 - acc: 0.9200
    Epoch 1855/2000
    0s - loss: 0.2036 - acc: 0.9200
    Epoch 1856/2000
    0s - loss: 0.2076 - acc: 0.9200
    Epoch 1857/2000
    0s - loss: 0.2059 - acc: 0.9200
    Epoch 1858/2000
    0s - loss: 0.2015 - acc: 0.9000
    Epoch 1859/2000
    0s - loss: 0.2119 - acc: 0.9200
    Epoch 1860/2000
    0s - loss: 0.2074 - acc: 0.8800
    Epoch 1861/2000
    0s - loss: 0.2111 - acc: 0.9000
    Epoch 1862/2000
    0s - loss: 0.2141 - acc: 0.9200
    Epoch 1863/2000
    0s - loss: 0.2080 - acc: 0.9200
    Epoch 1864/2000
    0s - loss: 0.2016 - acc: 0.9200
    Epoch 1865/2000
    0s - loss: 0.1985 - acc: 0.9000
    Epoch 1866/2000
    0s - loss: 0.2165 - acc: 0.9200
    Epoch 1867/2000
    0s - loss: 0.2139 - acc: 0.9200
    Epoch 1868/2000
    0s - loss: 0.1993 - acc: 0.9200
    Epoch 1869/2000
    0s - loss: 0.2143 - acc: 0.9200
    Epoch 1870/2000
    0s - loss: 0.1988 - acc: 0.9000
    Epoch 1871/2000
    0s - loss: 0.2223 - acc: 0.9000
    Epoch 1872/2000
    0s - loss: 0.1987 - acc: 0.9200
    Epoch 1873/2000
    0s - loss: 0.2419 - acc: 0.9000
    Epoch 1874/2000
    0s - loss: 0.2392 - acc: 0.8800
    Epoch 1875/2000
    0s - loss: 0.2082 - acc: 0.9200
    Epoch 1876/2000
    0s - loss: 0.2175 - acc: 0.9000
    Epoch 1877/2000
    0s - loss: 0.2098 - acc: 0.9200
    Epoch 1878/2000
    0s - loss: 0.1998 - acc: 0.9000
    Epoch 1879/2000
    0s - loss: 0.2321 - acc: 0.9000
    Epoch 1880/2000
    0s - loss: 0.2385 - acc: 0.9000
    Epoch 1881/2000
    0s - loss: 0.2066 - acc: 0.9200
    Epoch 1882/2000
    0s - loss: 0.2003 - acc: 0.9200
    Epoch 1883/2000
    0s - loss: 0.2033 - acc: 0.9000
    Epoch 1884/2000
    0s - loss: 0.2117 - acc: 0.9200
    Epoch 1885/2000
    0s - loss: 0.2074 - acc: 0.9200
    Epoch 1886/2000
    0s - loss: 0.2040 - acc: 0.9200
    Epoch 1887/2000
    0s - loss: 0.1987 - acc: 0.9000
    Epoch 1888/2000
    0s - loss: 0.1979 - acc: 0.9000
    Epoch 1889/2000
    0s - loss: 0.2059 - acc: 0.9200
    Epoch 1890/2000
    0s - loss: 0.2048 - acc: 0.9400
    Epoch 1891/2000
    0s - loss: 0.2004 - acc: 0.9000
    Epoch 1892/2000
    0s - loss: 0.2036 - acc: 0.9200
    Epoch 1893/2000
    0s - loss: 0.1964 - acc: 0.9000
    Epoch 1894/2000
    0s - loss: 0.2095 - acc: 0.9200
    Epoch 1895/2000
    0s - loss: 0.1969 - acc: 0.9200
    Epoch 1896/2000
    0s - loss: 0.2190 - acc: 0.9200
    Epoch 1897/2000
    0s - loss: 0.2048 - acc: 0.9200
    Epoch 1898/2000
    0s - loss: 0.2071 - acc: 0.9200
    Epoch 1899/2000
    0s - loss: 0.2016 - acc: 0.9200
    Epoch 1900/2000
    0s - loss: 0.1972 - acc: 0.9200
    Epoch 1901/2000
    0s - loss: 0.2033 - acc: 0.9000
    Epoch 1902/2000
    0s - loss: 0.2019 - acc: 0.9200
    Epoch 1903/2000
    0s - loss: 0.2067 - acc: 0.9200
    Epoch 1904/2000
    0s - loss: 0.1968 - acc: 0.9200
    Epoch 1905/2000
    0s - loss: 0.2061 - acc: 0.9000
    Epoch 1906/2000
    0s - loss: 0.2108 - acc: 0.9200
    Epoch 1907/2000
    0s - loss: 0.2078 - acc: 0.9000
    Epoch 1908/2000
    0s - loss: 0.1957 - acc: 0.9000
    Epoch 1909/2000
    0s - loss: 0.2004 - acc: 0.9000
    Epoch 1910/2000
    0s - loss: 0.1989 - acc: 0.9200
    Epoch 1911/2000
    0s - loss: 0.1911 - acc: 0.9000
    Epoch 1912/2000
    0s - loss: 0.2042 - acc: 0.9000
    Epoch 1913/2000
    0s - loss: 0.1991 - acc: 0.9200
    Epoch 1914/2000
    0s - loss: 0.2001 - acc: 0.9000
    Epoch 1915/2000
    0s - loss: 0.1984 - acc: 0.9200
    Epoch 1916/2000
    0s - loss: 0.1951 - acc: 0.9200
    Epoch 1917/2000
    0s - loss: 0.2105 - acc: 0.9200
    Epoch 1918/2000
    0s - loss: 0.2029 - acc: 0.9000
    Epoch 1919/2000
    0s - loss: 0.1965 - acc: 0.9200
    Epoch 1920/2000
    0s - loss: 0.1976 - acc: 0.8800
    Epoch 1921/2000
    0s - loss: 0.1913 - acc: 0.9200
    Epoch 1922/2000
    0s - loss: 0.2198 - acc: 0.9200
    Epoch 1923/2000
    0s - loss: 0.2046 - acc: 0.9000
    Epoch 1924/2000
    0s - loss: 0.2019 - acc: 0.9000
    Epoch 1925/2000
    0s - loss: 0.1976 - acc: 0.9200
    Epoch 1926/2000
    0s - loss: 0.2017 - acc: 0.9200
    Epoch 1927/2000
    0s - loss: 0.2096 - acc: 0.9000
    Epoch 1928/2000
    0s - loss: 0.1984 - acc: 0.9000
    Epoch 1929/2000
    0s - loss: 0.1908 - acc: 0.9200
    Epoch 1930/2000
    0s - loss: 0.1947 - acc: 0.9000
    Epoch 1931/2000
    0s - loss: 0.1935 - acc: 0.9000
    Epoch 1932/2000
    0s - loss: 0.1976 - acc: 0.9200
    Epoch 1933/2000
    0s - loss: 0.1997 - acc: 0.9200
    Epoch 1934/2000
    0s - loss: 0.1987 - acc: 0.9200
    Epoch 1935/2000
    0s - loss: 0.1917 - acc: 0.9200
    Epoch 1936/2000
    0s - loss: 0.1929 - acc: 0.9200
    Epoch 1937/2000
    0s - loss: 0.1985 - acc: 0.9000
    Epoch 1938/2000
    0s - loss: 0.1981 - acc: 0.9200
    Epoch 1939/2000
    0s - loss: 0.1942 - acc: 0.9200
    Epoch 1940/2000
    0s - loss: 0.2003 - acc: 0.9200
    Epoch 1941/2000
    0s - loss: 0.1941 - acc: 0.9200
    Epoch 1942/2000
    0s - loss: 0.2069 - acc: 0.9000
    Epoch 1943/2000
    0s - loss: 0.1928 - acc: 0.9000
    Epoch 1944/2000
    0s - loss: 0.1983 - acc: 0.9200
    Epoch 1945/2000
    0s - loss: 0.1948 - acc: 0.9000
    Epoch 1946/2000
    0s - loss: 0.2006 - acc: 0.9200
    Epoch 1947/2000
    0s - loss: 0.1991 - acc: 0.9000
    Epoch 1948/2000
    0s - loss: 0.1961 - acc: 0.9000
    Epoch 1949/2000
    0s - loss: 0.2010 - acc: 0.9200
    Epoch 1950/2000
    0s - loss: 0.2089 - acc: 0.9000
    Epoch 1951/2000
    0s - loss: 0.1896 - acc: 0.9200
    Epoch 1952/2000
    0s - loss: 0.1957 - acc: 0.9200
    Epoch 1953/2000
    0s - loss: 0.1938 - acc: 0.9000
    Epoch 1954/2000
    0s - loss: 0.1949 - acc: 0.9200
    Epoch 1955/2000
    0s - loss: 0.1945 - acc: 0.9000
    Epoch 1956/2000
    0s - loss: 0.1914 - acc: 0.9200
    Epoch 1957/2000
    0s - loss: 0.2015 - acc: 0.9200
    Epoch 1958/2000
    0s - loss: 0.1973 - acc: 0.9200
    Epoch 1959/2000
    0s - loss: 0.1939 - acc: 0.9200
    Epoch 1960/2000
    0s - loss: 0.2009 - acc: 0.9000
    Epoch 1961/2000
    0s - loss: 0.1894 - acc: 0.9200
    Epoch 1962/2000
    0s - loss: 0.1977 - acc: 0.9000
    Epoch 1963/2000
    0s - loss: 0.2035 - acc: 0.9000
    Epoch 1964/2000
    0s - loss: 0.1998 - acc: 0.9000
    Epoch 1965/2000
    0s - loss: 0.1872 - acc: 0.9200
    Epoch 1966/2000
    0s - loss: 0.1928 - acc: 0.9000
    Epoch 1967/2000
    0s - loss: 0.1973 - acc: 0.9200
    Epoch 1968/2000
    0s - loss: 0.1907 - acc: 0.9000
    Epoch 1969/2000
    0s - loss: 0.1901 - acc: 0.9200
    Epoch 1970/2000
    0s - loss: 0.1919 - acc: 0.9200
    Epoch 1971/2000
    0s - loss: 0.1920 - acc: 0.9200
    Epoch 1972/2000
    0s - loss: 0.1846 - acc: 0.9200
    Epoch 1973/2000
    0s - loss: 0.1916 - acc: 0.9000
    Epoch 1974/2000
    0s - loss: 0.1898 - acc: 0.9200
    Epoch 1975/2000
    0s - loss: 0.1907 - acc: 0.9000
    Epoch 1976/2000
    0s - loss: 0.1885 - acc: 0.9200
    Epoch 1977/2000
    0s - loss: 0.1863 - acc: 0.9200
    Epoch 1978/2000
    0s - loss: 0.1888 - acc: 0.9200
    Epoch 1979/2000
    0s - loss: 0.1865 - acc: 0.9000
    Epoch 1980/2000
    0s - loss: 0.1860 - acc: 0.9200
    Epoch 1981/2000
    0s - loss: 0.1954 - acc: 0.9200
    Epoch 1982/2000
    0s - loss: 0.1980 - acc: 0.9000
    Epoch 1983/2000
    0s - loss: 0.1884 - acc: 0.9000
    Epoch 1984/2000
    0s - loss: 0.1842 - acc: 0.9200
    Epoch 1985/2000
    0s - loss: 0.1905 - acc: 0.9000
    Epoch 1986/2000
    0s - loss: 0.1938 - acc: 0.9000
    Epoch 1987/2000
    0s - loss: 0.1912 - acc: 0.9200
    Epoch 1988/2000
    0s - loss: 0.1878 - acc: 0.9200
    Epoch 1989/2000
    0s - loss: 0.2000 - acc: 0.8800
    Epoch 1990/2000
    0s - loss: 0.1949 - acc: 0.9000
    Epoch 1991/2000
    0s - loss: 0.1831 - acc: 0.9200
    Epoch 1992/2000
    0s - loss: 0.1929 - acc: 0.9200
    Epoch 1993/2000
    0s - loss: 0.1805 - acc: 0.9200
    Epoch 1994/2000
    0s - loss: 0.1878 - acc: 0.9200
    Epoch 1995/2000
    0s - loss: 0.1901 - acc: 0.9200
    Epoch 1996/2000
    0s - loss: 0.1935 - acc: 0.9000
    Epoch 1997/2000
    0s - loss: 0.1853 - acc: 0.9200
    Epoch 1998/2000
    0s - loss: 0.1941 - acc: 0.9200
    Epoch 1999/2000
    0s - loss: 0.1927 - acc: 0.9200
    Epoch 2000/2000
    0s - loss: 0.1806 - acc: 0.9200
    32/50 [==================>...........] - ETA: 0sacc: 92.00%
    ('one step prediction : ', ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'e8', 'f8', 'g8', 'g8', 'g4', 'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'e8', 'd8', 'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4'])
    ('full song prediction : ', ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8'])


한 스텝 예측 결과과 곡 전체 예측 결과를 악보로 그려보았습니다. 이 중 틀린 부분을 빨간색 박스로 표시해봤습니다. 총 50개 예측 중 4개가 틀려서 92%의 정확도가 나왔습니다. 중간에 틀릭 부분이 생기면 곡 전체를 예측하는 데 있어서는 그리 좋은 성능이 나오지 않습니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_MLP_song.png)

---

### 기본 LSTM 모델

이번에는 간단한 기본 LSTM 모델로 먼저 테스트를 해보겠습니다. 모델 구성은 다음과 같이 하였습니다.
- 128 뉴런을 가진 LSTM 레이어 1개와 Dense 레이어로 구성
- 입력은 샘플이 50개, 타임스텝이 4개, 속성이 1개로 구성
- 상태유지(stateful) 모드 비활성화

케라스에서는 아래와 같이 LSTM을 구성할 수 있습니다.


```python
model = Sequential()
model.add(LSTM(128, input_shape = (4, 1)))
model.add(Dense(one_hot_vec_size, activation='softmax'))
```

LSTM을 제대로 활용하기 위해서는 `상태유지 모드`, `배치사이즈`, `타임스텝`, `속성`에 대한 개념에 이해가 필요합니다. 본 절에서는 `타임스텝`에 대해서 먼저 알아보겠습니다. `타임스텝`이란 하나의 샘플에 포함된 시퀀스 개수입니다. 이는 앞서 살펴본 "input_length"와 동일합니다. 현재 문제에서는 매 샘플마다 4개의 값을 입력하므로 타임스텝이 4개로 지정할 수 있습니다. 즉 윈도우 크기와 동일하게 타임스텝으로 설정하면 됩니다. `속성`에 대해서는 나중에 알아보겠지만, 입력되는 음표 1개당 하나의 인덱스 값을 입력하므로 속성이 1개입니다. 나중에 이 `속성`의 개수를 다르게 해서 테스트 해보겠습니다. 인자로 "input_shape = (4, 1)'과 "input_dim = 1, input_length = 4"는 동일합니다. 설정한 LSTM 모델에 따라 입력할 데이터셋도 샘플 수, 타임스텝 수, 속성 수 형식으로 맞추어야 합니다. 따라서 앞서 구성한 train_X를 아래와 같이 형식을 변환합니다.


```python
train_X = np.reshape(train_X, (50, 4, 1)) # 샘플 수, 타임스텝 수, 속성 수
```

이 모델로 악보를 학습할 경우, 다층 퍼셉트론 모델과 동일하게 4개의 음표를 입력으로 받고, 그 다음 음표가 라벨값으로 지정됩니다. 이 과정을 곡이 마칠 때까지 반복하게 됩니다. 다층 퍼셉트론 모델과 차이점이 있다면, 다층 퍼셉트론 모델에서는 4개의 음표가 4개의 속성으로 입력되고, LSTM에서는 4개의 음표가 4개의 시퀀스 입력으로 들어갑니다. 여기서 속성은 1개입니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_train_LSTM.png)

전체 소스는 다음과 같습니다.


```python
# 코드 사전 정의

code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}

idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
            7:'c8', 8:'d8', 9:'e8', 10:'f8', 11:'g8', 12:'a8', 13:'b8'}

# 데이터셋 생성 함수

import numpy as np

def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq)-window_size):
        subset = seq[i:(i+window_size+1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)

# 시퀀스 데이터 정의

seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
       'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
       'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
       'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

# 데이터셋 생성

dataset = seq2dataset(seq, window_size = 4)

print(dataset.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# 랜덤시드 고정시키기
np.random.seed(5)

# 입력(X)과 출력(Y) 변수로 분리하기
train_X = dataset[:,0:4]
train_Y = dataset[:,4]

max_idx_value = 13

# 입력값 정규화 시키기
train_X = train_X / float(max_idx_value)

# 입력을 (샘플 수, 타입스텝, 특성 수)로 형태 변환
train_X = np.reshape(train_X, (50, 4, 1))

# 라벨값에 대한 one-hot 인코딩 수행
train_Y = np_utils.to_categorical(train_Y)

one_hot_vec_size = train_Y.shape[1]

print("one hot encoding vector size is ", one_hot_vec_size)

# 모델 구성하기
model = Sequential()
model.add(LSTM(128, input_shape = (4, 1)))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습시키기
model.fit(train_X, train_Y, epochs=2000, batch_size=1, verbose=2)

# 모델 평가하기
scores = model.evaluate(train_X, train_Y)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 예측하기

pred_count = 50 # 최대 예측 개수 정의

# 한 스텝 예측

seq_out = ['g8', 'e8', 'e4', 'f8']
pred_out = model.predict(train_X)

for i in range(pred_count):
    idx = np.argmax(pred_out[i]) # one-hot 인코딩을 인덱스 값으로 변환
    seq_out.append(idx2code[idx]) # seq_out는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장
    
print("one step prediction : ", seq_out)

# 곡 전체 예측

seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in
seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in] # 코드를 인덱스값으로 변환

for i in range(pred_count):
    sample_in = np.array(seq_in)
    sample_in = np.reshape(sample_in, (1, 4, 1)) # 샘플 수, 타입스텝 수, 속성 수
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx / float(max_idx_value))
    seq_in.pop(0)

print("full song prediction : ", seq_out)
```

    (50, 5)
    ('one hot encoding vector size is ', 12)
    Epoch 1/2000
    2s - loss: 2.4062 - acc: 0.2200
    Epoch 2/2000
    0s - loss: 2.0701 - acc: 0.3400
    Epoch 3/2000
    0s - loss: 1.9598 - acc: 0.3400
    Epoch 4/2000
    0s - loss: 1.9511 - acc: 0.3400
    Epoch 5/2000
    0s - loss: 1.9421 - acc: 0.3400
    Epoch 6/2000
    0s - loss: 1.9409 - acc: 0.3400
    Epoch 7/2000
    0s - loss: 1.9406 - acc: 0.3400
    Epoch 8/2000
    0s - loss: 1.9268 - acc: 0.3400
    Epoch 9/2000
    0s - loss: 1.9109 - acc: 0.3400
    Epoch 10/2000
    0s - loss: 1.9286 - acc: 0.3400
    Epoch 11/2000
    0s - loss: 1.9215 - acc: 0.3400
    Epoch 12/2000
    0s - loss: 1.9186 - acc: 0.3400
    Epoch 13/2000
    0s - loss: 1.9156 - acc: 0.3400
    Epoch 14/2000
    0s - loss: 1.9388 - acc: 0.3400
    Epoch 15/2000
    0s - loss: 1.9290 - acc: 0.3400
    Epoch 16/2000
    0s - loss: 1.9071 - acc: 0.3400
    Epoch 17/2000
    0s - loss: 1.9079 - acc: 0.3400
    Epoch 18/2000
    0s - loss: 1.9058 - acc: 0.3400
    Epoch 19/2000
    0s - loss: 1.9100 - acc: 0.3400
    Epoch 20/2000
    0s - loss: 1.9093 - acc: 0.3400
    Epoch 21/2000
    0s - loss: 1.9033 - acc: 0.3400
    Epoch 22/2000
    0s - loss: 1.8946 - acc: 0.3400
    Epoch 23/2000
    0s - loss: 1.9006 - acc: 0.3400
    Epoch 24/2000
    0s - loss: 1.9119 - acc: 0.3400
    Epoch 25/2000
    0s - loss: 1.8994 - acc: 0.3400
    Epoch 26/2000
    0s - loss: 1.8873 - acc: 0.3400
    Epoch 27/2000
    0s - loss: 1.9196 - acc: 0.3400
    Epoch 28/2000
    0s - loss: 1.9065 - acc: 0.3400
    Epoch 29/2000
    0s - loss: 1.8808 - acc: 0.3400
    Epoch 30/2000
    0s - loss: 1.8888 - acc: 0.3400
    Epoch 31/2000
    0s - loss: 1.9157 - acc: 0.3400
    Epoch 32/2000
    0s - loss: 1.8739 - acc: 0.3400
    Epoch 33/2000
    0s - loss: 1.8561 - acc: 0.3400
    Epoch 34/2000
    0s - loss: 1.8639 - acc: 0.3200
    Epoch 35/2000
    0s - loss: 1.8313 - acc: 0.3400
    Epoch 36/2000
    0s - loss: 1.8259 - acc: 0.3000
    Epoch 37/2000
    0s - loss: 1.8055 - acc: 0.3600
    Epoch 38/2000
    0s - loss: 1.7823 - acc: 0.3000
    Epoch 39/2000
    0s - loss: 1.7718 - acc: 0.3200
    Epoch 40/2000
    0s - loss: 1.7436 - acc: 0.3800
    Epoch 41/2000
    0s - loss: 1.7383 - acc: 0.3200
    Epoch 42/2000
    0s - loss: 1.7151 - acc: 0.3800
    Epoch 43/2000
    0s - loss: 1.6846 - acc: 0.4000
    Epoch 44/2000
    0s - loss: 1.7210 - acc: 0.3200
    Epoch 45/2000
    0s - loss: 1.7071 - acc: 0.4400
    Epoch 46/2000
    0s - loss: 1.6826 - acc: 0.4000
    Epoch 47/2000
    0s - loss: 1.6671 - acc: 0.4000
    Epoch 48/2000
    0s - loss: 1.6615 - acc: 0.4200
    Epoch 49/2000
    0s - loss: 1.6773 - acc: 0.4000
    Epoch 50/2000
    0s - loss: 1.6229 - acc: 0.4200
    Epoch 51/2000
    0s - loss: 1.6370 - acc: 0.4600
    Epoch 52/2000
    0s - loss: 1.6019 - acc: 0.4400
    Epoch 53/2000
    0s - loss: 1.6091 - acc: 0.4000
    Epoch 54/2000
    0s - loss: 1.5958 - acc: 0.4200
    Epoch 55/2000
    0s - loss: 1.5548 - acc: 0.4600
    Epoch 56/2000
    0s - loss: 1.5828 - acc: 0.3800
    Epoch 57/2000
    0s - loss: 1.5659 - acc: 0.4400
    Epoch 58/2000
    0s - loss: 1.5247 - acc: 0.4600
    Epoch 59/2000
    0s - loss: 1.6228 - acc: 0.4400
    Epoch 60/2000
    0s - loss: 1.5156 - acc: 0.4600
    Epoch 61/2000
    0s - loss: 1.5929 - acc: 0.4400
    Epoch 62/2000
    0s - loss: 1.5005 - acc: 0.4200
    Epoch 63/2000
    0s - loss: 1.5009 - acc: 0.4600
    Epoch 64/2000
    0s - loss: 1.4883 - acc: 0.4800
    Epoch 65/2000
    0s - loss: 1.4978 - acc: 0.4600
    Epoch 66/2000
    0s - loss: 1.4725 - acc: 0.4200
    Epoch 67/2000
    0s - loss: 1.4799 - acc: 0.4200
    Epoch 68/2000
    0s - loss: 1.4900 - acc: 0.4400
    Epoch 69/2000
    0s - loss: 1.4796 - acc: 0.4600
    Epoch 70/2000
    0s - loss: 1.4401 - acc: 0.5200
    Epoch 71/2000
    0s - loss: 1.4472 - acc: 0.4600
    Epoch 72/2000
    0s - loss: 1.4355 - acc: 0.4600
    Epoch 73/2000
    1s - loss: 1.4485 - acc: 0.5200
    Epoch 74/2000
    0s - loss: 1.4257 - acc: 0.4400
    Epoch 75/2000
    0s - loss: 1.4053 - acc: 0.4600
    Epoch 76/2000
    0s - loss: 1.4065 - acc: 0.5200
    Epoch 77/2000
    0s - loss: 1.3979 - acc: 0.5000
    Epoch 78/2000
    0s - loss: 1.4339 - acc: 0.4200
    Epoch 79/2000
    0s - loss: 1.3999 - acc: 0.5000
    Epoch 80/2000
    1s - loss: 1.3967 - acc: 0.5000
    Epoch 81/2000
    0s - loss: 1.4041 - acc: 0.4800
    Epoch 82/2000
    0s - loss: 1.3486 - acc: 0.5400
    Epoch 83/2000
    0s - loss: 1.3735 - acc: 0.4600
    Epoch 84/2000
    0s - loss: 1.3552 - acc: 0.5600
    Epoch 85/2000
    0s - loss: 1.3610 - acc: 0.4800
    Epoch 86/2000
    0s - loss: 1.4165 - acc: 0.4400
    Epoch 87/2000
    0s - loss: 1.3672 - acc: 0.4800
    Epoch 88/2000
    0s - loss: 1.3823 - acc: 0.5000
    Epoch 89/2000
    0s - loss: 1.3635 - acc: 0.5000
    Epoch 90/2000
    0s - loss: 1.3461 - acc: 0.5000
    Epoch 91/2000
    0s - loss: 1.3223 - acc: 0.5400
    Epoch 92/2000
    0s - loss: 1.3344 - acc: 0.4800
    Epoch 93/2000
    0s - loss: 1.3398 - acc: 0.4600
    Epoch 94/2000
    0s - loss: 1.3109 - acc: 0.4800
    Epoch 95/2000
    0s - loss: 1.3061 - acc: 0.4800
    Epoch 96/2000
    0s - loss: 1.2951 - acc: 0.5400
    Epoch 97/2000
    0s - loss: 1.3299 - acc: 0.5000
    Epoch 98/2000
    0s - loss: 1.2676 - acc: 0.5400
    Epoch 99/2000
    0s - loss: 1.2919 - acc: 0.4800
    Epoch 100/2000
    0s - loss: 1.2671 - acc: 0.4400
    Epoch 101/2000
    0s - loss: 1.2735 - acc: 0.4800
    Epoch 102/2000
    0s - loss: 1.2777 - acc: 0.5000
    Epoch 103/2000
    0s - loss: 1.2507 - acc: 0.5000
    Epoch 104/2000
    0s - loss: 1.2532 - acc: 0.4800
    Epoch 105/2000
    0s - loss: 1.2610 - acc: 0.4600
    Epoch 106/2000
    0s - loss: 1.2203 - acc: 0.5400
    Epoch 107/2000
    0s - loss: 1.2414 - acc: 0.5200
    Epoch 108/2000
    0s - loss: 1.2331 - acc: 0.5000
    Epoch 109/2000
    0s - loss: 1.2090 - acc: 0.5600
    Epoch 110/2000
    0s - loss: 1.2448 - acc: 0.4800
    Epoch 111/2000
    0s - loss: 1.1634 - acc: 0.5800
    Epoch 112/2000
    0s - loss: 1.2206 - acc: 0.5200
    Epoch 113/2000
    0s - loss: 1.1844 - acc: 0.5400
    Epoch 114/2000
    0s - loss: 1.1929 - acc: 0.5000
    Epoch 115/2000
    0s - loss: 1.1951 - acc: 0.5400
    Epoch 116/2000
    0s - loss: 1.1870 - acc: 0.5200
    Epoch 117/2000
    0s - loss: 1.1449 - acc: 0.5400
    Epoch 118/2000
    0s - loss: 1.1835 - acc: 0.5600
    Epoch 119/2000
    0s - loss: 1.1673 - acc: 0.5000
    Epoch 120/2000
    0s - loss: 1.1223 - acc: 0.5600
    Epoch 121/2000
    0s - loss: 1.1400 - acc: 0.5600
    Epoch 122/2000
    0s - loss: 1.1756 - acc: 0.5000
    Epoch 123/2000
    0s - loss: 1.1405 - acc: 0.5400
    Epoch 124/2000
    0s - loss: 1.1855 - acc: 0.5600
    Epoch 125/2000
    0s - loss: 1.1019 - acc: 0.5000
    Epoch 126/2000
    0s - loss: 1.1074 - acc: 0.5600
    Epoch 127/2000
    0s - loss: 1.0848 - acc: 0.6000
    Epoch 128/2000
    0s - loss: 1.1104 - acc: 0.5400
    Epoch 129/2000
    0s - loss: 1.0609 - acc: 0.6000
    Epoch 130/2000
    0s - loss: 1.1342 - acc: 0.5600
    Epoch 131/2000
    0s - loss: 1.0707 - acc: 0.6000
    Epoch 132/2000
    0s - loss: 1.0653 - acc: 0.6000
    Epoch 133/2000
    0s - loss: 1.0584 - acc: 0.5800
    Epoch 134/2000
    0s - loss: 1.0980 - acc: 0.5400
    Epoch 135/2000
    0s - loss: 1.0768 - acc: 0.5600
    Epoch 136/2000
    0s - loss: 1.0717 - acc: 0.5400
    Epoch 137/2000
    0s - loss: 1.0753 - acc: 0.5800
    Epoch 138/2000
    0s - loss: 1.0407 - acc: 0.6000
    Epoch 139/2000
    0s - loss: 1.0163 - acc: 0.5800
    Epoch 140/2000
    0s - loss: 1.0128 - acc: 0.6000
    Epoch 141/2000
    0s - loss: 1.0428 - acc: 0.5800
    Epoch 142/2000
    0s - loss: 1.0179 - acc: 0.5800
    Epoch 143/2000
    0s - loss: 0.9712 - acc: 0.6800
    Epoch 144/2000
    0s - loss: 0.9830 - acc: 0.5600
    Epoch 145/2000
    0s - loss: 0.9522 - acc: 0.6400
    Epoch 146/2000
    0s - loss: 0.9776 - acc: 0.5200
    Epoch 147/2000
    0s - loss: 0.9725 - acc: 0.5400
    Epoch 148/2000
    0s - loss: 0.9544 - acc: 0.6000
    Epoch 149/2000
    0s - loss: 0.9511 - acc: 0.6000
    Epoch 150/2000
    0s - loss: 0.9473 - acc: 0.6400
    Epoch 151/2000
    0s - loss: 0.8941 - acc: 0.6600
    Epoch 152/2000
    0s - loss: 0.9959 - acc: 0.4800
    Epoch 153/2000
    0s - loss: 0.9600 - acc: 0.6200
    Epoch 154/2000
    0s - loss: 0.9525 - acc: 0.5800
    Epoch 155/2000
    0s - loss: 0.9978 - acc: 0.5800
    Epoch 156/2000
    0s - loss: 0.9256 - acc: 0.5400
    Epoch 157/2000
    0s - loss: 0.9165 - acc: 0.6200
    Epoch 158/2000
    0s - loss: 0.9444 - acc: 0.6200
    Epoch 159/2000
    0s - loss: 0.8695 - acc: 0.6200
    Epoch 160/2000
    0s - loss: 0.9119 - acc: 0.5800
    Epoch 161/2000
    0s - loss: 0.8802 - acc: 0.6000
    Epoch 162/2000
    0s - loss: 0.8766 - acc: 0.6000
    Epoch 163/2000
    0s - loss: 0.8930 - acc: 0.5800
    Epoch 164/2000
    0s - loss: 0.8282 - acc: 0.6600
    Epoch 165/2000
    0s - loss: 0.9165 - acc: 0.5800
    Epoch 166/2000
    0s - loss: 0.8951 - acc: 0.5800
    Epoch 167/2000
    0s - loss: 0.8293 - acc: 0.6800
    Epoch 168/2000
    0s - loss: 0.8156 - acc: 0.6400
    Epoch 169/2000
    0s - loss: 0.8450 - acc: 0.6200
    Epoch 170/2000
    0s - loss: 0.8147 - acc: 0.6400
    Epoch 171/2000
    0s - loss: 0.8146 - acc: 0.6400
    Epoch 172/2000
    0s - loss: 0.8364 - acc: 0.6800
    Epoch 173/2000
    0s - loss: 0.8566 - acc: 0.6800
    Epoch 174/2000
    0s - loss: 0.8296 - acc: 0.6400
    Epoch 175/2000
    0s - loss: 0.7963 - acc: 0.6200
    Epoch 176/2000
    0s - loss: 0.7903 - acc: 0.7000
    Epoch 177/2000
    0s - loss: 0.8361 - acc: 0.6800
    Epoch 178/2000
    0s - loss: 0.7916 - acc: 0.7200
    Epoch 179/2000
    0s - loss: 0.8291 - acc: 0.6600
    Epoch 180/2000
    0s - loss: 0.9389 - acc: 0.6000
    Epoch 181/2000
    0s - loss: 0.8482 - acc: 0.6400
    Epoch 182/2000
    0s - loss: 0.8098 - acc: 0.6400
    Epoch 183/2000
    0s - loss: 0.7982 - acc: 0.6400
    Epoch 184/2000
    0s - loss: 0.8166 - acc: 0.6800
    Epoch 185/2000
    0s - loss: 0.8662 - acc: 0.6200
    Epoch 186/2000
    0s - loss: 0.7702 - acc: 0.6600
    Epoch 187/2000
    0s - loss: 0.7887 - acc: 0.6600
    Epoch 188/2000
    0s - loss: 0.7556 - acc: 0.6800
    Epoch 189/2000
    0s - loss: 0.7361 - acc: 0.6000
    Epoch 190/2000
    0s - loss: 0.7531 - acc: 0.6600
    Epoch 191/2000
    0s - loss: 0.7494 - acc: 0.6600
    Epoch 192/2000
    0s - loss: 0.7306 - acc: 0.6200
    Epoch 193/2000
    0s - loss: 0.7183 - acc: 0.6600
    Epoch 194/2000
    0s - loss: 0.7627 - acc: 0.6600
    Epoch 195/2000
    0s - loss: 0.7102 - acc: 0.7000
    Epoch 196/2000
    1s - loss: 0.7121 - acc: 0.6600
    Epoch 197/2000
    0s - loss: 0.7227 - acc: 0.7200
    Epoch 198/2000
    0s - loss: 0.6921 - acc: 0.6800
    Epoch 199/2000
    0s - loss: 0.6943 - acc: 0.6800
    Epoch 200/2000
    0s - loss: 0.7388 - acc: 0.7000
    Epoch 201/2000
    0s - loss: 0.6766 - acc: 0.6800
    Epoch 202/2000
    0s - loss: 0.7016 - acc: 0.7000
    Epoch 203/2000
    0s - loss: 0.7649 - acc: 0.6400
    Epoch 204/2000
    0s - loss: 0.7095 - acc: 0.7200
    Epoch 205/2000
    0s - loss: 0.7063 - acc: 0.7000
    Epoch 206/2000
    0s - loss: 0.6922 - acc: 0.7800
    Epoch 207/2000
    0s - loss: 0.6736 - acc: 0.7200
    Epoch 208/2000
    0s - loss: 0.6788 - acc: 0.7000
    Epoch 209/2000
    0s - loss: 0.6770 - acc: 0.7000
    Epoch 210/2000
    0s - loss: 0.7543 - acc: 0.6200
    Epoch 211/2000
    0s - loss: 0.6645 - acc: 0.6800
    Epoch 212/2000
    0s - loss: 0.6819 - acc: 0.7200
    Epoch 213/2000
    0s - loss: 0.6423 - acc: 0.7800
    Epoch 214/2000
    0s - loss: 0.6841 - acc: 0.7200
    Epoch 215/2000
    0s - loss: 0.6289 - acc: 0.7200
    Epoch 216/2000
    0s - loss: 0.7503 - acc: 0.6800
    Epoch 217/2000
    0s - loss: 0.6851 - acc: 0.6800
    Epoch 218/2000
    0s - loss: 0.6468 - acc: 0.7200
    Epoch 219/2000
    0s - loss: 0.6618 - acc: 0.7000
    Epoch 220/2000
    0s - loss: 0.6676 - acc: 0.7200
    Epoch 221/2000
    0s - loss: 0.6571 - acc: 0.7000
    Epoch 222/2000
    0s - loss: 0.6336 - acc: 0.7400
    Epoch 223/2000
    0s - loss: 0.6376 - acc: 0.7800
    Epoch 224/2000
    0s - loss: 0.6053 - acc: 0.7600
    Epoch 225/2000
    0s - loss: 0.6701 - acc: 0.7000
    Epoch 226/2000
    0s - loss: 0.6094 - acc: 0.7800
    Epoch 227/2000
    0s - loss: 0.6744 - acc: 0.7400
    Epoch 228/2000
    0s - loss: 0.6361 - acc: 0.7400
    Epoch 229/2000
    0s - loss: 0.6306 - acc: 0.7400
    Epoch 230/2000
    0s - loss: 0.6494 - acc: 0.7800
    Epoch 231/2000
    0s - loss: 0.5848 - acc: 0.7600
    Epoch 232/2000
    0s - loss: 0.5857 - acc: 0.7600
    Epoch 233/2000
    0s - loss: 0.6274 - acc: 0.7800
    Epoch 234/2000
    0s - loss: 0.5928 - acc: 0.7800
    Epoch 235/2000
    0s - loss: 0.5855 - acc: 0.7400
    Epoch 236/2000
    0s - loss: 0.5820 - acc: 0.8200
    Epoch 237/2000
    0s - loss: 0.6384 - acc: 0.6800
    Epoch 238/2000
    0s - loss: 0.5987 - acc: 0.7600
    Epoch 239/2000
    0s - loss: 0.5604 - acc: 0.7600
    Epoch 240/2000
    0s - loss: 0.5833 - acc: 0.7400
    Epoch 241/2000
    0s - loss: 0.6232 - acc: 0.8000
    Epoch 242/2000
    0s - loss: 0.5738 - acc: 0.7400
    Epoch 243/2000
    0s - loss: 0.6295 - acc: 0.7600
    Epoch 244/2000
    0s - loss: 0.7596 - acc: 0.6600
    Epoch 245/2000
    0s - loss: 0.6045 - acc: 0.7400
    Epoch 246/2000
    0s - loss: 0.5569 - acc: 0.8000
    Epoch 247/2000
    0s - loss: 0.5578 - acc: 0.7800
    Epoch 248/2000
    0s - loss: 0.6055 - acc: 0.7400
    Epoch 249/2000
    0s - loss: 0.5631 - acc: 0.8000
    Epoch 250/2000
    0s - loss: 0.5552 - acc: 0.7600
    Epoch 251/2000
    0s - loss: 0.5578 - acc: 0.7600
    Epoch 252/2000
    0s - loss: 0.5048 - acc: 0.8400
    Epoch 253/2000
    0s - loss: 0.6553 - acc: 0.7000
    Epoch 254/2000
    0s - loss: 0.5693 - acc: 0.7800
    Epoch 255/2000
    0s - loss: 0.6365 - acc: 0.7400
    Epoch 256/2000
    0s - loss: 0.6853 - acc: 0.7400
    Epoch 257/2000
    0s - loss: 0.5651 - acc: 0.8400
    Epoch 258/2000
    0s - loss: 0.5541 - acc: 0.8200
    Epoch 259/2000
    0s - loss: 0.5368 - acc: 0.7800
    Epoch 260/2000
    0s - loss: 0.5493 - acc: 0.7600
    Epoch 261/2000
    0s - loss: 0.5695 - acc: 0.7600
    Epoch 262/2000
    0s - loss: 0.6000 - acc: 0.7400
    Epoch 263/2000
    0s - loss: 0.5321 - acc: 0.8200
    Epoch 264/2000
    0s - loss: 0.5091 - acc: 0.7800
    Epoch 265/2000
    0s - loss: 0.5778 - acc: 0.7800
    Epoch 266/2000
    0s - loss: 0.4991 - acc: 0.8000
    Epoch 267/2000
    0s - loss: 0.5437 - acc: 0.7800
    Epoch 268/2000
    0s - loss: 0.5495 - acc: 0.8000
    Epoch 269/2000
    0s - loss: 0.6317 - acc: 0.7000
    Epoch 270/2000
    0s - loss: 0.6183 - acc: 0.7000
    Epoch 271/2000
    0s - loss: 0.5716 - acc: 0.7800
    Epoch 272/2000
    0s - loss: 0.5689 - acc: 0.7600
    Epoch 273/2000
    0s - loss: 0.5171 - acc: 0.8000
    Epoch 274/2000
    0s - loss: 0.5407 - acc: 0.7600
    Epoch 275/2000
    0s - loss: 0.5333 - acc: 0.7800
    Epoch 276/2000
    0s - loss: 0.5267 - acc: 0.8200
    Epoch 277/2000
    0s - loss: 0.5505 - acc: 0.7800
    Epoch 278/2000
    0s - loss: 0.5093 - acc: 0.8200
    Epoch 279/2000
    0s - loss: 0.5137 - acc: 0.8000
    Epoch 280/2000
    0s - loss: 0.5223 - acc: 0.7600
    Epoch 281/2000
    0s - loss: 0.4984 - acc: 0.8200
    Epoch 282/2000
    0s - loss: 0.5142 - acc: 0.7400
    Epoch 283/2000
    0s - loss: 0.5175 - acc: 0.7800
    Epoch 284/2000
    0s - loss: 0.5350 - acc: 0.7000
    Epoch 285/2000
    0s - loss: 0.5165 - acc: 0.8000
    Epoch 286/2000
    0s - loss: 0.4855 - acc: 0.8200
    Epoch 287/2000
    0s - loss: 0.5055 - acc: 0.8400
    Epoch 288/2000
    0s - loss: 0.5349 - acc: 0.7600
    Epoch 289/2000
    0s - loss: 0.5152 - acc: 0.8000
    Epoch 290/2000
    0s - loss: 0.5047 - acc: 0.7800
    Epoch 291/2000
    0s - loss: 0.5319 - acc: 0.8000
    Epoch 292/2000
    0s - loss: 0.5043 - acc: 0.8000
    Epoch 293/2000
    0s - loss: 0.5539 - acc: 0.8000
    Epoch 294/2000
    0s - loss: 0.4795 - acc: 0.8200
    Epoch 295/2000
    0s - loss: 0.4966 - acc: 0.8000
    Epoch 296/2000
    0s - loss: 0.4795 - acc: 0.8000
    Epoch 297/2000
    0s - loss: 0.5093 - acc: 0.7800
    Epoch 298/2000
    0s - loss: 0.4973 - acc: 0.7200
    Epoch 299/2000
    0s - loss: 0.4942 - acc: 0.7800
    Epoch 300/2000
    0s - loss: 0.5067 - acc: 0.7400
    Epoch 301/2000
    0s - loss: 0.4837 - acc: 0.8000
    Epoch 302/2000
    0s - loss: 0.5070 - acc: 0.8200
    Epoch 303/2000
    0s - loss: 0.5177 - acc: 0.7800
    Epoch 304/2000
    0s - loss: 0.5094 - acc: 0.8000
    Epoch 305/2000
    0s - loss: 0.4906 - acc: 0.7800
    Epoch 306/2000
    0s - loss: 0.4796 - acc: 0.8000
    Epoch 307/2000
    0s - loss: 0.5097 - acc: 0.8000
    Epoch 308/2000
    0s - loss: 0.5528 - acc: 0.7600
    Epoch 309/2000
    0s - loss: 0.4782 - acc: 0.7800
    Epoch 310/2000
    0s - loss: 0.4576 - acc: 0.8000
    Epoch 311/2000
    0s - loss: 0.5013 - acc: 0.7800
    Epoch 312/2000
    0s - loss: 0.5173 - acc: 0.8000
    Epoch 313/2000
    0s - loss: 0.4961 - acc: 0.8000
    Epoch 314/2000
    0s - loss: 0.4745 - acc: 0.7800
    Epoch 315/2000
    0s - loss: 0.4823 - acc: 0.8200
    Epoch 316/2000
    0s - loss: 0.4834 - acc: 0.7800
    Epoch 317/2000
    0s - loss: 0.4391 - acc: 0.8000
    Epoch 318/2000
    0s - loss: 0.4599 - acc: 0.8000
    Epoch 319/2000
    0s - loss: 0.4471 - acc: 0.7800
    Epoch 320/2000
    0s - loss: 0.5189 - acc: 0.8000
    Epoch 321/2000
    0s - loss: 0.4734 - acc: 0.8000
    Epoch 322/2000
    0s - loss: 0.5589 - acc: 0.7800
    Epoch 323/2000
    0s - loss: 0.4690 - acc: 0.8200
    Epoch 324/2000
    0s - loss: 0.5530 - acc: 0.7400
    Epoch 325/2000
    0s - loss: 0.4542 - acc: 0.8000
    Epoch 326/2000
    0s - loss: 0.4686 - acc: 0.8000
    Epoch 327/2000
    0s - loss: 0.4444 - acc: 0.8000
    Epoch 328/2000
    0s - loss: 0.4316 - acc: 0.8000
    Epoch 329/2000
    0s - loss: 0.8618 - acc: 0.6400
    Epoch 330/2000
    0s - loss: 0.8269 - acc: 0.6800
    Epoch 331/2000
    0s - loss: 0.5456 - acc: 0.8400
    Epoch 332/2000
    0s - loss: 0.4203 - acc: 0.8400
    Epoch 333/2000
    0s - loss: 0.4751 - acc: 0.7800
    Epoch 334/2000
    0s - loss: 0.4362 - acc: 0.8200
    Epoch 335/2000
    0s - loss: 0.4615 - acc: 0.8000
    Epoch 336/2000
    0s - loss: 0.4537 - acc: 0.7800
    Epoch 337/2000
    0s - loss: 0.4623 - acc: 0.7800
    Epoch 338/2000
    0s - loss: 0.4809 - acc: 0.8200
    Epoch 339/2000
    0s - loss: 0.4700 - acc: 0.7800
    Epoch 340/2000
    0s - loss: 0.4476 - acc: 0.7600
    Epoch 341/2000
    0s - loss: 0.4488 - acc: 0.8200
    Epoch 342/2000
    0s - loss: 0.4477 - acc: 0.8000
    Epoch 343/2000
    0s - loss: 0.4674 - acc: 0.8400
    Epoch 344/2000
    0s - loss: 0.4483 - acc: 0.8000
    Epoch 345/2000
    0s - loss: 0.4390 - acc: 0.8000
    Epoch 346/2000
    0s - loss: 0.4148 - acc: 0.8200
    Epoch 347/2000
    0s - loss: 0.4501 - acc: 0.8000
    Epoch 348/2000
    0s - loss: 0.4719 - acc: 0.7800
    Epoch 349/2000
    0s - loss: 0.4352 - acc: 0.8200
    Epoch 350/2000
    0s - loss: 0.4636 - acc: 0.7800
    Epoch 351/2000
    0s - loss: 0.4451 - acc: 0.8200
    Epoch 352/2000
    0s - loss: 0.4160 - acc: 0.8000
    Epoch 353/2000
    0s - loss: 0.4432 - acc: 0.8200
    Epoch 354/2000
    0s - loss: 0.4720 - acc: 0.8000
    Epoch 355/2000
    0s - loss: 0.4643 - acc: 0.7600
    Epoch 356/2000
    0s - loss: 0.4336 - acc: 0.7800
    Epoch 357/2000
    0s - loss: 0.4195 - acc: 0.8600
    Epoch 358/2000
    0s - loss: 0.4268 - acc: 0.8200
    Epoch 359/2000
    0s - loss: 0.4482 - acc: 0.8200
    Epoch 360/2000
    0s - loss: 0.4406 - acc: 0.8000
    Epoch 361/2000
    0s - loss: 0.5031 - acc: 0.8200
    Epoch 362/2000
    0s - loss: 0.4452 - acc: 0.7800
    Epoch 363/2000
    0s - loss: 0.4361 - acc: 0.7800
    Epoch 364/2000
    0s - loss: 0.3964 - acc: 0.8400
    Epoch 365/2000
    0s - loss: 0.3946 - acc: 0.8200
    Epoch 366/2000
    0s - loss: 0.4394 - acc: 0.8200
    Epoch 367/2000
    0s - loss: 0.4335 - acc: 0.7800
    Epoch 368/2000
    0s - loss: 0.4275 - acc: 0.8200
    Epoch 369/2000
    0s - loss: 0.4326 - acc: 0.8000
    Epoch 370/2000
    0s - loss: 0.4423 - acc: 0.7800
    Epoch 371/2000
    0s - loss: 0.4165 - acc: 0.8200
    Epoch 372/2000
    0s - loss: 0.4012 - acc: 0.8000
    Epoch 373/2000
    0s - loss: 0.4342 - acc: 0.8000
    Epoch 374/2000
    0s - loss: 0.4407 - acc: 0.7800
    Epoch 375/2000
    0s - loss: 0.4397 - acc: 0.8000
    Epoch 376/2000
    0s - loss: 0.4222 - acc: 0.7800
    Epoch 377/2000
    0s - loss: 0.4359 - acc: 0.8200
    Epoch 378/2000
    0s - loss: 0.4336 - acc: 0.7800
    Epoch 379/2000
    0s - loss: 0.4093 - acc: 0.8200
    Epoch 380/2000
    0s - loss: 0.4032 - acc: 0.8600
    Epoch 381/2000
    0s - loss: 0.4137 - acc: 0.8000
    Epoch 382/2000
    0s - loss: 0.4528 - acc: 0.8400
    Epoch 383/2000
    0s - loss: 0.4231 - acc: 0.8000
    Epoch 384/2000
    0s - loss: 0.4264 - acc: 0.8400
    Epoch 385/2000
    0s - loss: 0.4212 - acc: 0.8200
    Epoch 386/2000
    0s - loss: 0.4353 - acc: 0.8000
    Epoch 387/2000
    0s - loss: 0.4135 - acc: 0.8000
    Epoch 388/2000
    0s - loss: 0.4332 - acc: 0.7800
    Epoch 389/2000
    0s - loss: 0.4642 - acc: 0.8000
    Epoch 390/2000
    0s - loss: 0.4640 - acc: 0.8000
    Epoch 391/2000
    0s - loss: 0.4734 - acc: 0.7200
    Epoch 392/2000
    0s - loss: 0.4304 - acc: 0.8200
    Epoch 393/2000
    0s - loss: 0.4122 - acc: 0.7600
    Epoch 394/2000
    0s - loss: 0.3779 - acc: 0.8600
    Epoch 395/2000
    0s - loss: 0.4154 - acc: 0.7600
    Epoch 396/2000
    0s - loss: 0.4029 - acc: 0.8400
    Epoch 397/2000
    0s - loss: 0.4112 - acc: 0.8000
    Epoch 398/2000
    0s - loss: 0.3869 - acc: 0.8000
    Epoch 399/2000
    0s - loss: 0.4105 - acc: 0.8000
    Epoch 400/2000
    0s - loss: 0.4020 - acc: 0.8200
    Epoch 401/2000
    0s - loss: 0.4342 - acc: 0.8000
    Epoch 402/2000
    0s - loss: 0.4288 - acc: 0.8000
    Epoch 403/2000
    0s - loss: 0.4199 - acc: 0.8200
    Epoch 404/2000
    0s - loss: 0.3964 - acc: 0.8400
    Epoch 405/2000
    0s - loss: 0.4149 - acc: 0.8000
    Epoch 406/2000
    0s - loss: 0.4066 - acc: 0.8200
    Epoch 407/2000
    0s - loss: 0.3968 - acc: 0.8200
    Epoch 408/2000
    0s - loss: 0.3881 - acc: 0.8400
    Epoch 409/2000
    0s - loss: 0.4187 - acc: 0.8000
    Epoch 410/2000
    0s - loss: 0.4333 - acc: 0.8000
    Epoch 411/2000
    0s - loss: 0.4097 - acc: 0.8400
    Epoch 412/2000
    0s - loss: 0.5000 - acc: 0.7800
    Epoch 413/2000
    0s - loss: 0.4433 - acc: 0.7800
    Epoch 414/2000
    0s - loss: 0.4181 - acc: 0.8400
    Epoch 415/2000
    0s - loss: 0.4104 - acc: 0.8200
    Epoch 416/2000
    0s - loss: 0.3922 - acc: 0.8400
    Epoch 417/2000
    0s - loss: 0.4098 - acc: 0.8200
    Epoch 418/2000
    0s - loss: 0.4336 - acc: 0.8200
    Epoch 419/2000
    0s - loss: 0.4057 - acc: 0.8400
    Epoch 420/2000
    0s - loss: 0.3873 - acc: 0.8200
    Epoch 421/2000
    0s - loss: 0.3724 - acc: 0.8400
    Epoch 422/2000
    0s - loss: 0.4201 - acc: 0.8000
    Epoch 423/2000
    0s - loss: 0.4110 - acc: 0.7800
    Epoch 424/2000
    0s - loss: 0.3642 - acc: 0.8400
    Epoch 425/2000
    0s - loss: 0.3635 - acc: 0.8200
    Epoch 426/2000
    0s - loss: 0.3730 - acc: 0.8400
    Epoch 427/2000
    0s - loss: 0.3898 - acc: 0.8400
    Epoch 428/2000
    0s - loss: 0.4116 - acc: 0.8000
    Epoch 429/2000
    0s - loss: 0.3812 - acc: 0.8400
    Epoch 430/2000
    0s - loss: 0.3728 - acc: 0.8000
    Epoch 431/2000
    0s - loss: 0.4181 - acc: 0.8200
    Epoch 432/2000
    0s - loss: 0.4186 - acc: 0.8000
    Epoch 433/2000
    0s - loss: 0.3738 - acc: 0.8400
    Epoch 434/2000
    0s - loss: 0.3906 - acc: 0.8200
    Epoch 435/2000
    0s - loss: 0.3754 - acc: 0.8000
    Epoch 436/2000
    0s - loss: 0.4257 - acc: 0.8400
    Epoch 437/2000
    0s - loss: 0.3878 - acc: 0.8000
    Epoch 438/2000
    0s - loss: 0.3775 - acc: 0.8400
    Epoch 439/2000
    0s - loss: 0.3807 - acc: 0.8200
    Epoch 440/2000
    0s - loss: 0.3665 - acc: 0.8600
    Epoch 441/2000
    0s - loss: 0.3923 - acc: 0.8400
    Epoch 442/2000
    0s - loss: 0.3877 - acc: 0.8200
    Epoch 443/2000
    0s - loss: 0.4010 - acc: 0.8400
    Epoch 444/2000
    0s - loss: 0.4182 - acc: 0.8000
    Epoch 445/2000
    0s - loss: 0.3902 - acc: 0.8600
    Epoch 446/2000
    0s - loss: 0.3597 - acc: 0.8400
    Epoch 447/2000
    0s - loss: 0.3749 - acc: 0.8000
    Epoch 448/2000
    0s - loss: 0.3910 - acc: 0.8200
    Epoch 449/2000
    0s - loss: 0.3668 - acc: 0.8800
    Epoch 450/2000
    0s - loss: 0.3641 - acc: 0.8200
    Epoch 451/2000
    0s - loss: 0.4072 - acc: 0.8200
    Epoch 452/2000
    0s - loss: 0.4199 - acc: 0.8200
    Epoch 453/2000
    0s - loss: 0.4235 - acc: 0.8000
    Epoch 454/2000
    0s - loss: 0.4476 - acc: 0.8400
    Epoch 455/2000
    0s - loss: 0.5885 - acc: 0.7800
    Epoch 456/2000
    0s - loss: 0.3507 - acc: 0.8200
    Epoch 457/2000
    0s - loss: 0.3700 - acc: 0.8200
    Epoch 458/2000
    0s - loss: 0.3675 - acc: 0.8400
    Epoch 459/2000
    0s - loss: 0.3732 - acc: 0.8200
    Epoch 460/2000
    0s - loss: 0.3641 - acc: 0.8200
    Epoch 461/2000
    0s - loss: 0.3639 - acc: 0.8800
    Epoch 462/2000
    0s - loss: 0.4286 - acc: 0.8000
    Epoch 463/2000
    0s - loss: 0.3909 - acc: 0.8400
    Epoch 464/2000
    0s - loss: 0.3516 - acc: 0.8400
    Epoch 465/2000
    0s - loss: 0.3638 - acc: 0.8000
    Epoch 466/2000
    0s - loss: 0.3782 - acc: 0.8400
    Epoch 467/2000
    0s - loss: 0.3841 - acc: 0.8000
    Epoch 468/2000
    0s - loss: 0.3469 - acc: 0.8400
    Epoch 469/2000
    0s - loss: 0.3649 - acc: 0.8600
    Epoch 470/2000
    0s - loss: 0.3906 - acc: 0.8200
    Epoch 471/2000
    0s - loss: 0.4275 - acc: 0.8000
    Epoch 472/2000
    0s - loss: 0.5099 - acc: 0.7800
    Epoch 473/2000
    0s - loss: 0.4642 - acc: 0.8000
    Epoch 474/2000
    0s - loss: 0.3665 - acc: 0.8000
    Epoch 475/2000
    0s - loss: 0.3887 - acc: 0.8000
    Epoch 476/2000
    0s - loss: 0.3561 - acc: 0.8600
    Epoch 477/2000
    0s - loss: 0.3599 - acc: 0.8200
    Epoch 478/2000
    0s - loss: 0.3636 - acc: 0.8200
    Epoch 479/2000
    0s - loss: 0.3520 - acc: 0.8200
    Epoch 480/2000
    0s - loss: 0.3459 - acc: 0.8400
    Epoch 481/2000
    0s - loss: 0.3595 - acc: 0.8800
    Epoch 482/2000
    0s - loss: 0.3420 - acc: 0.8000
    Epoch 483/2000
    0s - loss: 0.3509 - acc: 0.8400
    Epoch 484/2000
    0s - loss: 0.3600 - acc: 0.8600
    Epoch 485/2000
    0s - loss: 0.3594 - acc: 0.8000
    Epoch 486/2000
    0s - loss: 0.3653 - acc: 0.8200
    Epoch 487/2000
    0s - loss: 0.3627 - acc: 0.8400
    Epoch 488/2000
    0s - loss: 0.3482 - acc: 0.8600
    Epoch 489/2000
    0s - loss: 0.3545 - acc: 0.8200
    Epoch 490/2000
    0s - loss: 0.3529 - acc: 0.8400
    Epoch 491/2000
    0s - loss: 0.3820 - acc: 0.8000
    Epoch 492/2000
    0s - loss: 0.3409 - acc: 0.8200
    Epoch 493/2000
    0s - loss: 0.3585 - acc: 0.8200
    Epoch 494/2000
    0s - loss: 0.3775 - acc: 0.8000
    Epoch 495/2000
    0s - loss: 0.4001 - acc: 0.8400
    Epoch 496/2000
    0s - loss: 0.3794 - acc: 0.8000
    Epoch 497/2000
    0s - loss: 0.3542 - acc: 0.8600
    Epoch 498/2000
    0s - loss: 0.3669 - acc: 0.8600
    Epoch 499/2000
    0s - loss: 0.3778 - acc: 0.8000
    Epoch 500/2000
    0s - loss: 0.3714 - acc: 0.8200
    Epoch 501/2000
    0s - loss: 0.3328 - acc: 0.8600
    Epoch 502/2000
    0s - loss: 0.3430 - acc: 0.8400
    Epoch 503/2000
    0s - loss: 0.3456 - acc: 0.8200
    Epoch 504/2000
    0s - loss: 0.3422 - acc: 0.8000
    Epoch 505/2000
    0s - loss: 0.3359 - acc: 0.8600
    Epoch 506/2000
    0s - loss: 0.3615 - acc: 0.8400
    Epoch 507/2000
    0s - loss: 0.3630 - acc: 0.8400
    Epoch 508/2000
    0s - loss: 0.3676 - acc: 0.8200
    Epoch 509/2000
    0s - loss: 0.3735 - acc: 0.8400
    Epoch 510/2000
    0s - loss: 0.3437 - acc: 0.8400
    Epoch 511/2000
    0s - loss: 0.3394 - acc: 0.8200
    Epoch 512/2000
    0s - loss: 0.3691 - acc: 0.8400
    Epoch 513/2000
    0s - loss: 0.3497 - acc: 0.8400
    Epoch 514/2000
    0s - loss: 0.3959 - acc: 0.8600
    Epoch 515/2000
    0s - loss: 0.4673 - acc: 0.8600
    Epoch 516/2000
    0s - loss: 0.4480 - acc: 0.7600
    Epoch 517/2000
    0s - loss: 0.4637 - acc: 0.8200
    Epoch 518/2000
    0s - loss: 0.4457 - acc: 0.8600
    Epoch 519/2000
    0s - loss: 0.3353 - acc: 0.8600
    Epoch 520/2000
    0s - loss: 0.3458 - acc: 0.8800
    Epoch 521/2000
    0s - loss: 0.3509 - acc: 0.8400
    Epoch 522/2000
    0s - loss: 0.3229 - acc: 0.8800
    Epoch 523/2000
    0s - loss: 0.3469 - acc: 0.8200
    Epoch 524/2000
    0s - loss: 0.3607 - acc: 0.8000
    Epoch 525/2000
    0s - loss: 0.3558 - acc: 0.8600
    Epoch 526/2000
    0s - loss: 0.3327 - acc: 0.8400
    Epoch 527/2000
    0s - loss: 0.3874 - acc: 0.8200
    Epoch 528/2000
    0s - loss: 0.4119 - acc: 0.8000
    Epoch 529/2000
    0s - loss: 0.6174 - acc: 0.7400
    Epoch 530/2000
    0s - loss: 0.3249 - acc: 0.8800
    Epoch 531/2000
    0s - loss: 0.3295 - acc: 0.7800
    Epoch 532/2000
    0s - loss: 0.3411 - acc: 0.8200
    Epoch 533/2000
    0s - loss: 0.3445 - acc: 0.8400
    Epoch 534/2000
    0s - loss: 0.3379 - acc: 0.8400
    Epoch 535/2000
    0s - loss: 0.3497 - acc: 0.7400
    Epoch 536/2000
    0s - loss: 0.3380 - acc: 0.8800
    Epoch 537/2000
    0s - loss: 0.3082 - acc: 0.8600
    Epoch 538/2000
    0s - loss: 0.3223 - acc: 0.8600
    Epoch 539/2000
    0s - loss: 0.3715 - acc: 0.8000
    Epoch 540/2000
    0s - loss: 0.3455 - acc: 0.8200
    Epoch 541/2000
    0s - loss: 0.3839 - acc: 0.8000
    Epoch 542/2000
    0s - loss: 0.3192 - acc: 0.8800
    Epoch 543/2000
    0s - loss: 0.3275 - acc: 0.8200
    Epoch 544/2000
    0s - loss: 0.3391 - acc: 0.8400
    Epoch 545/2000
    0s - loss: 0.3491 - acc: 0.8000
    Epoch 546/2000
    0s - loss: 0.3266 - acc: 0.8400
    Epoch 547/2000
    0s - loss: 0.3807 - acc: 0.8000
    Epoch 548/2000
    0s - loss: 0.3446 - acc: 0.8200
    Epoch 549/2000
    0s - loss: 0.4089 - acc: 0.8000
    Epoch 550/2000
    0s - loss: 0.3310 - acc: 0.8400
    Epoch 551/2000
    0s - loss: 0.3307 - acc: 0.9000
    Epoch 552/2000
    0s - loss: 0.3262 - acc: 0.8200
    Epoch 553/2000
    0s - loss: 0.3360 - acc: 0.8400
    Epoch 554/2000
    0s - loss: 0.3548 - acc: 0.8400
    Epoch 555/2000
    0s - loss: 0.3355 - acc: 0.8000
    Epoch 556/2000
    0s - loss: 0.3249 - acc: 0.8000
    Epoch 557/2000
    0s - loss: 0.3278 - acc: 0.9000
    Epoch 558/2000
    0s - loss: 0.3449 - acc: 0.8400
    Epoch 559/2000
    0s - loss: 0.3543 - acc: 0.8600
    Epoch 560/2000
    0s - loss: 0.3171 - acc: 0.8400
    Epoch 561/2000
    0s - loss: 0.3112 - acc: 0.8800
    Epoch 562/2000
    0s - loss: 0.3280 - acc: 0.8600
    Epoch 563/2000
    0s - loss: 0.3214 - acc: 0.8600
    Epoch 564/2000
    0s - loss: 0.3127 - acc: 0.8600
    Epoch 565/2000
    0s - loss: 0.3218 - acc: 0.8600
    Epoch 566/2000
    0s - loss: 0.3114 - acc: 0.8800
    Epoch 567/2000
    0s - loss: 0.3518 - acc: 0.8600
    Epoch 568/2000
    0s - loss: 0.3736 - acc: 0.7800
    Epoch 569/2000
    0s - loss: 0.5150 - acc: 0.8000
    Epoch 570/2000
    0s - loss: 0.3373 - acc: 0.8400
    Epoch 571/2000
    0s - loss: 0.3359 - acc: 0.8600
    Epoch 572/2000
    0s - loss: 0.3171 - acc: 0.8600
    Epoch 573/2000
    0s - loss: 0.3082 - acc: 0.8400
    Epoch 574/2000
    0s - loss: 0.3265 - acc: 0.8600
    Epoch 575/2000
    0s - loss: 0.3521 - acc: 0.8400
    Epoch 576/2000
    0s - loss: 0.3715 - acc: 0.7800
    Epoch 577/2000
    0s - loss: 0.3537 - acc: 0.8400
    Epoch 578/2000
    0s - loss: 0.3633 - acc: 0.8400
    Epoch 579/2000
    0s - loss: 0.3213 - acc: 0.8000
    Epoch 580/2000
    0s - loss: 0.3281 - acc: 0.8400
    Epoch 581/2000
    0s - loss: 0.3237 - acc: 0.8400
    Epoch 582/2000
    0s - loss: 0.3390 - acc: 0.8400
    Epoch 583/2000
    0s - loss: 0.3609 - acc: 0.8400
    Epoch 584/2000
    0s - loss: 0.3355 - acc: 0.8400
    Epoch 585/2000
    0s - loss: 0.3215 - acc: 0.8200
    Epoch 586/2000
    0s - loss: 0.3053 - acc: 0.8800
    Epoch 587/2000
    0s - loss: 0.3575 - acc: 0.8200
    Epoch 588/2000
    0s - loss: 0.3185 - acc: 0.8200
    Epoch 589/2000
    0s - loss: 0.3245 - acc: 0.8600
    Epoch 590/2000
    0s - loss: 0.3398 - acc: 0.8000
    Epoch 591/2000
    0s - loss: 0.3477 - acc: 0.8400
    Epoch 592/2000
    0s - loss: 0.3228 - acc: 0.8600
    Epoch 593/2000
    0s - loss: 0.3072 - acc: 0.8800
    Epoch 594/2000
    0s - loss: 0.3150 - acc: 0.8800
    Epoch 595/2000
    0s - loss: 0.3174 - acc: 0.8400
    Epoch 596/2000
    0s - loss: 0.3304 - acc: 0.8400
    Epoch 597/2000
    0s - loss: 0.3077 - acc: 0.8600
    Epoch 598/2000
    0s - loss: 0.3155 - acc: 0.8400
    Epoch 599/2000
    0s - loss: 0.3140 - acc: 0.8400
    Epoch 600/2000
    0s - loss: 0.3304 - acc: 0.8200
    Epoch 601/2000
    0s - loss: 0.3231 - acc: 0.8200
    Epoch 602/2000
    0s - loss: 0.3668 - acc: 0.8000
    Epoch 603/2000
    0s - loss: 0.3200 - acc: 0.8800
    Epoch 604/2000
    0s - loss: 0.3296 - acc: 0.8000
    Epoch 605/2000
    0s - loss: 0.3081 - acc: 0.8800
    Epoch 606/2000
    0s - loss: 0.3222 - acc: 0.8800
    Epoch 607/2000
    0s - loss: 0.2936 - acc: 0.8800
    Epoch 608/2000
    0s - loss: 0.3234 - acc: 0.8400
    Epoch 609/2000
    0s - loss: 0.3855 - acc: 0.8200
    Epoch 610/2000
    0s - loss: 0.3412 - acc: 0.8400
    Epoch 611/2000
    0s - loss: 0.3143 - acc: 0.8600
    Epoch 612/2000
    0s - loss: 0.2946 - acc: 0.8400
    Epoch 613/2000
    0s - loss: 0.3080 - acc: 0.8600
    Epoch 614/2000
    0s - loss: 0.3262 - acc: 0.8200
    Epoch 615/2000
    0s - loss: 0.3043 - acc: 0.8800
    Epoch 616/2000
    0s - loss: 0.3082 - acc: 0.8400
    Epoch 617/2000
    0s - loss: 0.3040 - acc: 0.8200
    Epoch 618/2000
    0s - loss: 0.3067 - acc: 0.8200
    Epoch 619/2000
    0s - loss: 0.3071 - acc: 0.8600
    Epoch 620/2000
    0s - loss: 0.3066 - acc: 0.8400
    Epoch 621/2000
    0s - loss: 0.3002 - acc: 0.8800
    Epoch 622/2000
    0s - loss: 0.3140 - acc: 0.8200
    Epoch 623/2000
    0s - loss: 0.3104 - acc: 0.8800
    Epoch 624/2000
    0s - loss: 0.3136 - acc: 0.8400
    Epoch 625/2000
    0s - loss: 0.3019 - acc: 0.8800
    Epoch 626/2000
    0s - loss: 0.2916 - acc: 0.8600
    Epoch 627/2000
    0s - loss: 0.2972 - acc: 0.8600
    Epoch 628/2000
    0s - loss: 0.3244 - acc: 0.8400
    Epoch 629/2000
    0s - loss: 0.4146 - acc: 0.8000
    Epoch 630/2000
    0s - loss: 0.6840 - acc: 0.8000
    Epoch 631/2000
    0s - loss: 0.4392 - acc: 0.8000
    Epoch 632/2000
    0s - loss: 0.3405 - acc: 0.8200
    Epoch 633/2000
    0s - loss: 0.2960 - acc: 0.8600
    Epoch 634/2000
    0s - loss: 0.3272 - acc: 0.8600
    Epoch 635/2000
    0s - loss: 0.3335 - acc: 0.8000
    Epoch 636/2000
    0s - loss: 0.3087 - acc: 0.8400
    Epoch 637/2000
    0s - loss: 0.2875 - acc: 0.8800
    Epoch 638/2000
    0s - loss: 0.2925 - acc: 0.8400
    Epoch 639/2000
    0s - loss: 0.3125 - acc: 0.8400
    Epoch 640/2000
    0s - loss: 0.3137 - acc: 0.8200
    Epoch 641/2000
    0s - loss: 0.2934 - acc: 0.8400
    Epoch 642/2000
    0s - loss: 0.3307 - acc: 0.8400
    Epoch 643/2000
    0s - loss: 0.2897 - acc: 0.9000
    Epoch 644/2000
    0s - loss: 0.2940 - acc: 0.8800
    Epoch 645/2000
    0s - loss: 0.2915 - acc: 0.8400
    Epoch 646/2000
    0s - loss: 0.2990 - acc: 0.8600
    Epoch 647/2000
    0s - loss: 0.2959 - acc: 0.8600
    Epoch 648/2000
    0s - loss: 0.3116 - acc: 0.8600
    Epoch 649/2000
    0s - loss: 0.3514 - acc: 0.8000
    Epoch 650/2000
    0s - loss: 0.3679 - acc: 0.8600
    Epoch 651/2000
    0s - loss: 0.3123 - acc: 0.8600
    Epoch 652/2000
    0s - loss: 0.3051 - acc: 0.8600
    Epoch 653/2000
    0s - loss: 0.2878 - acc: 0.9000
    Epoch 654/2000
    0s - loss: 0.2925 - acc: 0.8400
    Epoch 655/2000
    0s - loss: 0.3131 - acc: 0.8200
    Epoch 656/2000
    0s - loss: 0.2948 - acc: 0.8600
    Epoch 657/2000
    0s - loss: 0.2916 - acc: 0.8600
    Epoch 658/2000
    0s - loss: 0.2823 - acc: 0.8800
    Epoch 659/2000
    0s - loss: 0.2943 - acc: 0.8600
    Epoch 660/2000
    0s - loss: 0.3210 - acc: 0.8600
    Epoch 661/2000
    0s - loss: 0.2858 - acc: 0.8400
    Epoch 662/2000
    0s - loss: 0.2888 - acc: 0.8600
    Epoch 663/2000
    0s - loss: 0.2943 - acc: 0.8600
    Epoch 664/2000
    0s - loss: 0.2919 - acc: 0.8400
    Epoch 665/2000
    0s - loss: 0.4163 - acc: 0.8000
    Epoch 666/2000
    0s - loss: 0.3188 - acc: 0.8600
    Epoch 667/2000
    0s - loss: 0.3156 - acc: 0.8400
    Epoch 668/2000
    0s - loss: 0.2995 - acc: 0.8200
    Epoch 669/2000
    0s - loss: 0.2881 - acc: 0.8600
    Epoch 670/2000
    0s - loss: 0.2925 - acc: 0.8800
    Epoch 671/2000
    0s - loss: 0.3022 - acc: 0.8800
    Epoch 672/2000
    0s - loss: 0.3069 - acc: 0.8200
    Epoch 673/2000
    0s - loss: 0.3066 - acc: 0.8800
    Epoch 674/2000
    0s - loss: 0.2872 - acc: 0.8600
    Epoch 675/2000
    0s - loss: 0.3125 - acc: 0.8400
    Epoch 676/2000
    0s - loss: 0.2906 - acc: 0.8400
    Epoch 677/2000
    0s - loss: 0.2782 - acc: 0.9000
    Epoch 678/2000
    0s - loss: 0.2865 - acc: 0.8400
    Epoch 679/2000
    0s - loss: 0.2894 - acc: 0.8400
    Epoch 680/2000
    0s - loss: 0.2892 - acc: 0.8600
    Epoch 681/2000
    0s - loss: 0.2966 - acc: 0.8800
    Epoch 682/2000
    0s - loss: 0.3202 - acc: 0.8600
    Epoch 683/2000
    0s - loss: 0.3278 - acc: 0.8000
    Epoch 684/2000
    0s - loss: 0.3671 - acc: 0.8400
    Epoch 685/2000
    0s - loss: 0.5090 - acc: 0.8000
    Epoch 686/2000
    0s - loss: 0.5305 - acc: 0.8000
    Epoch 687/2000
    0s - loss: 0.4847 - acc: 0.8200
    Epoch 688/2000
    0s - loss: 0.3448 - acc: 0.8600
    Epoch 689/2000
    0s - loss: 0.3715 - acc: 0.8200
    Epoch 690/2000
    0s - loss: 0.4123 - acc: 0.8200
    Epoch 691/2000
    0s - loss: 0.2817 - acc: 0.8600
    Epoch 692/2000
    0s - loss: 0.2855 - acc: 0.8600
    Epoch 693/2000
    0s - loss: 0.2819 - acc: 0.8400
    Epoch 694/2000
    0s - loss: 0.2734 - acc: 0.8600
    Epoch 695/2000
    0s - loss: 0.2777 - acc: 0.8600
    Epoch 696/2000
    0s - loss: 0.2916 - acc: 0.8600
    Epoch 697/2000
    0s - loss: 0.2830 - acc: 0.8600
    Epoch 698/2000
    0s - loss: 0.2711 - acc: 0.9000
    Epoch 699/2000
    0s - loss: 0.2710 - acc: 0.8400
    Epoch 700/2000
    0s - loss: 0.2696 - acc: 0.9000
    Epoch 701/2000
    0s - loss: 0.2911 - acc: 0.8400
    Epoch 702/2000
    0s - loss: 0.2824 - acc: 0.8600
    Epoch 703/2000
    0s - loss: 0.2758 - acc: 0.8600
    Epoch 704/2000
    0s - loss: 0.2870 - acc: 0.8400
    Epoch 705/2000
    0s - loss: 0.2732 - acc: 0.8600
    Epoch 706/2000
    0s - loss: 0.2740 - acc: 0.8600
    Epoch 707/2000
    0s - loss: 0.2830 - acc: 0.8600
    Epoch 708/2000
    0s - loss: 0.2859 - acc: 0.8600
    Epoch 709/2000
    0s - loss: 0.2922 - acc: 0.8400
    Epoch 710/2000
    0s - loss: 0.2994 - acc: 0.8600
    Epoch 711/2000
    0s - loss: 0.2835 - acc: 0.8600
    Epoch 712/2000
    0s - loss: 0.2793 - acc: 0.8800
    Epoch 713/2000
    0s - loss: 0.2891 - acc: 0.8600
    Epoch 714/2000
    0s - loss: 0.2875 - acc: 0.8000
    Epoch 715/2000
    0s - loss: 0.2839 - acc: 0.8800
    Epoch 716/2000
    0s - loss: 0.2830 - acc: 0.8200
    Epoch 717/2000
    0s - loss: 0.2721 - acc: 0.8400
    Epoch 718/2000
    0s - loss: 0.2844 - acc: 0.8600
    Epoch 719/2000
    0s - loss: 0.2862 - acc: 0.8400
    Epoch 720/2000
    0s - loss: 0.2732 - acc: 0.8800
    Epoch 721/2000
    0s - loss: 0.2928 - acc: 0.8800
    Epoch 722/2000
    0s - loss: 0.2856 - acc: 0.8600
    Epoch 723/2000
    0s - loss: 0.2863 - acc: 0.8400
    Epoch 724/2000
    0s - loss: 0.2930 - acc: 0.8000
    Epoch 725/2000
    0s - loss: 0.2920 - acc: 0.8600
    Epoch 726/2000
    0s - loss: 0.3122 - acc: 0.8600
    Epoch 727/2000
    0s - loss: 0.4181 - acc: 0.8200
    Epoch 728/2000
    0s - loss: 0.3653 - acc: 0.8200
    Epoch 729/2000
    0s - loss: 0.2942 - acc: 0.8400
    Epoch 730/2000
    0s - loss: 0.2785 - acc: 0.8800
    Epoch 731/2000
    0s - loss: 0.2786 - acc: 0.9000
    Epoch 732/2000
    0s - loss: 0.2754 - acc: 0.8600
    Epoch 733/2000
    0s - loss: 0.2799 - acc: 0.8400
    Epoch 734/2000
    0s - loss: 0.2743 - acc: 0.8400
    Epoch 735/2000
    0s - loss: 0.2798 - acc: 0.8400
    Epoch 736/2000
    0s - loss: 0.2855 - acc: 0.8200
    Epoch 737/2000
    0s - loss: 0.2911 - acc: 0.8800
    Epoch 738/2000
    0s - loss: 0.2963 - acc: 0.8600
    Epoch 739/2000
    0s - loss: 0.3041 - acc: 0.8800
    Epoch 740/2000
    0s - loss: 0.3123 - acc: 0.8600
    Epoch 741/2000
    0s - loss: 0.2856 - acc: 0.8800
    Epoch 742/2000
    0s - loss: 0.2754 - acc: 0.8600
    Epoch 743/2000
    0s - loss: 0.2760 - acc: 0.8400
    Epoch 744/2000
    0s - loss: 0.2636 - acc: 0.8600
    Epoch 745/2000
    0s - loss: 0.2730 - acc: 0.8400
    Epoch 746/2000
    0s - loss: 0.2876 - acc: 0.8600
    Epoch 747/2000
    0s - loss: 0.2702 - acc: 0.8600
    Epoch 748/2000
    0s - loss: 0.2724 - acc: 0.8600
    Epoch 749/2000
    0s - loss: 0.2746 - acc: 0.8600
    Epoch 750/2000
    0s - loss: 0.2808 - acc: 0.8000
    Epoch 751/2000
    0s - loss: 0.2783 - acc: 0.8800
    Epoch 752/2000
    0s - loss: 0.2809 - acc: 0.8600
    Epoch 753/2000
    0s - loss: 0.2755 - acc: 0.8600
    Epoch 754/2000
    0s - loss: 0.2763 - acc: 0.8400
    Epoch 755/2000
    0s - loss: 0.2829 - acc: 0.8200
    Epoch 756/2000
    0s - loss: 0.2706 - acc: 0.8600
    Epoch 757/2000
    0s - loss: 0.2659 - acc: 0.8800
    Epoch 758/2000
    0s - loss: 0.2804 - acc: 0.8600
    Epoch 759/2000
    0s - loss: 0.2817 - acc: 0.8400
    Epoch 760/2000
    0s - loss: 0.2838 - acc: 0.8400
    Epoch 761/2000
    0s - loss: 0.2812 - acc: 0.8600
    Epoch 762/2000
    0s - loss: 0.5395 - acc: 0.8000
    Epoch 763/2000
    0s - loss: 0.5712 - acc: 0.6800
    Epoch 764/2000
    0s - loss: 1.1180 - acc: 0.7000
    Epoch 765/2000
    0s - loss: 0.6549 - acc: 0.7200
    Epoch 766/2000
    0s - loss: 0.3403 - acc: 0.8000
    Epoch 767/2000
    0s - loss: 0.2847 - acc: 0.8600
    Epoch 768/2000
    0s - loss: 0.2742 - acc: 0.8600
    Epoch 769/2000
    0s - loss: 0.2714 - acc: 0.8800
    Epoch 770/2000
    0s - loss: 0.2727 - acc: 0.8800
    Epoch 771/2000
    0s - loss: 0.2658 - acc: 0.8600
    Epoch 772/2000
    0s - loss: 0.2675 - acc: 0.8800
    Epoch 773/2000
    0s - loss: 0.2695 - acc: 0.8600
    Epoch 774/2000
    0s - loss: 0.2669 - acc: 0.8800
    Epoch 775/2000
    0s - loss: 0.2672 - acc: 0.8600
    Epoch 776/2000
    0s - loss: 0.2731 - acc: 0.8800
    Epoch 777/2000
    0s - loss: 0.2694 - acc: 0.8200
    Epoch 778/2000
    0s - loss: 0.2664 - acc: 0.8800
    Epoch 779/2000
    0s - loss: 0.2675 - acc: 0.8400
    Epoch 780/2000
    0s - loss: 0.2577 - acc: 0.8800
    Epoch 781/2000
    0s - loss: 0.2671 - acc: 0.9000
    Epoch 782/2000
    0s - loss: 0.2646 - acc: 0.8600
    Epoch 783/2000
    0s - loss: 0.2757 - acc: 0.8600
    Epoch 784/2000
    0s - loss: 0.2649 - acc: 0.8400
    Epoch 785/2000
    0s - loss: 0.2658 - acc: 0.8600
    Epoch 786/2000
    0s - loss: 0.2638 - acc: 0.8800
    Epoch 787/2000
    0s - loss: 0.2596 - acc: 0.8800
    Epoch 788/2000
    0s - loss: 0.2639 - acc: 0.8600
    Epoch 789/2000
    0s - loss: 0.2638 - acc: 0.8600
    Epoch 790/2000
    0s - loss: 0.2610 - acc: 0.8800
    Epoch 791/2000
    0s - loss: 0.2661 - acc: 0.8600
    Epoch 792/2000
    0s - loss: 0.2613 - acc: 0.8800
    Epoch 793/2000
    0s - loss: 0.2638 - acc: 0.8600
    Epoch 794/2000
    0s - loss: 0.2663 - acc: 0.8800
    Epoch 795/2000
    0s - loss: 0.2703 - acc: 0.8800
    Epoch 796/2000
    0s - loss: 0.2697 - acc: 0.8200
    Epoch 797/2000
    0s - loss: 0.2674 - acc: 0.8400
    Epoch 798/2000
    0s - loss: 0.2718 - acc: 0.8600
    Epoch 799/2000
    0s - loss: 0.2657 - acc: 0.8800
    Epoch 800/2000
    0s - loss: 0.2600 - acc: 0.8400
    Epoch 801/2000
    0s - loss: 0.2684 - acc: 0.8800
    Epoch 802/2000
    0s - loss: 0.2673 - acc: 0.8800
    Epoch 803/2000
    0s - loss: 0.2684 - acc: 0.8600
    Epoch 804/2000
    0s - loss: 0.2672 - acc: 0.8800
    Epoch 805/2000
    0s - loss: 0.2673 - acc: 0.8400
    Epoch 806/2000
    0s - loss: 0.2703 - acc: 0.8600
    Epoch 807/2000
    0s - loss: 0.2620 - acc: 0.8800
    Epoch 808/2000
    0s - loss: 0.2690 - acc: 0.8600
    Epoch 809/2000
    0s - loss: 0.3543 - acc: 0.8400
    Epoch 810/2000
    0s - loss: 0.3435 - acc: 0.8600
    Epoch 811/2000
    0s - loss: 0.3216 - acc: 0.8600
    Epoch 812/2000
    0s - loss: 0.3002 - acc: 0.8800
    Epoch 813/2000
    0s - loss: 0.2862 - acc: 0.8600
    Epoch 814/2000
    0s - loss: 0.2933 - acc: 0.8400
    Epoch 815/2000
    0s - loss: 0.2759 - acc: 0.8600
    Epoch 816/2000
    0s - loss: 0.2755 - acc: 0.7800
    Epoch 817/2000
    0s - loss: 0.2629 - acc: 0.9000
    Epoch 818/2000
    0s - loss: 0.2626 - acc: 0.8600
    Epoch 819/2000
    0s - loss: 0.2808 - acc: 0.8200
    Epoch 820/2000
    0s - loss: 0.2769 - acc: 0.8600
    Epoch 821/2000
    0s - loss: 0.2544 - acc: 0.9000
    Epoch 822/2000
    0s - loss: 0.2683 - acc: 0.8400
    Epoch 823/2000
    0s - loss: 0.2573 - acc: 0.9000
    Epoch 824/2000
    0s - loss: 0.2616 - acc: 0.8800
    Epoch 825/2000
    0s - loss: 0.2563 - acc: 0.8600
    Epoch 826/2000
    0s - loss: 0.2577 - acc: 0.8800
    Epoch 827/2000
    0s - loss: 0.2540 - acc: 0.9000
    Epoch 828/2000
    0s - loss: 0.2475 - acc: 0.8600
    Epoch 829/2000
    0s - loss: 0.2679 - acc: 0.8400
    Epoch 830/2000
    0s - loss: 0.2599 - acc: 0.8800
    Epoch 831/2000
    0s - loss: 0.2586 - acc: 0.8600
    Epoch 832/2000
    0s - loss: 0.2566 - acc: 0.8600
    Epoch 833/2000
    0s - loss: 0.2511 - acc: 0.8800
    Epoch 834/2000
    0s - loss: 0.2637 - acc: 0.8800
    Epoch 835/2000
    0s - loss: 0.2598 - acc: 0.8600
    Epoch 836/2000
    0s - loss: 0.2648 - acc: 0.9000
    Epoch 837/2000
    0s - loss: 0.2594 - acc: 0.8800
    Epoch 838/2000
    0s - loss: 0.2600 - acc: 0.8600
    Epoch 839/2000
    0s - loss: 0.2597 - acc: 0.8800
    Epoch 840/2000
    0s - loss: 0.2640 - acc: 0.8800
    Epoch 841/2000
    0s - loss: 0.2595 - acc: 0.8600
    Epoch 842/2000
    0s - loss: 0.2678 - acc: 0.8400
    Epoch 843/2000
    0s - loss: 0.2522 - acc: 0.8600
    Epoch 844/2000
    0s - loss: 0.2582 - acc: 0.8600
    Epoch 845/2000
    0s - loss: 0.2668 - acc: 0.8800
    Epoch 846/2000
    0s - loss: 0.2611 - acc: 0.8800
    Epoch 847/2000
    0s - loss: 0.2646 - acc: 0.8600
    Epoch 848/2000
    0s - loss: 0.2628 - acc: 0.8600
    Epoch 849/2000
    0s - loss: 0.2662 - acc: 0.8400
    Epoch 850/2000
    0s - loss: 0.2565 - acc: 0.9200
    Epoch 851/2000
    0s - loss: 0.2721 - acc: 0.8400
    Epoch 852/2000
    0s - loss: 0.2556 - acc: 0.8600
    Epoch 853/2000
    0s - loss: 0.2617 - acc: 0.8600
    Epoch 854/2000
    0s - loss: 0.2693 - acc: 0.8400
    Epoch 855/2000
    0s - loss: 0.2680 - acc: 0.8400
    Epoch 856/2000
    0s - loss: 0.2523 - acc: 0.9000
    Epoch 857/2000
    0s - loss: 0.2589 - acc: 0.8800
    Epoch 858/2000
    0s - loss: 0.2677 - acc: 0.8800
    Epoch 859/2000
    0s - loss: 0.2504 - acc: 0.9000
    Epoch 860/2000
    0s - loss: 0.2568 - acc: 0.8800
    Epoch 861/2000
    0s - loss: 0.2509 - acc: 0.8400
    Epoch 862/2000
    0s - loss: 0.2488 - acc: 0.9000
    Epoch 863/2000
    0s - loss: 0.2561 - acc: 0.8600
    Epoch 864/2000
    0s - loss: 0.2734 - acc: 0.9000
    Epoch 865/2000
    0s - loss: 0.5168 - acc: 0.8200
    Epoch 866/2000
    0s - loss: 0.4612 - acc: 0.8000
    Epoch 867/2000
    0s - loss: 0.4630 - acc: 0.8000
    Epoch 868/2000
    0s - loss: 0.5498 - acc: 0.8000
    Epoch 869/2000
    0s - loss: 0.2820 - acc: 0.8800
    Epoch 870/2000
    0s - loss: 0.2533 - acc: 0.8800
    Epoch 871/2000
    0s - loss: 0.2544 - acc: 0.8600
    Epoch 872/2000
    0s - loss: 0.2605 - acc: 0.8800
    Epoch 873/2000
    0s - loss: 0.2589 - acc: 0.8400
    Epoch 874/2000
    0s - loss: 0.2538 - acc: 0.8800
    Epoch 875/2000
    0s - loss: 0.2473 - acc: 0.8800
    Epoch 876/2000
    0s - loss: 0.2556 - acc: 0.8800
    Epoch 877/2000
    0s - loss: 0.2500 - acc: 0.9000
    Epoch 878/2000
    0s - loss: 0.2497 - acc: 0.8400
    Epoch 879/2000
    0s - loss: 0.2474 - acc: 0.8600
    Epoch 880/2000
    0s - loss: 0.2532 - acc: 0.8800
    Epoch 881/2000
    0s - loss: 0.2573 - acc: 0.9000
    Epoch 882/2000
    0s - loss: 0.2482 - acc: 0.8800
    Epoch 883/2000
    0s - loss: 0.2467 - acc: 0.8800
    Epoch 884/2000
    0s - loss: 0.2531 - acc: 0.8800
    Epoch 885/2000
    0s - loss: 0.2514 - acc: 0.8600
    Epoch 886/2000
    0s - loss: 0.2602 - acc: 0.9000
    Epoch 887/2000
    0s - loss: 0.2482 - acc: 0.8600
    Epoch 888/2000
    0s - loss: 0.2535 - acc: 0.8600
    Epoch 889/2000
    0s - loss: 0.2485 - acc: 0.8800
    Epoch 890/2000
    0s - loss: 0.2548 - acc: 0.8200
    Epoch 891/2000
    0s - loss: 0.2533 - acc: 0.8600
    Epoch 892/2000
    0s - loss: 0.2540 - acc: 0.8400
    Epoch 893/2000
    0s - loss: 0.2529 - acc: 0.8800
    Epoch 894/2000
    0s - loss: 0.2476 - acc: 0.8600
    Epoch 895/2000
    0s - loss: 0.2481 - acc: 0.8600
    Epoch 896/2000
    0s - loss: 0.2536 - acc: 0.8400
    Epoch 897/2000
    0s - loss: 0.2474 - acc: 0.8800
    Epoch 898/2000
    0s - loss: 0.2533 - acc: 0.8600
    Epoch 899/2000
    0s - loss: 0.2522 - acc: 0.8600
    Epoch 900/2000
    0s - loss: 0.2534 - acc: 0.8600
    Epoch 901/2000
    0s - loss: 0.2484 - acc: 0.9000
    Epoch 902/2000
    0s - loss: 0.2519 - acc: 0.8600
    Epoch 903/2000
    0s - loss: 0.2479 - acc: 0.8800
    Epoch 904/2000
    0s - loss: 0.2537 - acc: 0.9000
    Epoch 905/2000
    0s - loss: 0.2507 - acc: 0.8600
    Epoch 906/2000
    0s - loss: 0.2425 - acc: 0.8600
    Epoch 907/2000
    0s - loss: 0.2465 - acc: 0.8800
    Epoch 908/2000
    0s - loss: 0.2553 - acc: 0.8800
    Epoch 909/2000
    0s - loss: 0.2599 - acc: 0.8600
    Epoch 910/2000
    0s - loss: 0.2619 - acc: 0.8200
    Epoch 911/2000
    0s - loss: 0.2718 - acc: 0.8800
    Epoch 912/2000
    0s - loss: 0.2502 - acc: 0.8600
    Epoch 913/2000
    0s - loss: 0.2488 - acc: 0.8800
    Epoch 914/2000
    0s - loss: 0.2377 - acc: 0.8600
    Epoch 915/2000
    0s - loss: 0.2584 - acc: 0.8400
    Epoch 916/2000
    0s - loss: 0.2689 - acc: 0.8600
    Epoch 917/2000
    0s - loss: 0.2578 - acc: 0.8600
    Epoch 918/2000
    0s - loss: 0.2495 - acc: 0.8400
    Epoch 919/2000
    0s - loss: 0.2468 - acc: 0.8800
    Epoch 920/2000
    0s - loss: 0.2613 - acc: 0.8800
    Epoch 921/2000
    0s - loss: 0.3806 - acc: 0.8200
    Epoch 922/2000
    0s - loss: 0.4735 - acc: 0.7800
    Epoch 923/2000
    0s - loss: 0.3717 - acc: 0.8400
    Epoch 924/2000
    0s - loss: 0.2730 - acc: 0.8800
    Epoch 925/2000
    0s - loss: 0.2571 - acc: 0.8400
    Epoch 926/2000
    0s - loss: 0.2568 - acc: 0.8600
    Epoch 927/2000
    0s - loss: 0.2479 - acc: 0.8600
    Epoch 928/2000
    0s - loss: 0.2493 - acc: 0.8800
    Epoch 929/2000
    0s - loss: 0.2438 - acc: 0.8600
    Epoch 930/2000
    0s - loss: 0.2485 - acc: 0.8600
    Epoch 931/2000
    0s - loss: 0.2415 - acc: 0.8600
    Epoch 932/2000
    0s - loss: 0.2411 - acc: 0.8800
    Epoch 933/2000
    0s - loss: 0.2451 - acc: 0.9000
    Epoch 934/2000
    0s - loss: 0.2435 - acc: 0.8600
    Epoch 935/2000
    0s - loss: 0.2390 - acc: 0.8600
    Epoch 936/2000
    0s - loss: 0.2415 - acc: 0.9000
    Epoch 937/2000
    0s - loss: 0.2398 - acc: 0.8400
    Epoch 938/2000
    0s - loss: 0.2396 - acc: 0.8800
    Epoch 939/2000
    0s - loss: 0.2414 - acc: 0.8600
    Epoch 940/2000
    0s - loss: 0.2436 - acc: 0.8800
    Epoch 941/2000
    0s - loss: 0.2380 - acc: 0.8600
    Epoch 942/2000
    0s - loss: 0.2510 - acc: 0.8800
    Epoch 943/2000
    0s - loss: 0.2413 - acc: 0.8600
    Epoch 944/2000
    0s - loss: 0.2443 - acc: 0.8800
    Epoch 945/2000
    0s - loss: 0.2450 - acc: 0.8600
    Epoch 946/2000
    0s - loss: 0.2426 - acc: 0.8600
    Epoch 947/2000
    0s - loss: 0.2457 - acc: 0.8800
    Epoch 948/2000
    0s - loss: 0.2529 - acc: 0.8800
    Epoch 949/2000
    0s - loss: 0.2396 - acc: 0.9000
    Epoch 950/2000
    0s - loss: 0.2414 - acc: 0.8600
    Epoch 951/2000
    0s - loss: 0.2389 - acc: 0.8600
    Epoch 952/2000
    0s - loss: 0.2537 - acc: 0.8400
    Epoch 953/2000
    0s - loss: 0.2336 - acc: 0.8600
    Epoch 954/2000
    0s - loss: 0.2403 - acc: 0.8600
    Epoch 955/2000
    0s - loss: 0.2348 - acc: 0.9000
    Epoch 956/2000
    0s - loss: 0.2438 - acc: 0.8400
    Epoch 957/2000
    0s - loss: 0.2474 - acc: 0.8800
    Epoch 958/2000
    0s - loss: 0.2429 - acc: 0.9000
    Epoch 959/2000
    0s - loss: 0.2387 - acc: 0.8600
    Epoch 960/2000
    0s - loss: 0.2405 - acc: 0.8800
    Epoch 961/2000
    0s - loss: 0.2529 - acc: 0.8800
    Epoch 962/2000
    0s - loss: 0.2430 - acc: 0.8600
    Epoch 963/2000
    0s - loss: 0.2360 - acc: 0.8800
    Epoch 964/2000
    0s - loss: 0.2551 - acc: 0.8600
    Epoch 965/2000
    0s - loss: 0.2360 - acc: 0.8800
    Epoch 966/2000
    0s - loss: 0.2335 - acc: 0.8800
    Epoch 967/2000
    0s - loss: 0.2405 - acc: 0.8600
    Epoch 968/2000
    0s - loss: 0.2428 - acc: 0.9000
    Epoch 969/2000
    0s - loss: 0.2357 - acc: 0.8800
    Epoch 970/2000
    0s - loss: 0.2472 - acc: 0.9000
    Epoch 971/2000
    0s - loss: 0.2388 - acc: 0.8800
    Epoch 972/2000
    0s - loss: 0.2456 - acc: 0.8400
    Epoch 973/2000
    0s - loss: 0.2462 - acc: 0.8600
    Epoch 974/2000
    0s - loss: 0.2439 - acc: 0.8600
    Epoch 975/2000
    0s - loss: 0.2505 - acc: 0.8600
    Epoch 976/2000
    0s - loss: 0.2432 - acc: 0.8800
    Epoch 977/2000
    0s - loss: 0.2373 - acc: 0.8800
    Epoch 978/2000
    0s - loss: 0.2422 - acc: 0.9000
    Epoch 979/2000
    0s - loss: 0.2744 - acc: 0.8200
    Epoch 980/2000
    0s - loss: 0.9449 - acc: 0.7400
    Epoch 981/2000
    0s - loss: 0.7776 - acc: 0.7800
    Epoch 982/2000
    0s - loss: 0.3772 - acc: 0.8000
    Epoch 983/2000
    0s - loss: 0.3521 - acc: 0.8400
    Epoch 984/2000
    0s - loss: 0.2347 - acc: 0.9000
    Epoch 985/2000
    0s - loss: 0.2331 - acc: 0.9000
    Epoch 986/2000
    0s - loss: 0.2358 - acc: 0.8600
    Epoch 987/2000
    0s - loss: 0.2335 - acc: 0.8600
    Epoch 988/2000
    0s - loss: 0.2289 - acc: 0.8800
    Epoch 989/2000
    0s - loss: 0.2349 - acc: 0.8800
    Epoch 990/2000
    0s - loss: 0.2300 - acc: 0.8800
    Epoch 991/2000
    0s - loss: 0.2278 - acc: 0.9000
    Epoch 992/2000
    0s - loss: 0.2323 - acc: 0.8400
    Epoch 993/2000
    0s - loss: 0.2310 - acc: 0.8800
    Epoch 994/2000
    0s - loss: 0.2368 - acc: 0.8800
    Epoch 995/2000
    0s - loss: 0.2289 - acc: 0.9000
    Epoch 996/2000
    0s - loss: 0.2344 - acc: 0.9200
    Epoch 997/2000
    0s - loss: 0.2320 - acc: 0.8600
    Epoch 998/2000
    0s - loss: 0.2291 - acc: 0.8600
    Epoch 999/2000
    0s - loss: 0.2339 - acc: 0.9000
    Epoch 1000/2000
    0s - loss: 0.2261 - acc: 0.8800
    Epoch 1001/2000
    0s - loss: 0.2341 - acc: 0.9000
    Epoch 1002/2000
    0s - loss: 0.2306 - acc: 0.8800
    Epoch 1003/2000
    0s - loss: 0.2296 - acc: 0.9200
    Epoch 1004/2000
    0s - loss: 0.2359 - acc: 0.8800
    Epoch 1005/2000
    0s - loss: 0.2333 - acc: 0.8800
    Epoch 1006/2000
    0s - loss: 0.2300 - acc: 0.8600
    Epoch 1007/2000
    0s - loss: 0.2299 - acc: 0.8800
    Epoch 1008/2000
    0s - loss: 0.2317 - acc: 0.8800
    Epoch 1009/2000
    0s - loss: 0.2361 - acc: 0.8800
    Epoch 1010/2000
    0s - loss: 0.2265 - acc: 0.8600
    Epoch 1011/2000
    0s - loss: 0.2309 - acc: 0.8600
    Epoch 1012/2000
    0s - loss: 0.2326 - acc: 0.9000
    Epoch 1013/2000
    0s - loss: 0.2367 - acc: 0.8600
    Epoch 1014/2000
    0s - loss: 0.2273 - acc: 0.9000
    Epoch 1015/2000
    0s - loss: 0.2338 - acc: 0.8800
    Epoch 1016/2000
    0s - loss: 0.2297 - acc: 0.9000
    Epoch 1017/2000
    0s - loss: 0.2294 - acc: 0.8600
    Epoch 1018/2000
    0s - loss: 0.2256 - acc: 0.9000
    Epoch 1019/2000
    0s - loss: 0.2288 - acc: 0.9000
    Epoch 1020/2000
    0s - loss: 0.2267 - acc: 0.8800
    Epoch 1021/2000
    0s - loss: 0.2324 - acc: 0.8600
    Epoch 1022/2000
    0s - loss: 0.2250 - acc: 0.8800
    Epoch 1023/2000
    0s - loss: 0.2216 - acc: 0.8600
    Epoch 1024/2000
    0s - loss: 0.2321 - acc: 0.8600
    Epoch 1025/2000
    0s - loss: 0.2301 - acc: 0.8600
    Epoch 1026/2000
    0s - loss: 0.2329 - acc: 0.8600
    Epoch 1027/2000
    0s - loss: 0.2291 - acc: 0.8600
    Epoch 1028/2000
    0s - loss: 0.2255 - acc: 0.9000
    Epoch 1029/2000
    0s - loss: 0.2289 - acc: 0.9000
    Epoch 1030/2000
    0s - loss: 0.2239 - acc: 0.9000
    Epoch 1031/2000
    0s - loss: 0.2344 - acc: 0.8600
    Epoch 1032/2000
    0s - loss: 0.2280 - acc: 0.9000
    Epoch 1033/2000
    0s - loss: 0.2380 - acc: 0.8600
    Epoch 1034/2000
    0s - loss: 0.2350 - acc: 0.8800
    Epoch 1035/2000
    0s - loss: 0.2362 - acc: 0.8600
    Epoch 1036/2000
    0s - loss: 0.2524 - acc: 0.8600
    Epoch 1037/2000
    0s - loss: 0.2363 - acc: 0.8400
    Epoch 1038/2000
    0s - loss: 0.2284 - acc: 0.8800
    Epoch 1039/2000
    0s - loss: 0.2326 - acc: 0.8800
    Epoch 1040/2000
    0s - loss: 0.2269 - acc: 0.8800
    Epoch 1041/2000
    0s - loss: 0.2272 - acc: 0.8800
    Epoch 1042/2000
    0s - loss: 0.2297 - acc: 0.9000
    Epoch 1043/2000
    0s - loss: 0.2320 - acc: 0.8600
    Epoch 1044/2000
    0s - loss: 0.2303 - acc: 0.9000
    Epoch 1045/2000
    0s - loss: 0.2290 - acc: 0.8800
    Epoch 1046/2000
    0s - loss: 0.2277 - acc: 0.8800
    Epoch 1047/2000
    0s - loss: 0.2237 - acc: 0.8800
    Epoch 1048/2000
    0s - loss: 0.2250 - acc: 0.8800
    Epoch 1049/2000
    0s - loss: 0.2288 - acc: 0.8800
    Epoch 1050/2000
    0s - loss: 0.2252 - acc: 0.8800
    Epoch 1051/2000
    0s - loss: 0.2532 - acc: 0.9000
    Epoch 1052/2000
    0s - loss: 0.2567 - acc: 0.8200
    Epoch 1053/2000
    0s - loss: 0.2394 - acc: 0.8600
    Epoch 1054/2000
    0s - loss: 0.2361 - acc: 0.8600
    Epoch 1055/2000
    0s - loss: 0.2337 - acc: 0.8400
    Epoch 1056/2000
    0s - loss: 0.2328 - acc: 0.8800
    Epoch 1057/2000
    0s - loss: 0.2324 - acc: 0.8800
    Epoch 1058/2000
    0s - loss: 0.2194 - acc: 0.8800
    Epoch 1059/2000
    0s - loss: 0.2258 - acc: 0.8600
    Epoch 1060/2000
    0s - loss: 0.2325 - acc: 0.9000
    Epoch 1061/2000
    0s - loss: 0.2278 - acc: 0.8600
    Epoch 1062/2000
    0s - loss: 0.2445 - acc: 0.8800
    Epoch 1063/2000
    0s - loss: 0.2333 - acc: 0.8800
    Epoch 1064/2000
    0s - loss: 0.2202 - acc: 0.9000
    Epoch 1065/2000
    0s - loss: 0.2318 - acc: 0.8600
    Epoch 1066/2000
    0s - loss: 0.2361 - acc: 0.9000
    Epoch 1067/2000
    0s - loss: 0.2263 - acc: 0.8400
    Epoch 1068/2000
    0s - loss: 0.2190 - acc: 0.8800
    Epoch 1069/2000
    0s - loss: 0.2293 - acc: 0.9000
    Epoch 1070/2000
    0s - loss: 0.2153 - acc: 0.8800
    Epoch 1071/2000
    0s - loss: 0.2257 - acc: 0.9000
    Epoch 1072/2000
    0s - loss: 0.2278 - acc: 0.8800
    Epoch 1073/2000
    0s - loss: 0.2189 - acc: 0.8400
    Epoch 1074/2000
    0s - loss: 0.2333 - acc: 0.8800
    Epoch 1075/2000
    0s - loss: 0.2370 - acc: 0.8600
    Epoch 1076/2000
    0s - loss: 0.2565 - acc: 0.8400
    Epoch 1077/2000
    0s - loss: 0.4734 - acc: 0.7800
    Epoch 1078/2000
    0s - loss: 0.5477 - acc: 0.7800
    Epoch 1079/2000
    0s - loss: 0.3577 - acc: 0.8400
    Epoch 1080/2000
    0s - loss: 0.2459 - acc: 0.9000
    Epoch 1081/2000
    0s - loss: 0.2198 - acc: 0.9000
    Epoch 1082/2000
    0s - loss: 0.2254 - acc: 0.8800
    Epoch 1083/2000
    0s - loss: 0.2158 - acc: 0.8800
    Epoch 1084/2000
    0s - loss: 0.2132 - acc: 0.9000
    Epoch 1085/2000
    0s - loss: 0.2122 - acc: 0.9000
    Epoch 1086/2000
    0s - loss: 0.2191 - acc: 0.8600
    Epoch 1087/2000
    0s - loss: 0.2159 - acc: 0.8800
    Epoch 1088/2000
    0s - loss: 0.2204 - acc: 0.8600
    Epoch 1089/2000
    0s - loss: 0.2158 - acc: 0.9000
    Epoch 1090/2000
    0s - loss: 0.2166 - acc: 0.9000
    Epoch 1091/2000
    0s - loss: 0.2176 - acc: 0.8800
    Epoch 1092/2000
    0s - loss: 0.2148 - acc: 0.9000
    Epoch 1093/2000
    0s - loss: 0.2220 - acc: 0.8600
    Epoch 1094/2000
    0s - loss: 0.2141 - acc: 0.8600
    Epoch 1095/2000
    0s - loss: 0.2050 - acc: 0.9200
    Epoch 1096/2000
    0s - loss: 0.2174 - acc: 0.8600
    Epoch 1097/2000
    0s - loss: 0.2216 - acc: 0.8800
    Epoch 1098/2000
    0s - loss: 0.2111 - acc: 0.9000
    Epoch 1099/2000
    0s - loss: 0.2191 - acc: 0.8800
    Epoch 1100/2000
    0s - loss: 0.2197 - acc: 0.8800
    Epoch 1101/2000
    0s - loss: 0.2202 - acc: 0.8800
    Epoch 1102/2000
    0s - loss: 0.2240 - acc: 0.9000
    Epoch 1103/2000
    0s - loss: 0.2150 - acc: 0.8600
    Epoch 1104/2000
    0s - loss: 0.2205 - acc: 0.8800
    Epoch 1105/2000
    0s - loss: 0.2178 - acc: 0.9000
    Epoch 1106/2000
    0s - loss: 0.2149 - acc: 0.8800
    Epoch 1107/2000
    0s - loss: 0.2137 - acc: 0.8800
    Epoch 1108/2000
    0s - loss: 0.2156 - acc: 0.9000
    Epoch 1109/2000
    0s - loss: 0.2216 - acc: 0.9000
    Epoch 1110/2000
    0s - loss: 0.2357 - acc: 0.8600
    Epoch 1111/2000
    0s - loss: 0.2118 - acc: 0.9000
    Epoch 1112/2000
    0s - loss: 0.2189 - acc: 0.8600
    Epoch 1113/2000
    0s - loss: 0.2171 - acc: 0.8800
    Epoch 1114/2000
    0s - loss: 0.2168 - acc: 0.8600
    Epoch 1115/2000
    0s - loss: 0.2198 - acc: 0.9000
    Epoch 1116/2000
    0s - loss: 0.2104 - acc: 0.9200
    Epoch 1117/2000
    0s - loss: 0.2105 - acc: 0.9000
    Epoch 1118/2000
    0s - loss: 0.2170 - acc: 0.8800
    Epoch 1119/2000
    0s - loss: 0.2084 - acc: 0.8800
    Epoch 1120/2000
    0s - loss: 0.2125 - acc: 0.8600
    Epoch 1121/2000
    0s - loss: 0.2214 - acc: 0.8600
    Epoch 1122/2000
    0s - loss: 0.2201 - acc: 0.9000
    Epoch 1123/2000
    0s - loss: 0.2177 - acc: 0.8800
    Epoch 1124/2000
    0s - loss: 0.2137 - acc: 0.9000
    Epoch 1125/2000
    0s - loss: 0.2139 - acc: 0.9000
    Epoch 1126/2000
    0s - loss: 0.2358 - acc: 0.8600
    Epoch 1127/2000
    0s - loss: 0.2139 - acc: 0.9000
    Epoch 1128/2000
    0s - loss: 0.2134 - acc: 0.9000
    Epoch 1129/2000
    0s - loss: 0.2169 - acc: 0.8600
    Epoch 1130/2000
    0s - loss: 0.2166 - acc: 0.8800
    Epoch 1131/2000
    0s - loss: 0.2201 - acc: 0.9000
    Epoch 1132/2000
    0s - loss: 0.2183 - acc: 0.8800
    Epoch 1133/2000
    0s - loss: 0.2142 - acc: 0.9000
    Epoch 1134/2000
    0s - loss: 0.2240 - acc: 0.8800
    Epoch 1135/2000
    0s - loss: 0.2131 - acc: 0.8800
    Epoch 1136/2000
    0s - loss: 0.2160 - acc: 0.8800
    Epoch 1137/2000
    0s - loss: 0.2166 - acc: 0.9000
    Epoch 1138/2000
    0s - loss: 0.2053 - acc: 0.9200
    Epoch 1139/2000
    0s - loss: 0.2198 - acc: 0.8600
    Epoch 1140/2000
    0s - loss: 0.2239 - acc: 0.8800
    Epoch 1141/2000
    0s - loss: 0.2084 - acc: 0.8600
    Epoch 1142/2000
    0s - loss: 0.2136 - acc: 0.9000
    Epoch 1143/2000
    0s - loss: 0.2121 - acc: 0.8800
    Epoch 1144/2000
    0s - loss: 0.2032 - acc: 0.8800
    Epoch 1145/2000
    0s - loss: 0.2160 - acc: 0.9000
    Epoch 1146/2000
    0s - loss: 0.2121 - acc: 0.8800
    Epoch 1147/2000
    0s - loss: 0.2090 - acc: 0.8800
    Epoch 1148/2000
    0s - loss: 0.2131 - acc: 0.8800
    Epoch 1149/2000
    0s - loss: 0.2152 - acc: 0.9000
    Epoch 1150/2000
    0s - loss: 0.2271 - acc: 0.8600
    Epoch 1151/2000
    0s - loss: 0.2106 - acc: 0.9000
    Epoch 1152/2000
    0s - loss: 0.2294 - acc: 0.8800
    Epoch 1153/2000
    0s - loss: 0.2180 - acc: 0.8800
    Epoch 1154/2000
    0s - loss: 0.2353 - acc: 0.8800
    Epoch 1155/2000
    0s - loss: 0.3202 - acc: 0.8800
    Epoch 1156/2000
    0s - loss: 0.7671 - acc: 0.7600
    Epoch 1157/2000
    0s - loss: 0.5885 - acc: 0.7400
    Epoch 1158/2000
    0s - loss: 0.2472 - acc: 0.8800
    Epoch 1159/2000
    0s - loss: 0.2248 - acc: 0.9000
    Epoch 1160/2000
    0s - loss: 0.2162 - acc: 0.8800
    Epoch 1161/2000
    0s - loss: 0.2086 - acc: 0.9200
    Epoch 1162/2000
    0s - loss: 0.2071 - acc: 0.8800
    Epoch 1163/2000
    0s - loss: 0.2027 - acc: 0.9200
    Epoch 1164/2000
    0s - loss: 0.2082 - acc: 0.9000
    Epoch 1165/2000
    0s - loss: 0.2231 - acc: 0.9000
    Epoch 1166/2000
    0s - loss: 0.2006 - acc: 0.9200
    Epoch 1167/2000
    0s - loss: 0.2090 - acc: 0.8800
    Epoch 1168/2000
    0s - loss: 0.2104 - acc: 0.9000
    Epoch 1169/2000
    0s - loss: 0.2066 - acc: 0.8800
    Epoch 1170/2000
    0s - loss: 0.2009 - acc: 0.9200
    Epoch 1171/2000
    0s - loss: 0.2078 - acc: 0.8600
    Epoch 1172/2000
    0s - loss: 0.2169 - acc: 0.8800
    Epoch 1173/2000
    0s - loss: 0.2107 - acc: 0.9000
    Epoch 1174/2000
    0s - loss: 0.2045 - acc: 0.9000
    Epoch 1175/2000
    0s - loss: 0.2007 - acc: 0.9000
    Epoch 1176/2000
    0s - loss: 0.2089 - acc: 0.8800
    Epoch 1177/2000
    0s - loss: 0.2066 - acc: 0.9000
    Epoch 1178/2000
    0s - loss: 0.2156 - acc: 0.8800
    Epoch 1179/2000
    0s - loss: 0.2025 - acc: 0.9000
    Epoch 1180/2000
    0s - loss: 0.2072 - acc: 0.8800
    Epoch 1181/2000
    0s - loss: 0.1994 - acc: 0.8800
    Epoch 1182/2000
    0s - loss: 0.2064 - acc: 0.9000
    Epoch 1183/2000
    0s - loss: 0.2058 - acc: 0.8800
    Epoch 1184/2000
    0s - loss: 0.2095 - acc: 0.9000
    Epoch 1185/2000
    0s - loss: 0.2042 - acc: 0.9000
    Epoch 1186/2000
    0s - loss: 0.2051 - acc: 0.9000
    Epoch 1187/2000
    0s - loss: 0.2185 - acc: 0.9000
    Epoch 1188/2000
    0s - loss: 0.2056 - acc: 0.8800
    Epoch 1189/2000
    0s - loss: 0.2086 - acc: 0.8800
    Epoch 1190/2000
    0s - loss: 0.2028 - acc: 0.9000
    Epoch 1191/2000
    0s - loss: 0.2066 - acc: 0.8800
    Epoch 1192/2000
    0s - loss: 0.2043 - acc: 0.9000
    Epoch 1193/2000
    0s - loss: 0.2013 - acc: 0.9200
    Epoch 1194/2000
    0s - loss: 0.2096 - acc: 0.8600
    Epoch 1195/2000
    0s - loss: 0.2128 - acc: 0.8800
    Epoch 1196/2000
    0s - loss: 0.2021 - acc: 0.8800
    Epoch 1197/2000
    0s - loss: 0.2084 - acc: 0.8600
    Epoch 1198/2000
    0s - loss: 0.2108 - acc: 0.8800
    Epoch 1199/2000
    0s - loss: 0.2027 - acc: 0.8600
    Epoch 1200/2000
    0s - loss: 0.2063 - acc: 0.8800
    Epoch 1201/2000
    0s - loss: 0.2058 - acc: 0.9000
    Epoch 1202/2000
    0s - loss: 0.2050 - acc: 0.8800
    Epoch 1203/2000
    0s - loss: 0.1985 - acc: 0.9000
    Epoch 1204/2000
    0s - loss: 0.2042 - acc: 0.9000
    Epoch 1205/2000
    0s - loss: 0.2080 - acc: 0.9000
    Epoch 1206/2000
    0s - loss: 0.2093 - acc: 0.8800
    Epoch 1207/2000
    0s - loss: 0.2152 - acc: 0.8800
    Epoch 1208/2000
    0s - loss: 0.2077 - acc: 0.8800
    Epoch 1209/2000
    0s - loss: 0.2038 - acc: 0.9000
    Epoch 1210/2000
    0s - loss: 0.2048 - acc: 0.8800
    Epoch 1211/2000
    0s - loss: 0.2019 - acc: 0.8600
    Epoch 1212/2000
    0s - loss: 0.1978 - acc: 0.8600
    Epoch 1213/2000
    0s - loss: 0.1965 - acc: 0.9000
    Epoch 1214/2000
    0s - loss: 0.2023 - acc: 0.8800
    Epoch 1215/2000
    0s - loss: 0.2049 - acc: 0.8800
    Epoch 1216/2000
    0s - loss: 0.2034 - acc: 0.9200
    Epoch 1217/2000
    0s - loss: 0.2073 - acc: 0.9000
    Epoch 1218/2000
    0s - loss: 0.2016 - acc: 0.8800
    Epoch 1219/2000
    0s - loss: 0.2051 - acc: 0.8600
    Epoch 1220/2000
    0s - loss: 0.2085 - acc: 0.9000
    Epoch 1221/2000
    0s - loss: 0.2038 - acc: 0.9200
    Epoch 1222/2000
    0s - loss: 0.2029 - acc: 0.8600
    Epoch 1223/2000
    0s - loss: 0.1993 - acc: 0.8800
    Epoch 1224/2000
    0s - loss: 0.2016 - acc: 0.9000
    Epoch 1225/2000
    0s - loss: 0.2055 - acc: 0.8800
    Epoch 1226/2000
    0s - loss: 0.2043 - acc: 0.8800
    Epoch 1227/2000
    0s - loss: 0.2186 - acc: 0.8400
    Epoch 1228/2000
    0s - loss: 0.2067 - acc: 0.8800
    Epoch 1229/2000
    0s - loss: 0.2814 - acc: 0.8600
    Epoch 1230/2000
    0s - loss: 0.7262 - acc: 0.7800
    Epoch 1231/2000
    0s - loss: 0.6907 - acc: 0.7600
    Epoch 1232/2000
    0s - loss: 0.5163 - acc: 0.7800
    Epoch 1233/2000
    0s - loss: 0.2342 - acc: 0.9000
    Epoch 1234/2000
    0s - loss: 0.2573 - acc: 0.8800
    Epoch 1235/2000
    0s - loss: 0.2980 - acc: 0.8600
    Epoch 1236/2000
    0s - loss: 0.2084 - acc: 0.9000
    Epoch 1237/2000
    0s - loss: 0.2000 - acc: 0.8800
    Epoch 1238/2000
    0s - loss: 0.2039 - acc: 0.9000
    Epoch 1239/2000
    0s - loss: 0.1967 - acc: 0.9200
    Epoch 1240/2000
    0s - loss: 0.1888 - acc: 0.9200
    Epoch 1241/2000
    0s - loss: 0.1967 - acc: 0.8800
    Epoch 1242/2000
    0s - loss: 0.1930 - acc: 0.8800
    Epoch 1243/2000
    0s - loss: 0.1997 - acc: 0.9000
    Epoch 1244/2000
    0s - loss: 0.1931 - acc: 0.9200
    Epoch 1245/2000
    0s - loss: 0.2002 - acc: 0.8600
    Epoch 1246/2000
    0s - loss: 0.2011 - acc: 0.9200
    Epoch 1247/2000
    0s - loss: 0.2049 - acc: 0.8800
    Epoch 1248/2000
    0s - loss: 0.1932 - acc: 0.9000
    Epoch 1249/2000
    0s - loss: 0.1983 - acc: 0.8800
    Epoch 1250/2000
    0s - loss: 0.2079 - acc: 0.9200
    Epoch 1251/2000
    0s - loss: 0.1914 - acc: 0.9200
    Epoch 1252/2000
    0s - loss: 0.2035 - acc: 0.8600
    Epoch 1253/2000
    0s - loss: 0.2043 - acc: 0.8400
    Epoch 1254/2000
    0s - loss: 0.2038 - acc: 0.8600
    Epoch 1255/2000
    0s - loss: 0.2015 - acc: 0.8800
    Epoch 1256/2000
    0s - loss: 0.1954 - acc: 0.9000
    Epoch 1257/2000
    0s - loss: 0.1957 - acc: 0.8600
    Epoch 1258/2000
    0s - loss: 0.1980 - acc: 0.9000
    Epoch 1259/2000
    0s - loss: 0.1856 - acc: 0.9200
    Epoch 1260/2000
    0s - loss: 0.1957 - acc: 0.8800
    Epoch 1261/2000
    0s - loss: 0.1961 - acc: 0.8800
    Epoch 1262/2000
    0s - loss: 0.1938 - acc: 0.9200
    Epoch 1263/2000
    0s - loss: 0.1996 - acc: 0.9000
    Epoch 1264/2000
    0s - loss: 0.2016 - acc: 0.8600
    Epoch 1265/2000
    0s - loss: 0.1942 - acc: 0.8600
    Epoch 1266/2000
    0s - loss: 0.1931 - acc: 0.8800
    Epoch 1267/2000
    0s - loss: 0.1873 - acc: 0.8800
    Epoch 1268/2000
    0s - loss: 0.1941 - acc: 0.9000
    Epoch 1269/2000
    0s - loss: 0.1990 - acc: 0.9200
    Epoch 1270/2000
    0s - loss: 0.1934 - acc: 0.9000
    Epoch 1271/2000
    0s - loss: 0.1974 - acc: 0.8800
    Epoch 1272/2000
    0s - loss: 0.1959 - acc: 0.8800
    Epoch 1273/2000
    0s - loss: 0.2005 - acc: 0.8800
    Epoch 1274/2000
    0s - loss: 0.1929 - acc: 0.8800
    Epoch 1275/2000
    0s - loss: 0.2012 - acc: 0.8800
    Epoch 1276/2000
    0s - loss: 0.2001 - acc: 0.8800
    Epoch 1277/2000
    0s - loss: 0.2045 - acc: 0.8800
    Epoch 1278/2000
    0s - loss: 0.2002 - acc: 0.8800
    Epoch 1279/2000
    0s - loss: 0.1918 - acc: 0.9000
    Epoch 1280/2000
    0s - loss: 0.1935 - acc: 0.8800
    Epoch 1281/2000
    0s - loss: 0.1947 - acc: 0.8600
    Epoch 1282/2000
    0s - loss: 0.1961 - acc: 0.9000
    Epoch 1283/2000
    0s - loss: 0.1961 - acc: 0.9000
    Epoch 1284/2000
    0s - loss: 0.1942 - acc: 0.8800
    Epoch 1285/2000
    0s - loss: 0.2012 - acc: 0.8600
    Epoch 1286/2000
    0s - loss: 0.1978 - acc: 0.9200
    Epoch 1287/2000
    0s - loss: 0.1990 - acc: 0.9000
    Epoch 1288/2000
    0s - loss: 0.1958 - acc: 0.8800
    Epoch 1289/2000
    0s - loss: 0.1943 - acc: 0.8800
    Epoch 1290/2000
    0s - loss: 0.1929 - acc: 0.9000
    Epoch 1291/2000
    0s - loss: 0.1974 - acc: 0.8800
    Epoch 1292/2000
    0s - loss: 0.2034 - acc: 0.9000
    Epoch 1293/2000
    0s - loss: 0.1952 - acc: 0.9000
    Epoch 1294/2000
    0s - loss: 0.1950 - acc: 0.8800
    Epoch 1295/2000
    0s - loss: 0.1951 - acc: 0.9200
    Epoch 1296/2000
    0s - loss: 0.1922 - acc: 0.9000
    Epoch 1297/2000
    0s - loss: 0.1885 - acc: 0.9200
    Epoch 1298/2000
    0s - loss: 0.2016 - acc: 0.9000
    Epoch 1299/2000
    0s - loss: 0.2008 - acc: 0.8800
    Epoch 1300/2000
    0s - loss: 0.1891 - acc: 0.9200
    Epoch 1301/2000
    0s - loss: 0.1961 - acc: 0.9000
    Epoch 1302/2000
    0s - loss: 0.1910 - acc: 0.9000
    Epoch 1303/2000
    0s - loss: 0.1981 - acc: 0.9200
    Epoch 1304/2000
    0s - loss: 0.1962 - acc: 0.8800
    Epoch 1305/2000
    0s - loss: 0.1933 - acc: 0.9200
    Epoch 1306/2000
    0s - loss: 0.2000 - acc: 0.8600
    Epoch 1307/2000
    0s - loss: 0.1903 - acc: 0.8800
    Epoch 1308/2000
    0s - loss: 0.1928 - acc: 0.9000
    Epoch 1309/2000
    0s - loss: 0.1873 - acc: 0.8800
    Epoch 1310/2000
    0s - loss: 0.1909 - acc: 0.8800
    Epoch 1311/2000
    0s - loss: 0.1920 - acc: 0.8600
    Epoch 1312/2000
    0s - loss: 0.1959 - acc: 0.8800
    Epoch 1313/2000
    0s - loss: 0.1995 - acc: 0.9200
    Epoch 1314/2000
    0s - loss: 0.1859 - acc: 0.9000
    Epoch 1315/2000
    0s - loss: 0.1913 - acc: 0.9000
    Epoch 1316/2000
    0s - loss: 0.1949 - acc: 0.9200
    Epoch 1317/2000
    0s - loss: 0.1917 - acc: 0.8800
    Epoch 1318/2000
    0s - loss: 0.2008 - acc: 0.8400
    Epoch 1319/2000
    0s - loss: 0.1850 - acc: 0.9000
    Epoch 1320/2000
    0s - loss: 0.1911 - acc: 0.8600
    Epoch 1321/2000
    0s - loss: 0.2699 - acc: 0.8800
    Epoch 1322/2000
    0s - loss: 0.5231 - acc: 0.7800
    Epoch 1323/2000
    0s - loss: 0.2412 - acc: 0.8800
    Epoch 1324/2000
    0s - loss: 0.4358 - acc: 0.8400
    Epoch 1325/2000
    0s - loss: 0.1973 - acc: 0.8800
    Epoch 1326/2000
    0s - loss: 0.1919 - acc: 0.9000
    Epoch 1327/2000
    0s - loss: 0.1877 - acc: 0.9200
    Epoch 1328/2000
    0s - loss: 0.1813 - acc: 0.9200
    Epoch 1329/2000
    0s - loss: 0.1852 - acc: 0.9000
    Epoch 1330/2000
    0s - loss: 0.1936 - acc: 0.8800
    Epoch 1331/2000
    0s - loss: 0.1822 - acc: 0.8800
    Epoch 1332/2000
    0s - loss: 0.1837 - acc: 0.8800
    Epoch 1333/2000
    0s - loss: 0.1888 - acc: 0.8800
    Epoch 1334/2000
    0s - loss: 0.1895 - acc: 0.8600
    Epoch 1335/2000
    0s - loss: 0.1956 - acc: 0.8600
    Epoch 1336/2000
    0s - loss: 0.1817 - acc: 0.9200
    Epoch 1337/2000
    0s - loss: 0.1887 - acc: 0.8600
    Epoch 1338/2000
    0s - loss: 0.1921 - acc: 0.9000
    Epoch 1339/2000
    0s - loss: 0.1848 - acc: 0.9000
    Epoch 1340/2000
    0s - loss: 0.1887 - acc: 0.9200
    Epoch 1341/2000
    0s - loss: 0.1866 - acc: 0.8800
    Epoch 1342/2000
    0s - loss: 0.1884 - acc: 0.9000
    Epoch 1343/2000
    0s - loss: 0.1900 - acc: 0.8800
    Epoch 1344/2000
    0s - loss: 0.1818 - acc: 0.9000
    Epoch 1345/2000
    0s - loss: 0.1903 - acc: 0.9200
    Epoch 1346/2000
    0s - loss: 0.1852 - acc: 0.9000
    Epoch 1347/2000
    0s - loss: 0.1847 - acc: 0.9000
    Epoch 1348/2000
    0s - loss: 0.1869 - acc: 0.9000
    Epoch 1349/2000
    0s - loss: 0.1862 - acc: 0.9000
    Epoch 1350/2000
    0s - loss: 0.1997 - acc: 0.8600
    Epoch 1351/2000
    0s - loss: 0.1912 - acc: 0.8800
    Epoch 1352/2000
    0s - loss: 0.1857 - acc: 0.8600
    Epoch 1353/2000
    0s - loss: 0.1851 - acc: 0.9000
    Epoch 1354/2000
    0s - loss: 0.1879 - acc: 0.9000
    Epoch 1355/2000
    0s - loss: 0.1888 - acc: 0.9000
    Epoch 1356/2000
    0s - loss: 0.1958 - acc: 0.9000
    Epoch 1357/2000
    0s - loss: 0.1871 - acc: 0.8800
    Epoch 1358/2000
    0s - loss: 0.1885 - acc: 0.9200
    Epoch 1359/2000
    0s - loss: 0.1832 - acc: 0.9000
    Epoch 1360/2000
    0s - loss: 0.1799 - acc: 0.8800
    Epoch 1361/2000
    0s - loss: 0.1839 - acc: 0.9000
    Epoch 1362/2000
    0s - loss: 0.1961 - acc: 0.8800
    Epoch 1363/2000
    0s - loss: 0.1971 - acc: 0.8600
    Epoch 1364/2000
    0s - loss: 0.1797 - acc: 0.9000
    Epoch 1365/2000
    0s - loss: 0.1936 - acc: 0.8800
    Epoch 1366/2000
    0s - loss: 0.1861 - acc: 0.9200
    Epoch 1367/2000
    0s - loss: 0.1932 - acc: 0.8800
    Epoch 1368/2000
    0s - loss: 0.1832 - acc: 0.9000
    Epoch 1369/2000
    0s - loss: 0.1846 - acc: 0.9200
    Epoch 1370/2000
    0s - loss: 0.1905 - acc: 0.9000
    Epoch 1371/2000
    0s - loss: 0.1861 - acc: 0.9000
    Epoch 1372/2000
    0s - loss: 0.1852 - acc: 0.9200
    Epoch 1373/2000
    0s - loss: 0.1816 - acc: 0.9000
    Epoch 1374/2000
    0s - loss: 0.1945 - acc: 0.8600
    Epoch 1375/2000
    0s - loss: 0.1939 - acc: 0.9000
    Epoch 1376/2000
    0s - loss: 0.1806 - acc: 0.9200
    Epoch 1377/2000
    0s - loss: 0.1916 - acc: 0.8800
    Epoch 1378/2000
    0s - loss: 0.1897 - acc: 0.9200
    Epoch 1379/2000
    0s - loss: 0.2021 - acc: 0.8800
    Epoch 1380/2000
    0s - loss: 0.1956 - acc: 0.9000
    Epoch 1381/2000
    0s - loss: 0.1955 - acc: 0.8600
    Epoch 1382/2000
    0s - loss: 0.1985 - acc: 0.9000
    Epoch 1383/2000
    0s - loss: 0.1902 - acc: 0.9000
    Epoch 1384/2000
    0s - loss: 0.1834 - acc: 0.8800
    Epoch 1385/2000
    0s - loss: 0.1825 - acc: 0.9000
    Epoch 1386/2000
    0s - loss: 0.1899 - acc: 0.8800
    Epoch 1387/2000
    0s - loss: 0.1830 - acc: 0.9000
    Epoch 1388/2000
    0s - loss: 0.1785 - acc: 0.9000
    Epoch 1389/2000
    0s - loss: 0.1858 - acc: 0.8800
    Epoch 1390/2000
    0s - loss: 0.1993 - acc: 0.8800
    Epoch 1391/2000
    0s - loss: 0.1790 - acc: 0.9200
    Epoch 1392/2000
    0s - loss: 0.1879 - acc: 0.9000
    Epoch 1393/2000
    0s - loss: 0.1814 - acc: 0.9200
    Epoch 1394/2000
    0s - loss: 0.1868 - acc: 0.8800
    Epoch 1395/2000
    0s - loss: 0.1970 - acc: 0.8800
    Epoch 1396/2000
    0s - loss: 0.2020 - acc: 0.9000
    Epoch 1397/2000
    0s - loss: 0.1929 - acc: 0.9200
    Epoch 1398/2000
    0s - loss: 0.1846 - acc: 0.9000
    Epoch 1399/2000
    0s - loss: 0.1863 - acc: 0.8600
    Epoch 1400/2000
    0s - loss: 0.2004 - acc: 0.8800
    Epoch 1401/2000
    0s - loss: 0.1859 - acc: 0.9000
    Epoch 1402/2000
    0s - loss: 0.1820 - acc: 0.9000
    Epoch 1403/2000
    0s - loss: 0.1891 - acc: 0.8800
    Epoch 1404/2000
    0s - loss: 0.1870 - acc: 0.8800
    Epoch 1405/2000
    0s - loss: 0.1801 - acc: 0.9200
    Epoch 1406/2000
    0s - loss: 0.1930 - acc: 0.8800
    Epoch 1407/2000
    0s - loss: 0.1839 - acc: 0.9000
    Epoch 1408/2000
    0s - loss: 0.1836 - acc: 0.9000
    Epoch 1409/2000
    0s - loss: 0.1845 - acc: 0.9200
    Epoch 1410/2000
    0s - loss: 0.1941 - acc: 0.8600
    Epoch 1411/2000
    0s - loss: 0.1792 - acc: 0.8800
    Epoch 1412/2000
    0s - loss: 0.1755 - acc: 0.9200
    Epoch 1413/2000
    0s - loss: 0.1806 - acc: 0.9200
    Epoch 1414/2000
    0s - loss: 0.1775 - acc: 0.8800
    Epoch 1415/2000
    0s - loss: 0.1826 - acc: 0.9000
    Epoch 1416/2000
    0s - loss: 0.1820 - acc: 0.8800
    Epoch 1417/2000
    0s - loss: 0.1766 - acc: 0.8800
    Epoch 1418/2000
    0s - loss: 0.1921 - acc: 0.8600
    Epoch 1419/2000
    0s - loss: 0.1846 - acc: 0.9000
    Epoch 1420/2000
    0s - loss: 0.1899 - acc: 0.8800
    Epoch 1421/2000
    0s - loss: 0.1830 - acc: 0.9000
    Epoch 1422/2000
    0s - loss: 0.1821 - acc: 0.9200
    Epoch 1423/2000
    0s - loss: 0.1977 - acc: 0.8800
    Epoch 1424/2000
    0s - loss: 0.1844 - acc: 0.8400
    Epoch 1425/2000
    0s - loss: 0.1844 - acc: 0.8800
    Epoch 1426/2000
    0s - loss: 0.1971 - acc: 0.8600
    Epoch 1427/2000
    0s - loss: 0.1863 - acc: 0.8800
    Epoch 1428/2000
    0s - loss: 0.1862 - acc: 0.9000
    Epoch 1429/2000
    0s - loss: 0.1810 - acc: 0.9000
    Epoch 1430/2000
    0s - loss: 0.1779 - acc: 0.9000
    Epoch 1431/2000
    0s - loss: 0.1798 - acc: 0.9000
    Epoch 1432/2000
    0s - loss: 0.1743 - acc: 0.9200
    Epoch 1433/2000
    0s - loss: 0.1814 - acc: 0.9000
    Epoch 1434/2000
    0s - loss: 0.1859 - acc: 0.8800
    Epoch 1435/2000
    0s - loss: 0.1724 - acc: 0.8800
    Epoch 1436/2000
    0s - loss: 0.1933 - acc: 0.9000
    Epoch 1437/2000
    0s - loss: 0.1925 - acc: 0.8800
    Epoch 1438/2000
    0s - loss: 0.1836 - acc: 0.8800
    Epoch 1439/2000
    0s - loss: 0.2423 - acc: 0.8600
    Epoch 1440/2000
    0s - loss: 0.5898 - acc: 0.7800
    Epoch 1441/2000
    0s - loss: 0.3214 - acc: 0.8400
    Epoch 1442/2000
    0s - loss: 0.1984 - acc: 0.9000
    Epoch 1443/2000
    0s - loss: 0.1813 - acc: 0.9200
    Epoch 1444/2000
    0s - loss: 0.1792 - acc: 0.9000
    Epoch 1445/2000
    0s - loss: 0.1792 - acc: 0.9000
    Epoch 1446/2000
    0s - loss: 0.1732 - acc: 0.9000
    Epoch 1447/2000
    0s - loss: 0.1728 - acc: 0.8800
    Epoch 1448/2000
    0s - loss: 0.1733 - acc: 0.9000
    Epoch 1449/2000
    0s - loss: 0.1705 - acc: 0.9200
    Epoch 1450/2000
    0s - loss: 0.1754 - acc: 0.8800
    Epoch 1451/2000
    0s - loss: 0.1825 - acc: 0.8800
    Epoch 1452/2000
    0s - loss: 0.1791 - acc: 0.8600
    Epoch 1453/2000
    0s - loss: 0.1693 - acc: 0.9000
    Epoch 1454/2000
    0s - loss: 0.1700 - acc: 0.9200
    Epoch 1455/2000
    0s - loss: 0.1717 - acc: 0.9000
    Epoch 1456/2000
    0s - loss: 0.1784 - acc: 0.8800
    Epoch 1457/2000
    0s - loss: 0.1767 - acc: 0.9000
    Epoch 1458/2000
    0s - loss: 0.1708 - acc: 0.9200
    Epoch 1459/2000
    0s - loss: 0.1715 - acc: 0.9200
    Epoch 1460/2000
    0s - loss: 0.1774 - acc: 0.8800
    Epoch 1461/2000
    0s - loss: 0.1735 - acc: 0.9000
    Epoch 1462/2000
    0s - loss: 0.1696 - acc: 0.9000
    Epoch 1463/2000
    0s - loss: 0.1758 - acc: 0.8800
    Epoch 1464/2000
    0s - loss: 0.1755 - acc: 0.8600
    Epoch 1465/2000
    0s - loss: 0.1705 - acc: 0.9200
    Epoch 1466/2000
    0s - loss: 0.1758 - acc: 0.8800
    Epoch 1467/2000
    0s - loss: 0.1773 - acc: 0.9000
    Epoch 1468/2000
    0s - loss: 0.1747 - acc: 0.9200
    Epoch 1469/2000
    0s - loss: 0.1875 - acc: 0.8600
    Epoch 1470/2000
    0s - loss: 0.1738 - acc: 0.9000
    Epoch 1471/2000
    0s - loss: 0.1790 - acc: 0.9200
    Epoch 1472/2000
    0s - loss: 0.1747 - acc: 0.9200
    Epoch 1473/2000
    0s - loss: 0.1693 - acc: 0.9000
    Epoch 1474/2000
    0s - loss: 0.1695 - acc: 0.9000
    Epoch 1475/2000
    0s - loss: 0.1692 - acc: 0.8800
    Epoch 1476/2000
    0s - loss: 0.1729 - acc: 0.8800
    Epoch 1477/2000
    0s - loss: 0.1657 - acc: 0.9200
    Epoch 1478/2000
    0s - loss: 0.1841 - acc: 0.8800
    Epoch 1479/2000
    0s - loss: 0.1902 - acc: 0.8600
    Epoch 1480/2000
    0s - loss: 0.1710 - acc: 0.9000
    Epoch 1481/2000
    0s - loss: 0.1690 - acc: 0.9000
    Epoch 1482/2000
    0s - loss: 0.1756 - acc: 0.9200
    Epoch 1483/2000
    0s - loss: 0.1740 - acc: 0.8800
    Epoch 1484/2000
    0s - loss: 0.1703 - acc: 0.9000
    Epoch 1485/2000
    0s - loss: 0.1735 - acc: 0.8800
    Epoch 1486/2000
    0s - loss: 0.1717 - acc: 0.9200
    Epoch 1487/2000
    0s - loss: 0.1779 - acc: 0.9200
    Epoch 1488/2000
    0s - loss: 0.1658 - acc: 0.9200
    Epoch 1489/2000
    0s - loss: 0.1768 - acc: 0.8800
    Epoch 1490/2000
    0s - loss: 0.1707 - acc: 0.9000
    Epoch 1491/2000
    0s - loss: 0.1640 - acc: 0.8800
    Epoch 1492/2000
    0s - loss: 0.1753 - acc: 0.8800
    Epoch 1493/2000
    0s - loss: 0.1691 - acc: 0.9200
    Epoch 1494/2000
    0s - loss: 0.1746 - acc: 0.8600
    Epoch 1495/2000
    0s - loss: 0.1698 - acc: 0.9000
    Epoch 1496/2000
    0s - loss: 0.1704 - acc: 0.9000
    Epoch 1497/2000
    0s - loss: 0.1736 - acc: 0.9000
    Epoch 1498/2000
    0s - loss: 0.1781 - acc: 0.9000
    Epoch 1499/2000
    0s - loss: 0.1771 - acc: 0.8800
    Epoch 1500/2000
    0s - loss: 0.1711 - acc: 0.8600
    Epoch 1501/2000
    0s - loss: 0.1702 - acc: 0.9000
    Epoch 1502/2000
    0s - loss: 0.1734 - acc: 0.9000
    Epoch 1503/2000
    0s - loss: 0.1726 - acc: 0.8800
    Epoch 1504/2000
    0s - loss: 0.1718 - acc: 0.9000
    Epoch 1505/2000
    0s - loss: 0.1765 - acc: 0.8600
    Epoch 1506/2000
    0s - loss: 0.1772 - acc: 0.9000
    Epoch 1507/2000
    0s - loss: 0.1841 - acc: 0.8600
    Epoch 1508/2000
    0s - loss: 0.1745 - acc: 0.9000
    Epoch 1509/2000
    0s - loss: 0.1832 - acc: 0.9000
    Epoch 1510/2000
    0s - loss: 0.2086 - acc: 0.9000
    Epoch 1511/2000
    0s - loss: 0.3093 - acc: 0.8400
    Epoch 1512/2000
    0s - loss: 0.1869 - acc: 0.8800
    Epoch 1513/2000
    0s - loss: 0.1758 - acc: 0.8800
    Epoch 1514/2000
    0s - loss: 0.1715 - acc: 0.9000
    Epoch 1515/2000
    0s - loss: 0.1698 - acc: 0.9200
    Epoch 1516/2000
    0s - loss: 0.1758 - acc: 0.9000
    Epoch 1517/2000
    0s - loss: 0.1727 - acc: 0.9000
    Epoch 1518/2000
    0s - loss: 0.1691 - acc: 0.8800
    Epoch 1519/2000
    0s - loss: 0.1696 - acc: 0.9000
    Epoch 1520/2000
    0s - loss: 0.1682 - acc: 0.9000
    Epoch 1521/2000
    0s - loss: 0.1719 - acc: 0.8600
    Epoch 1522/2000
    0s - loss: 0.1707 - acc: 0.9000
    Epoch 1523/2000
    0s - loss: 0.1632 - acc: 0.9000
    Epoch 1524/2000
    0s - loss: 0.1688 - acc: 0.9000
    Epoch 1525/2000
    0s - loss: 0.1721 - acc: 0.8600
    Epoch 1526/2000
    0s - loss: 0.1687 - acc: 0.9000
    Epoch 1527/2000
    0s - loss: 0.1693 - acc: 0.8600
    Epoch 1528/2000
    0s - loss: 0.1723 - acc: 0.8600
    Epoch 1529/2000
    0s - loss: 0.1732 - acc: 0.8800
    Epoch 1530/2000
    0s - loss: 0.1833 - acc: 0.8800
    Epoch 1531/2000
    0s - loss: 0.1821 - acc: 0.9000
    Epoch 1532/2000
    0s - loss: 0.1727 - acc: 0.8800
    Epoch 1533/2000
    0s - loss: 0.1713 - acc: 0.8800
    Epoch 1534/2000
    0s - loss: 0.1847 - acc: 0.8600
    Epoch 1535/2000
    0s - loss: 0.1649 - acc: 0.9000
    Epoch 1536/2000
    0s - loss: 0.1622 - acc: 0.9000
    Epoch 1537/2000
    0s - loss: 0.1698 - acc: 0.8800
    Epoch 1538/2000
    0s - loss: 0.1755 - acc: 0.9000
    Epoch 1539/2000
    0s - loss: 0.1740 - acc: 0.8800
    Epoch 1540/2000
    0s - loss: 0.1813 - acc: 0.8800
    Epoch 1541/2000
    0s - loss: 0.1936 - acc: 0.8600
    Epoch 1542/2000
    0s - loss: 0.1773 - acc: 0.9000
    Epoch 1543/2000
    0s - loss: 0.1651 - acc: 0.9000
    Epoch 1544/2000
    0s - loss: 0.1765 - acc: 0.9200
    Epoch 1545/2000
    0s - loss: 0.1767 - acc: 0.9000
    Epoch 1546/2000
    0s - loss: 0.1707 - acc: 0.8800
    Epoch 1547/2000
    0s - loss: 0.1601 - acc: 0.9200
    Epoch 1548/2000
    0s - loss: 0.1697 - acc: 0.8600
    Epoch 1549/2000
    0s - loss: 0.1771 - acc: 0.8800
    Epoch 1550/2000
    0s - loss: 0.1657 - acc: 0.9200
    Epoch 1551/2000
    0s - loss: 0.1755 - acc: 0.8800
    Epoch 1552/2000
    0s - loss: 0.1709 - acc: 0.9000
    Epoch 1553/2000
    0s - loss: 0.1929 - acc: 0.9000
    Epoch 1554/2000
    0s - loss: 0.4704 - acc: 0.8200
    Epoch 1555/2000
    0s - loss: 0.1976 - acc: 0.9000
    Epoch 1556/2000
    0s - loss: 0.1703 - acc: 0.8800
    Epoch 1557/2000
    0s - loss: 0.1698 - acc: 0.8600
    Epoch 1558/2000
    0s - loss: 0.1662 - acc: 0.9000
    Epoch 1559/2000
    0s - loss: 0.1660 - acc: 0.9000
    Epoch 1560/2000
    0s - loss: 0.1704 - acc: 0.9000
    Epoch 1561/2000
    0s - loss: 0.1657 - acc: 0.9000
    Epoch 1562/2000
    0s - loss: 0.1650 - acc: 0.8600
    Epoch 1563/2000
    0s - loss: 0.1571 - acc: 0.9200
    Epoch 1564/2000
    0s - loss: 0.1695 - acc: 0.9200
    Epoch 1565/2000
    0s - loss: 0.1633 - acc: 0.8600
    Epoch 1566/2000
    0s - loss: 0.1673 - acc: 0.8800
    Epoch 1567/2000
    0s - loss: 0.1721 - acc: 0.9000
    Epoch 1568/2000
    0s - loss: 0.1741 - acc: 0.9200
    Epoch 1569/2000
    0s - loss: 0.1631 - acc: 0.8800
    Epoch 1570/2000
    0s - loss: 0.1640 - acc: 0.8600
    Epoch 1571/2000
    0s - loss: 0.1641 - acc: 0.9200
    Epoch 1572/2000
    0s - loss: 0.1648 - acc: 0.8800
    Epoch 1573/2000
    0s - loss: 0.1599 - acc: 0.9000
    Epoch 1574/2000
    0s - loss: 0.1767 - acc: 0.8800
    Epoch 1575/2000
    0s - loss: 0.1765 - acc: 0.8800
    Epoch 1576/2000
    0s - loss: 0.1689 - acc: 0.8600
    Epoch 1577/2000
    0s - loss: 0.1654 - acc: 0.8600
    Epoch 1578/2000
    0s - loss: 0.1683 - acc: 0.9000
    Epoch 1579/2000
    0s - loss: 0.1759 - acc: 0.9000
    Epoch 1580/2000
    0s - loss: 0.1658 - acc: 0.9000
    Epoch 1581/2000
    0s - loss: 0.1701 - acc: 0.8800
    Epoch 1582/2000
    0s - loss: 0.1687 - acc: 0.8600
    Epoch 1583/2000
    0s - loss: 0.1650 - acc: 0.9200
    Epoch 1584/2000
    0s - loss: 0.1594 - acc: 0.9200
    Epoch 1585/2000
    0s - loss: 0.1639 - acc: 0.9200
    Epoch 1586/2000
    0s - loss: 0.1620 - acc: 0.9200
    Epoch 1587/2000
    0s - loss: 0.1610 - acc: 0.8800
    Epoch 1588/2000
    0s - loss: 0.1671 - acc: 0.8800
    Epoch 1589/2000
    0s - loss: 0.1669 - acc: 0.8800
    Epoch 1590/2000
    0s - loss: 0.1615 - acc: 0.8800
    Epoch 1591/2000
    0s - loss: 0.1621 - acc: 0.9000
    Epoch 1592/2000
    0s - loss: 0.1633 - acc: 0.9000
    Epoch 1593/2000
    0s - loss: 0.1581 - acc: 0.8800
    Epoch 1594/2000
    0s - loss: 0.1629 - acc: 0.8800
    Epoch 1595/2000
    0s - loss: 0.1680 - acc: 0.8800
    Epoch 1596/2000
    0s - loss: 0.1746 - acc: 0.8600
    Epoch 1597/2000
    0s - loss: 0.1606 - acc: 0.8800
    Epoch 1598/2000
    0s - loss: 0.1707 - acc: 0.8800
    Epoch 1599/2000
    0s - loss: 0.1610 - acc: 0.9200
    Epoch 1600/2000
    0s - loss: 0.1673 - acc: 0.9000
    Epoch 1601/2000
    0s - loss: 0.1713 - acc: 0.9200
    Epoch 1602/2000
    0s - loss: 0.1644 - acc: 0.9000
    Epoch 1603/2000
    0s - loss: 0.1671 - acc: 0.9000
    Epoch 1604/2000
    0s - loss: 0.1633 - acc: 0.9000
    Epoch 1605/2000
    0s - loss: 0.1648 - acc: 0.8800
    Epoch 1606/2000
    0s - loss: 0.1684 - acc: 0.9200
    Epoch 1607/2000
    0s - loss: 0.1756 - acc: 0.9000
    Epoch 1608/2000
    0s - loss: 0.1832 - acc: 0.8400
    Epoch 1609/2000
    0s - loss: 0.1663 - acc: 0.8800
    Epoch 1610/2000
    0s - loss: 0.1714 - acc: 0.9200
    Epoch 1611/2000
    0s - loss: 0.1643 - acc: 0.9000
    Epoch 1612/2000
    0s - loss: 0.1661 - acc: 0.9000
    Epoch 1613/2000
    0s - loss: 0.1642 - acc: 0.8800
    Epoch 1614/2000
    0s - loss: 0.1626 - acc: 0.8800
    Epoch 1615/2000
    0s - loss: 0.1638 - acc: 0.9000
    Epoch 1616/2000
    0s - loss: 0.1566 - acc: 0.9200
    Epoch 1617/2000
    0s - loss: 0.1588 - acc: 0.8800
    Epoch 1618/2000
    0s - loss: 0.1634 - acc: 0.8800
    Epoch 1619/2000
    0s - loss: 0.1688 - acc: 0.8600
    Epoch 1620/2000
    0s - loss: 0.1620 - acc: 0.9200
    Epoch 1621/2000
    0s - loss: 0.1663 - acc: 0.9000
    Epoch 1622/2000
    0s - loss: 0.1909 - acc: 0.8400
    Epoch 1623/2000
    0s - loss: 0.1637 - acc: 0.8800
    Epoch 1624/2000
    0s - loss: 0.1636 - acc: 0.8800
    Epoch 1625/2000
    0s - loss: 0.1609 - acc: 0.9000
    Epoch 1626/2000
    0s - loss: 0.1776 - acc: 0.9200
    Epoch 1627/2000
    0s - loss: 0.1640 - acc: 0.9000
    Epoch 1628/2000
    0s - loss: 0.1651 - acc: 0.9000
    Epoch 1629/2000
    0s - loss: 0.1680 - acc: 0.8800
    Epoch 1630/2000
    0s - loss: 0.1711 - acc: 0.9200
    Epoch 1631/2000
    0s - loss: 0.1740 - acc: 0.8800
    Epoch 1632/2000
    0s - loss: 0.1906 - acc: 0.8600
    Epoch 1633/2000
    0s - loss: 0.5986 - acc: 0.8200
    Epoch 1634/2000
    0s - loss: 0.6148 - acc: 0.8200
    Epoch 1635/2000
    0s - loss: 0.3390 - acc: 0.8000
    Epoch 1636/2000
    0s - loss: 0.1886 - acc: 0.9000
    Epoch 1637/2000
    0s - loss: 0.1606 - acc: 0.9200
    Epoch 1638/2000
    0s - loss: 0.1634 - acc: 0.8800
    Epoch 1639/2000
    0s - loss: 0.1604 - acc: 0.9000
    Epoch 1640/2000
    0s - loss: 0.1628 - acc: 0.8800
    Epoch 1641/2000
    0s - loss: 0.1628 - acc: 0.9000
    Epoch 1642/2000
    0s - loss: 0.1540 - acc: 0.9000
    Epoch 1643/2000
    0s - loss: 0.1569 - acc: 0.9000
    Epoch 1644/2000
    0s - loss: 0.1581 - acc: 0.9200
    Epoch 1645/2000
    0s - loss: 0.1564 - acc: 0.8800
    Epoch 1646/2000
    0s - loss: 0.1566 - acc: 0.8800
    Epoch 1647/2000
    0s - loss: 0.1600 - acc: 0.9000
    Epoch 1648/2000
    0s - loss: 0.1647 - acc: 0.9000
    Epoch 1649/2000
    0s - loss: 0.1648 - acc: 0.9000
    Epoch 1650/2000
    0s - loss: 0.1640 - acc: 0.9000
    Epoch 1651/2000
    0s - loss: 0.1589 - acc: 0.9000
    Epoch 1652/2000
    0s - loss: 0.1526 - acc: 0.9000
    Epoch 1653/2000
    0s - loss: 0.1553 - acc: 0.9200
    Epoch 1654/2000
    0s - loss: 0.1598 - acc: 0.9200
    Epoch 1655/2000
    0s - loss: 0.1546 - acc: 0.9000
    Epoch 1656/2000
    0s - loss: 0.1601 - acc: 0.9000
    Epoch 1657/2000
    0s - loss: 0.1646 - acc: 0.9000
    Epoch 1658/2000
    0s - loss: 0.1615 - acc: 0.9200
    Epoch 1659/2000
    0s - loss: 0.1597 - acc: 0.9000
    Epoch 1660/2000
    0s - loss: 0.1593 - acc: 0.9000
    Epoch 1661/2000
    0s - loss: 0.1565 - acc: 0.9000
    Epoch 1662/2000
    0s - loss: 0.1585 - acc: 0.9200
    Epoch 1663/2000
    0s - loss: 0.1647 - acc: 0.8800
    Epoch 1664/2000
    0s - loss: 0.1564 - acc: 0.9200
    Epoch 1665/2000
    0s - loss: 0.1574 - acc: 0.9000
    Epoch 1666/2000
    0s - loss: 0.1584 - acc: 0.9200
    Epoch 1667/2000
    0s - loss: 0.1608 - acc: 0.9000
    Epoch 1668/2000
    0s - loss: 0.1611 - acc: 0.9000
    Epoch 1669/2000
    0s - loss: 0.1579 - acc: 0.8800
    Epoch 1670/2000
    0s - loss: 0.1534 - acc: 0.9000
    Epoch 1671/2000
    0s - loss: 0.1617 - acc: 0.8800
    Epoch 1672/2000
    0s - loss: 0.1614 - acc: 0.9000
    Epoch 1673/2000
    0s - loss: 0.1699 - acc: 0.8600
    Epoch 1674/2000
    0s - loss: 0.1558 - acc: 0.9000
    Epoch 1675/2000
    0s - loss: 0.1589 - acc: 0.8800
    Epoch 1676/2000
    0s - loss: 0.1609 - acc: 0.9000
    Epoch 1677/2000
    0s - loss: 0.1547 - acc: 0.8800
    Epoch 1678/2000
    0s - loss: 0.1649 - acc: 0.8800
    Epoch 1679/2000
    0s - loss: 0.1566 - acc: 0.9000
    Epoch 1680/2000
    0s - loss: 0.1572 - acc: 0.9000
    Epoch 1681/2000
    0s - loss: 0.1591 - acc: 0.8800
    Epoch 1682/2000
    0s - loss: 0.1667 - acc: 0.8800
    Epoch 1683/2000
    0s - loss: 0.1562 - acc: 0.8800
    Epoch 1684/2000
    0s - loss: 0.1572 - acc: 0.9000
    Epoch 1685/2000
    0s - loss: 0.1543 - acc: 0.9200
    Epoch 1686/2000
    0s - loss: 0.1584 - acc: 0.9000
    Epoch 1687/2000
    0s - loss: 0.1663 - acc: 0.8800
    Epoch 1688/2000
    0s - loss: 0.1683 - acc: 0.8800
    Epoch 1689/2000
    0s - loss: 0.1537 - acc: 0.9200
    Epoch 1690/2000
    0s - loss: 0.1546 - acc: 0.8800
    Epoch 1691/2000
    0s - loss: 0.1706 - acc: 0.8800
    Epoch 1692/2000
    0s - loss: 0.1704 - acc: 0.9200
    Epoch 1693/2000
    0s - loss: 0.1602 - acc: 0.9000
    Epoch 1694/2000
    0s - loss: 0.1615 - acc: 0.9000
    Epoch 1695/2000
    0s - loss: 0.1673 - acc: 0.8800
    Epoch 1696/2000
    0s - loss: 0.1530 - acc: 0.9000
    Epoch 1697/2000
    0s - loss: 0.1610 - acc: 0.8600
    Epoch 1698/2000
    0s - loss: 0.1620 - acc: 0.9000
    Epoch 1699/2000
    0s - loss: 0.1515 - acc: 0.9200
    Epoch 1700/2000
    0s - loss: 0.1556 - acc: 0.9200
    Epoch 1701/2000
    0s - loss: 0.1601 - acc: 0.9000
    Epoch 1702/2000
    0s - loss: 0.1545 - acc: 0.9200
    Epoch 1703/2000
    0s - loss: 0.1670 - acc: 0.9000
    Epoch 1704/2000
    0s - loss: 0.1588 - acc: 0.9000
    Epoch 1705/2000
    0s - loss: 0.1637 - acc: 0.8800
    Epoch 1706/2000
    0s - loss: 0.1591 - acc: 0.9200
    Epoch 1707/2000
    0s - loss: 0.1707 - acc: 0.8600
    Epoch 1708/2000
    0s - loss: 0.1597 - acc: 0.9000
    Epoch 1709/2000
    0s - loss: 0.1548 - acc: 0.9000
    Epoch 1710/2000
    0s - loss: 0.1590 - acc: 0.9000
    Epoch 1711/2000
    0s - loss: 0.1533 - acc: 0.9000
    Epoch 1712/2000
    0s - loss: 0.1567 - acc: 0.8800
    Epoch 1713/2000
    0s - loss: 0.1540 - acc: 0.9000
    Epoch 1714/2000
    0s - loss: 0.1641 - acc: 0.8800
    Epoch 1715/2000
    0s - loss: 0.1586 - acc: 0.8800
    Epoch 1716/2000
    0s - loss: 0.1541 - acc: 0.9200
    Epoch 1717/2000
    0s - loss: 0.1533 - acc: 0.8800
    Epoch 1718/2000
    0s - loss: 0.1557 - acc: 0.8800
    Epoch 1719/2000
    0s - loss: 0.1557 - acc: 0.8600
    Epoch 1720/2000
    0s - loss: 0.1572 - acc: 0.9000
    Epoch 1721/2000
    0s - loss: 0.1593 - acc: 0.9200
    Epoch 1722/2000
    0s - loss: 0.1552 - acc: 0.9000
    Epoch 1723/2000
    0s - loss: 0.1610 - acc: 0.9000
    Epoch 1724/2000
    0s - loss: 0.1661 - acc: 0.9200
    Epoch 1725/2000
    0s - loss: 0.1585 - acc: 0.9000
    Epoch 1726/2000
    0s - loss: 0.1619 - acc: 0.9000
    Epoch 1727/2000
    0s - loss: 0.1799 - acc: 0.9000
    Epoch 1728/2000
    0s - loss: 0.1867 - acc: 0.8600
    Epoch 1729/2000
    0s - loss: 0.1599 - acc: 0.9000
    Epoch 1730/2000
    0s - loss: 0.1628 - acc: 0.8800
    Epoch 1731/2000
    0s - loss: 0.1594 - acc: 0.9000
    Epoch 1732/2000
    0s - loss: 0.1592 - acc: 0.9000
    Epoch 1733/2000
    0s - loss: 0.1540 - acc: 0.9200
    Epoch 1734/2000
    0s - loss: 0.1514 - acc: 0.9200
    Epoch 1735/2000
    0s - loss: 0.1553 - acc: 0.8800
    Epoch 1736/2000
    0s - loss: 0.1543 - acc: 0.8800
    Epoch 1737/2000
    0s - loss: 0.1539 - acc: 0.9000
    Epoch 1738/2000
    0s - loss: 0.1500 - acc: 0.9000
    Epoch 1739/2000
    0s - loss: 0.1529 - acc: 0.9000
    Epoch 1740/2000
    0s - loss: 0.1516 - acc: 0.9000
    Epoch 1741/2000
    0s - loss: 0.1548 - acc: 0.9000
    Epoch 1742/2000
    0s - loss: 0.1664 - acc: 0.8800
    Epoch 1743/2000
    0s - loss: 0.1569 - acc: 0.8800
    Epoch 1744/2000
    0s - loss: 0.1606 - acc: 0.9000
    Epoch 1745/2000
    0s - loss: 0.1548 - acc: 0.9200
    Epoch 1746/2000
    0s - loss: 0.1551 - acc: 0.8800
    Epoch 1747/2000
    0s - loss: 0.1585 - acc: 0.9000
    Epoch 1748/2000
    0s - loss: 0.1572 - acc: 0.9200
    Epoch 1749/2000
    0s - loss: 0.1563 - acc: 0.8800
    Epoch 1750/2000
    0s - loss: 0.1507 - acc: 0.8800
    Epoch 1751/2000
    0s - loss: 0.1531 - acc: 0.9200
    Epoch 1752/2000
    0s - loss: 0.1583 - acc: 0.9000
    Epoch 1753/2000
    0s - loss: 0.1602 - acc: 0.9200
    Epoch 1754/2000
    0s - loss: 0.1567 - acc: 0.8800
    Epoch 1755/2000
    0s - loss: 0.1528 - acc: 0.8800
    Epoch 1756/2000
    0s - loss: 0.1567 - acc: 0.9000
    Epoch 1757/2000
    0s - loss: 0.1514 - acc: 0.9200
    Epoch 1758/2000
    0s - loss: 0.1565 - acc: 0.9000
    Epoch 1759/2000
    0s - loss: 0.1470 - acc: 0.8800
    Epoch 1760/2000
    0s - loss: 0.1593 - acc: 0.9200
    Epoch 1761/2000
    0s - loss: 0.1554 - acc: 0.9200
    Epoch 1762/2000
    0s - loss: 0.1564 - acc: 0.9200
    Epoch 1763/2000
    0s - loss: 0.1538 - acc: 0.9000
    Epoch 1764/2000
    0s - loss: 0.1527 - acc: 0.9000
    Epoch 1765/2000
    0s - loss: 0.1501 - acc: 0.9000
    Epoch 1766/2000
    0s - loss: 0.1623 - acc: 0.9000
    Epoch 1767/2000
    0s - loss: 0.1603 - acc: 0.8800
    Epoch 1768/2000
    0s - loss: 0.1509 - acc: 0.8800
    Epoch 1769/2000
    0s - loss: 0.1530 - acc: 0.9000
    Epoch 1770/2000
    0s - loss: 0.1553 - acc: 0.9000
    Epoch 1771/2000
    0s - loss: 0.1544 - acc: 0.8600
    Epoch 1772/2000
    0s - loss: 0.1573 - acc: 0.9000
    Epoch 1773/2000
    0s - loss: 0.1530 - acc: 0.9200
    Epoch 1774/2000
    0s - loss: 0.1534 - acc: 0.9200
    Epoch 1775/2000
    0s - loss: 0.1578 - acc: 0.8800
    Epoch 1776/2000
    0s - loss: 0.1583 - acc: 0.8800
    Epoch 1777/2000
    0s - loss: 0.1548 - acc: 0.8600
    Epoch 1778/2000
    0s - loss: 0.1547 - acc: 0.9000
    Epoch 1779/2000
    0s - loss: 0.1518 - acc: 0.8800
    Epoch 1780/2000
    0s - loss: 0.1565 - acc: 0.9200
    Epoch 1781/2000
    0s - loss: 0.1596 - acc: 0.8800
    Epoch 1782/2000
    0s - loss: 0.1605 - acc: 0.8800
    Epoch 1783/2000
    0s - loss: 0.1711 - acc: 0.8800
    Epoch 1784/2000
    0s - loss: 0.2533 - acc: 0.8400
    Epoch 1785/2000
    0s - loss: 0.5757 - acc: 0.8000
    Epoch 1786/2000
    0s - loss: 1.0819 - acc: 0.7400
    Epoch 1787/2000
    0s - loss: 0.7413 - acc: 0.7400
    Epoch 1788/2000
    0s - loss: 0.1762 - acc: 0.9200
    Epoch 1789/2000
    0s - loss: 0.1569 - acc: 0.9200
    Epoch 1790/2000
    0s - loss: 0.1588 - acc: 0.9200
    Epoch 1791/2000
    0s - loss: 0.1520 - acc: 0.9200
    Epoch 1792/2000
    0s - loss: 0.1492 - acc: 0.9000
    Epoch 1793/2000
    0s - loss: 0.1522 - acc: 0.9200
    Epoch 1794/2000
    0s - loss: 0.1508 - acc: 0.9000
    Epoch 1795/2000
    0s - loss: 0.1477 - acc: 0.8800
    Epoch 1796/2000
    0s - loss: 0.1495 - acc: 0.9000
    Epoch 1797/2000
    0s - loss: 0.1461 - acc: 0.9000
    Epoch 1798/2000
    0s - loss: 0.1491 - acc: 0.8800
    Epoch 1799/2000
    0s - loss: 0.1488 - acc: 0.8800
    Epoch 1800/2000
    0s - loss: 0.1513 - acc: 0.9000
    Epoch 1801/2000
    0s - loss: 0.1487 - acc: 0.9000
    Epoch 1802/2000
    0s - loss: 0.1514 - acc: 0.8800
    Epoch 1803/2000
    0s - loss: 0.1501 - acc: 0.8800
    Epoch 1804/2000
    0s - loss: 0.1514 - acc: 0.8800
    Epoch 1805/2000
    0s - loss: 0.1479 - acc: 0.9200
    Epoch 1806/2000
    0s - loss: 0.1493 - acc: 0.9000
    Epoch 1807/2000
    0s - loss: 0.1474 - acc: 0.9200
    Epoch 1808/2000
    0s - loss: 0.1463 - acc: 0.9200
    Epoch 1809/2000
    0s - loss: 0.1514 - acc: 0.9000
    Epoch 1810/2000
    0s - loss: 0.1495 - acc: 0.8800
    Epoch 1811/2000
    0s - loss: 0.1531 - acc: 0.9000
    Epoch 1812/2000
    0s - loss: 0.1520 - acc: 0.9200
    Epoch 1813/2000
    0s - loss: 0.1474 - acc: 0.9200
    Epoch 1814/2000
    0s - loss: 0.1465 - acc: 0.9000
    Epoch 1815/2000
    0s - loss: 0.1517 - acc: 0.8800
    Epoch 1816/2000
    0s - loss: 0.1476 - acc: 0.9200
    Epoch 1817/2000
    0s - loss: 0.1491 - acc: 0.9000
    Epoch 1818/2000
    0s - loss: 0.1548 - acc: 0.9000
    Epoch 1819/2000
    0s - loss: 0.1508 - acc: 0.8800
    Epoch 1820/2000
    0s - loss: 0.1530 - acc: 0.8800
    Epoch 1821/2000
    0s - loss: 0.1522 - acc: 0.9000
    Epoch 1822/2000
    0s - loss: 0.1468 - acc: 0.9200
    Epoch 1823/2000
    0s - loss: 0.1442 - acc: 0.9000
    Epoch 1824/2000
    0s - loss: 0.1454 - acc: 0.9000
    Epoch 1825/2000
    0s - loss: 0.1471 - acc: 0.9200
    Epoch 1826/2000
    0s - loss: 0.1494 - acc: 0.9000
    Epoch 1827/2000
    0s - loss: 0.1493 - acc: 0.9000
    Epoch 1828/2000
    0s - loss: 0.1447 - acc: 0.9200
    Epoch 1829/2000
    0s - loss: 0.1502 - acc: 0.9000
    Epoch 1830/2000
    0s - loss: 0.1527 - acc: 0.9000
    Epoch 1831/2000
    0s - loss: 0.1511 - acc: 0.9000
    Epoch 1832/2000
    0s - loss: 0.1551 - acc: 0.9200
    Epoch 1833/2000
    0s - loss: 0.1464 - acc: 0.8800
    Epoch 1834/2000
    0s - loss: 0.1511 - acc: 0.8800
    Epoch 1835/2000
    0s - loss: 0.1468 - acc: 0.9200
    Epoch 1836/2000
    0s - loss: 0.1505 - acc: 0.9000
    Epoch 1837/2000
    0s - loss: 0.1438 - acc: 0.9200
    Epoch 1838/2000
    0s - loss: 0.1475 - acc: 0.9000
    Epoch 1839/2000
    0s - loss: 0.1525 - acc: 0.9000
    Epoch 1840/2000
    0s - loss: 0.1574 - acc: 0.9000
    Epoch 1841/2000
    0s - loss: 0.1459 - acc: 0.9200
    Epoch 1842/2000
    0s - loss: 0.1504 - acc: 0.8600
    Epoch 1843/2000
    0s - loss: 0.1478 - acc: 0.8800
    Epoch 1844/2000
    0s - loss: 0.1447 - acc: 0.9200
    Epoch 1845/2000
    0s - loss: 0.1546 - acc: 0.8800
    Epoch 1846/2000
    0s - loss: 0.1541 - acc: 0.8800
    Epoch 1847/2000
    0s - loss: 0.1511 - acc: 0.8800
    Epoch 1848/2000
    0s - loss: 0.1466 - acc: 0.9000
    Epoch 1849/2000
    0s - loss: 0.1498 - acc: 0.9000
    Epoch 1850/2000
    0s - loss: 0.1463 - acc: 0.9000
    Epoch 1851/2000
    0s - loss: 0.1480 - acc: 0.8800
    Epoch 1852/2000
    0s - loss: 0.1474 - acc: 0.9200
    Epoch 1853/2000
    0s - loss: 0.1497 - acc: 0.8600
    Epoch 1854/2000
    0s - loss: 0.1503 - acc: 0.9000
    Epoch 1855/2000
    0s - loss: 0.1479 - acc: 0.9000
    Epoch 1856/2000
    0s - loss: 0.1487 - acc: 0.9200
    Epoch 1857/2000
    0s - loss: 0.1526 - acc: 0.9200
    Epoch 1858/2000
    0s - loss: 0.1522 - acc: 0.9000
    Epoch 1859/2000
    0s - loss: 0.1553 - acc: 0.8800
    Epoch 1860/2000
    0s - loss: 0.1478 - acc: 0.8600
    Epoch 1861/2000
    0s - loss: 0.1460 - acc: 0.9200
    Epoch 1862/2000
    0s - loss: 0.1543 - acc: 0.8600
    Epoch 1863/2000
    0s - loss: 0.1478 - acc: 0.9000
    Epoch 1864/2000
    0s - loss: 0.1501 - acc: 0.8800
    Epoch 1865/2000
    0s - loss: 0.1556 - acc: 0.9200
    Epoch 1866/2000
    0s - loss: 0.1533 - acc: 0.8600
    Epoch 1867/2000
    0s - loss: 0.1509 - acc: 0.9200
    Epoch 1868/2000
    0s - loss: 0.1481 - acc: 0.8800
    Epoch 1869/2000
    0s - loss: 0.1496 - acc: 0.9000
    Epoch 1870/2000
    0s - loss: 0.1496 - acc: 0.8800
    Epoch 1871/2000
    0s - loss: 0.1525 - acc: 0.8600
    Epoch 1872/2000
    0s - loss: 0.1471 - acc: 0.9000
    Epoch 1873/2000
    0s - loss: 0.1496 - acc: 0.8800
    Epoch 1874/2000
    0s - loss: 0.1556 - acc: 0.9000
    Epoch 1875/2000
    0s - loss: 0.1497 - acc: 0.8600
    Epoch 1876/2000
    0s - loss: 0.1481 - acc: 0.8800
    Epoch 1877/2000
    0s - loss: 0.1492 - acc: 0.9000
    Epoch 1878/2000
    0s - loss: 0.1519 - acc: 0.8800
    Epoch 1879/2000
    0s - loss: 0.1495 - acc: 0.8600
    Epoch 1880/2000
    0s - loss: 0.1490 - acc: 0.9000
    Epoch 1881/2000
    0s - loss: 0.1467 - acc: 0.9000
    Epoch 1882/2000
    0s - loss: 0.1748 - acc: 0.8800
    Epoch 1883/2000
    0s - loss: 0.1566 - acc: 0.8800
    Epoch 1884/2000
    0s - loss: 0.1739 - acc: 0.8800
    Epoch 1885/2000
    0s - loss: 0.1872 - acc: 0.8800
    Epoch 1886/2000
    0s - loss: 0.2470 - acc: 0.8600
    Epoch 1887/2000
    0s - loss: 0.3445 - acc: 0.8600
    Epoch 1888/2000
    0s - loss: 0.4486 - acc: 0.8400
    Epoch 1889/2000
    0s - loss: 0.6717 - acc: 0.8200
    Epoch 1890/2000
    0s - loss: 0.2690 - acc: 0.8600
    Epoch 1891/2000
    0s - loss: 0.1570 - acc: 0.9000
    Epoch 1892/2000
    0s - loss: 0.1535 - acc: 0.9000
    Epoch 1893/2000
    0s - loss: 0.1497 - acc: 0.9000
    Epoch 1894/2000
    0s - loss: 0.1518 - acc: 0.9000
    Epoch 1895/2000
    0s - loss: 0.1443 - acc: 0.8800
    Epoch 1896/2000
    0s - loss: 0.1510 - acc: 0.8800
    Epoch 1897/2000
    0s - loss: 0.1439 - acc: 0.9000
    Epoch 1898/2000
    0s - loss: 0.1464 - acc: 0.8800
    Epoch 1899/2000
    0s - loss: 0.1498 - acc: 0.9000
    Epoch 1900/2000
    0s - loss: 0.1482 - acc: 0.8800
    Epoch 1901/2000
    0s - loss: 0.1457 - acc: 0.8800
    Epoch 1902/2000
    0s - loss: 0.1492 - acc: 0.8800
    Epoch 1903/2000
    0s - loss: 0.1422 - acc: 0.9200
    Epoch 1904/2000
    0s - loss: 0.1486 - acc: 0.8800
    Epoch 1905/2000
    0s - loss: 0.1453 - acc: 0.9000
    Epoch 1906/2000
    0s - loss: 0.1460 - acc: 0.9000
    Epoch 1907/2000
    0s - loss: 0.1414 - acc: 0.9000
    Epoch 1908/2000
    0s - loss: 0.1450 - acc: 0.9000
    Epoch 1909/2000
    0s - loss: 0.1450 - acc: 0.8800
    Epoch 1910/2000
    0s - loss: 0.1437 - acc: 0.9000
    Epoch 1911/2000
    0s - loss: 0.1495 - acc: 0.9000
    Epoch 1912/2000
    0s - loss: 0.1467 - acc: 0.8800
    Epoch 1913/2000
    0s - loss: 0.1449 - acc: 0.9000
    Epoch 1914/2000
    0s - loss: 0.1477 - acc: 0.9000
    Epoch 1915/2000
    0s - loss: 0.1420 - acc: 0.9000
    Epoch 1916/2000
    0s - loss: 0.1442 - acc: 0.9000
    Epoch 1917/2000
    0s - loss: 0.1476 - acc: 0.8800
    Epoch 1918/2000
    0s - loss: 0.1459 - acc: 0.8800
    Epoch 1919/2000
    0s - loss: 0.1462 - acc: 0.8800
    Epoch 1920/2000
    0s - loss: 0.1472 - acc: 0.9000
    Epoch 1921/2000
    0s - loss: 0.1465 - acc: 0.9000
    Epoch 1922/2000
    0s - loss: 0.1429 - acc: 0.9000
    Epoch 1923/2000
    0s - loss: 0.1502 - acc: 0.9000
    Epoch 1924/2000
    0s - loss: 0.1445 - acc: 0.9200
    Epoch 1925/2000
    0s - loss: 0.1487 - acc: 0.8800
    Epoch 1926/2000
    0s - loss: 0.1451 - acc: 0.8800
    Epoch 1927/2000
    0s - loss: 0.1469 - acc: 0.9000
    Epoch 1928/2000
    0s - loss: 0.1483 - acc: 0.8800
    Epoch 1929/2000
    0s - loss: 0.1428 - acc: 0.9000
    Epoch 1930/2000
    0s - loss: 0.1448 - acc: 0.8800
    Epoch 1931/2000
    0s - loss: 0.1468 - acc: 0.9000
    Epoch 1932/2000
    0s - loss: 0.1440 - acc: 0.8800
    Epoch 1933/2000
    0s - loss: 0.1452 - acc: 0.9000
    Epoch 1934/2000
    0s - loss: 0.1444 - acc: 0.8800
    Epoch 1935/2000
    0s - loss: 0.1482 - acc: 0.8600
    Epoch 1936/2000
    0s - loss: 0.1463 - acc: 0.8800
    Epoch 1937/2000
    0s - loss: 0.1440 - acc: 0.8800
    Epoch 1938/2000
    0s - loss: 0.1464 - acc: 0.8800
    Epoch 1939/2000
    0s - loss: 0.1449 - acc: 0.9000
    Epoch 1940/2000
    0s - loss: 0.1421 - acc: 0.9000
    Epoch 1941/2000
    0s - loss: 0.1457 - acc: 0.8600
    Epoch 1942/2000
    0s - loss: 0.1455 - acc: 0.8800
    Epoch 1943/2000
    0s - loss: 0.1466 - acc: 0.8800
    Epoch 1944/2000
    0s - loss: 0.1512 - acc: 0.9000
    Epoch 1945/2000
    0s - loss: 0.1535 - acc: 0.8800
    Epoch 1946/2000
    0s - loss: 0.1465 - acc: 0.8800
    Epoch 1947/2000
    0s - loss: 0.1410 - acc: 0.9200
    Epoch 1948/2000
    0s - loss: 0.1425 - acc: 0.9000
    Epoch 1949/2000
    0s - loss: 0.1519 - acc: 0.8800
    Epoch 1950/2000
    0s - loss: 0.1539 - acc: 0.8600
    Epoch 1951/2000
    0s - loss: 0.1604 - acc: 0.8600
    Epoch 1952/2000
    0s - loss: 0.1472 - acc: 0.9200
    Epoch 1953/2000
    0s - loss: 0.1467 - acc: 0.9000
    Epoch 1954/2000
    0s - loss: 0.1466 - acc: 0.8800
    Epoch 1955/2000
    0s - loss: 0.1473 - acc: 0.9000
    Epoch 1956/2000
    0s - loss: 0.1489 - acc: 0.8800
    Epoch 1957/2000
    0s - loss: 0.1456 - acc: 0.9000
    Epoch 1958/2000
    0s - loss: 0.1526 - acc: 0.9200
    Epoch 1959/2000
    0s - loss: 0.1483 - acc: 0.9000
    Epoch 1960/2000
    0s - loss: 0.1423 - acc: 0.9000
    Epoch 1961/2000
    0s - loss: 0.1459 - acc: 0.9200
    Epoch 1962/2000
    0s - loss: 0.1452 - acc: 0.9000
    Epoch 1963/2000
    0s - loss: 0.1436 - acc: 0.8600
    Epoch 1964/2000
    0s - loss: 0.1463 - acc: 0.9000
    Epoch 1965/2000
    0s - loss: 0.1400 - acc: 0.9000
    Epoch 1966/2000
    0s - loss: 0.1458 - acc: 0.9000
    Epoch 1967/2000
    0s - loss: 0.1532 - acc: 0.9000
    Epoch 1968/2000
    0s - loss: 0.1481 - acc: 0.8800
    Epoch 1969/2000
    0s - loss: 0.1460 - acc: 0.9000
    Epoch 1970/2000
    0s - loss: 0.1491 - acc: 0.9200
    Epoch 1971/2000
    0s - loss: 0.1463 - acc: 0.8800
    Epoch 1972/2000
    0s - loss: 0.1652 - acc: 0.8600
    Epoch 1973/2000
    0s - loss: 0.1545 - acc: 0.9200
    Epoch 1974/2000
    0s - loss: 0.2409 - acc: 0.8400
    Epoch 1975/2000
    0s - loss: 0.5337 - acc: 0.8200
    Epoch 1976/2000
    0s - loss: 0.1753 - acc: 0.8800
    Epoch 1977/2000
    0s - loss: 0.1457 - acc: 0.8800
    Epoch 1978/2000
    0s - loss: 0.1451 - acc: 0.9200
    Epoch 1979/2000
    0s - loss: 0.1466 - acc: 0.8800
    Epoch 1980/2000
    0s - loss: 0.1403 - acc: 0.9000
    Epoch 1981/2000
    0s - loss: 0.1440 - acc: 0.8600
    Epoch 1982/2000
    0s - loss: 0.1408 - acc: 0.9200
    Epoch 1983/2000
    0s - loss: 0.1415 - acc: 0.8800
    Epoch 1984/2000
    0s - loss: 0.1466 - acc: 0.8800
    Epoch 1985/2000
    0s - loss: 0.1399 - acc: 0.9000
    Epoch 1986/2000
    0s - loss: 0.1449 - acc: 0.8600
    Epoch 1987/2000
    0s - loss: 0.1394 - acc: 0.9000
    Epoch 1988/2000
    0s - loss: 0.1446 - acc: 0.9200
    Epoch 1989/2000
    0s - loss: 0.1422 - acc: 0.9000
    Epoch 1990/2000
    0s - loss: 0.1429 - acc: 0.8800
    Epoch 1991/2000
    0s - loss: 0.1465 - acc: 0.8800
    Epoch 1992/2000
    0s - loss: 0.1426 - acc: 0.9000
    Epoch 1993/2000
    0s - loss: 0.1383 - acc: 0.9000
    Epoch 1994/2000
    0s - loss: 0.1458 - acc: 0.8800
    Epoch 1995/2000
    0s - loss: 0.1414 - acc: 0.9000
    Epoch 1996/2000
    0s - loss: 0.1449 - acc: 0.8600
    Epoch 1997/2000
    0s - loss: 0.1440 - acc: 0.9000
    Epoch 1998/2000
    0s - loss: 0.1441 - acc: 0.9200
    Epoch 1999/2000
    0s - loss: 0.1429 - acc: 0.8800
    Epoch 2000/2000
    0s - loss: 0.1470 - acc: 0.9000
    32/50 [==================>...........] - ETA: 0sacc: 92.00%
    ('one step prediction : ', ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'e8', 'f8', 'g8', 'g8', 'g4', 'g8', 'e8', 'e8', 'e8', 'f8', 'g4', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'f4', 'e8', 'e8', 'e8', 'e8', 'f8', 'f8', 'g4', 'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4'])
    ('full song prediction : ', ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8', 'd8'])


한 스텝 예측 결과과 곡 전체 예측 결과를 악보로 그려보았습니다. 이 중 틀린 부분을 빨간색 박스로 표시해봤습니다. 총 50개 예측 중 4개가 틀려서 92%의 정확도가 나왔습니다. 중간에 틀릭 부분이 생기면 곡 전체를 예측하는 데 있어서는 그리 좋은 성능이 나오지 않습니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_LSTM_song.png)

---

### 상태유지 LSTM 모델

이번에는 상태유지(Stateful) LSTM 모델에 대해서 알아보겠습니다. 여기서 `상태유지`라는 것은 현재 학습된 상태가 다음 학습 시 초기 상태로 전달된다는 것을 의미합니다. 

    상태유지 모드에서는 현재 샘플의 학습 상태가 다음 샘플의 초기 상태로 전달된다.
    
긴 시퀀드 데이터를 처리할 때, LSTM 모델은 상태유지 모드에서 그 진가를 발휘합니다. 긴 시퀀스 데이터를 샘플 단위로 잘라서 학습을 시키더라도 LSTM 내부적으로 기억할 것은 기억하고 버릴 것은 버려서 기억해야할 중요한 정보만 이어갈 수 있도록 상태가 유지되기 때문입니다. 상태유지 LSTM 모델을 생성하기 위해서는 LSTM 레이어 생성 시, stateful=True로 설정하면 됩니다. 또한 상태유지 모드에서는 입력형태를 batch_input_shape = (배치사이즈, 타임스텝, 속성)으로 설정해야 합니다. 상태유지 모드에서 배치사이즈 개념은 조금 어려우므로 다음 장에서 다루기로 하겠습니다. 


```python
model = Sequential()
model.add(LSTM(128, batch_input_shape = (1, 4, 1), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))
```

상태유지 모드에서는 모델 학습 시에 `상태 초기화`에 대한 고민이 필요합니다. 현재 샘플 학습 상태가 다음 샘플 학습의 초기상태로 전달되는 식인데, 현재 샘플과 다음 샘플 간의 순차적인 관계가 없을 경우에는 상태가 유지되지 않고 초기화가 되어야 합니다. 다음 상황이 이러한 경우에 해당됩니다.

- 마지막 샘플 학습이 마치고, 새로운 에포크 수행 시에는 새로운 샘플 학습을 해야하므로 상태 초기화 필요
- 한 에포크 안에 여러 시퀀스 데이터 세트가 있을 경우, 새로운 시퀀스 데이터 세트를 학습 전에 상태 초기화 필요

현재 코드에서는 한 곡을 가지고 계속 학습을 시키고 있으므로 새로운 에포크 시작 시에만 상태 초기화를 수행하면 됩니다.


```python
# 모델 학습시키기
num_epochs = 2000

for epoch_idx in range(num_epochs):
    print ('epochs : ' + str(epoch_idx) )
    model.fit(train_X, train_Y, epochs=1, batch_size=1, verbose=2, shuffle=False) # 50 is X.shape[0]
    model.reset_states()
```

아래 그림은 이 모델로 악보를 학습할 경우를 나타낸 것입니다. 거의 기본 LSTM 모델과 동일하지만 학습된 상태가 다음 샘플 학습 시에 초기 상태로 입력되는 것을 보실 수 있습니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_train_stateful_LSTM.png)

전체 소스는 다음과 같습니다.


```python
# 코드 사전 정의

code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}

idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
            7:'c8', 8:'d8', 9:'e8', 10:'f8', 11:'g8', 12:'a8', 13:'b8'}

# 데이터셋 생성 함수

import numpy as np

def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq)-window_size):
        subset = seq[i:(i+window_size+1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)

# 시퀀스 데이터 정의

seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
       'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
       'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
       'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

# 데이터셋 생성

dataset = seq2dataset(seq, window_size = 4)

print(dataset.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# 랜덤시드 고정시키기
np.random.seed(5)

# 입력(X)과 출력(Y) 변수로 분리하기
train_X = dataset[:,0:4]
train_Y = dataset[:,4]

max_idx_value = 13

# 입력값 정규화 시키기
train_X = train_X / float(max_idx_value)

# 입력을 (샘플 수, 타임스텝, 특성 수)로 형태 변환
train_X = np.reshape(train_X, (50, 4, 1))

# 라벨값에 대한 one-hot 인코딩 수행
train_Y = np_utils.to_categorical(train_Y)

one_hot_vec_size = train_Y.shape[1]

print("one hot encoding vector size is ", one_hot_vec_size)

# 모델 구성하기
model = Sequential()
model.add(LSTM(128, batch_input_shape = (1, 4, 1), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))
    
# 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습시키기
num_epochs = 2000

for epoch_idx in range(num_epochs):
    print ('epochs : ' + str(epoch_idx) )
    model.fit(train_X, train_Y, epochs=1, batch_size=1, verbose=2, shuffle=False) # 50 is X.shape[0]
    model.reset_states()

# 모델 평가하기
scores = model.evaluate(train_X, train_Y, batch_size=1)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
model.reset_states()

# 예측하기

pred_count = 50 # 최대 예측 개수 정의

# 한 스텝 예측

seq_out = ['g8', 'e8', 'e4', 'f8']
pred_out = model.predict(train_X, batch_size=1)

for i in range(pred_count):
    idx = np.argmax(pred_out[i]) # one-hot 인코딩을 인덱스 값으로 변환
    seq_out.append(idx2code[idx]) # seq_out는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장
    
print("one step prediction : ", seq_out)

# 곡 전체 예측

seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in
seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in] # 코드를 인덱스값으로 변환

for i in range(pred_count):
    sample_in = np.array(seq_in)
    sample_in = np.reshape(sample_in, (1, 4, 1)) # 샘플 수, 타입스텝 수, 속성 수
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx / float(max_idx_value))
    seq_in.pop(0)

model.reset_states()
    
print("full song prediction : ", seq_out)
```

    (50, 5)
    ('one hot encoding vector size is ', 12)
    epochs : 0
    Epoch 1/1
    3s - loss: 2.3478 - acc: 0.1400
    epochs : 1
    Epoch 1/1
    0s - loss: 2.0412 - acc: 0.3400
    epochs : 2
    Epoch 1/1
    0s - loss: 1.9632 - acc: 0.3400
    epochs : 3
    Epoch 1/1
    0s - loss: 1.9467 - acc: 0.3400
    epochs : 4
    Epoch 1/1
    0s - loss: 1.9367 - acc: 0.3400
    epochs : 5
    Epoch 1/1
    0s - loss: 1.9299 - acc: 0.3400
    epochs : 6
    Epoch 1/1
    0s - loss: 1.9247 - acc: 0.3600
    epochs : 7
    Epoch 1/1
    0s - loss: 1.9203 - acc: 0.3600
    epochs : 8
    Epoch 1/1
    0s - loss: 1.9166 - acc: 0.3600
    epochs : 9
    Epoch 1/1
    0s - loss: 1.9132 - acc: 0.3600
    epochs : 10
    Epoch 1/1
    0s - loss: 1.9100 - acc: 0.3600
    epochs : 11
    Epoch 1/1
    0s - loss: 1.9068 - acc: 0.3600
    epochs : 12
    Epoch 1/1
    0s - loss: 1.9035 - acc: 0.3600
    epochs : 13
    Epoch 1/1
    0s - loss: 1.8996 - acc: 0.3600
    epochs : 14
    Epoch 1/1
    0s - loss: 1.8954 - acc: 0.3600
    epochs : 15
    Epoch 1/1
    0s - loss: 1.9017 - acc: 0.3600
    epochs : 16
    Epoch 1/1
    0s - loss: 1.8903 - acc: 0.3600
    epochs : 17
    Epoch 1/1
    0s - loss: 2.0696 - acc: 0.3000
    epochs : 18
    Epoch 1/1
    0s - loss: 1.9036 - acc: 0.3600
    epochs : 19
    Epoch 1/1
    0s - loss: 1.8964 - acc: 0.3600
    epochs : 20
    Epoch 1/1
    0s - loss: 1.8917 - acc: 0.3600
    epochs : 21
    Epoch 1/1
    0s - loss: 1.8863 - acc: 0.3600
    epochs : 22
    Epoch 1/1
    0s - loss: 1.8805 - acc: 0.3600
    epochs : 23
    Epoch 1/1
    0s - loss: 1.8734 - acc: 0.3600
    epochs : 24
    Epoch 1/1
    0s - loss: 1.8667 - acc: 0.3600
    epochs : 25
    Epoch 1/1
    0s - loss: 1.8560 - acc: 0.3600
    epochs : 26
    Epoch 1/1
    0s - loss: 1.8461 - acc: 0.3600
    epochs : 27
    Epoch 1/1
    0s - loss: 1.8308 - acc: 0.3800
    epochs : 28
    Epoch 1/1
    0s - loss: 1.8014 - acc: 0.3800
    epochs : 29
    Epoch 1/1
    0s - loss: 1.8327 - acc: 0.3800
    epochs : 30
    Epoch 1/1
    0s - loss: 1.8288 - acc: 0.3600
    epochs : 31
    Epoch 1/1
    0s - loss: 1.7854 - acc: 0.3800
    epochs : 32
    Epoch 1/1
    0s - loss: 1.7542 - acc: 0.3800
    epochs : 33
    Epoch 1/1
    0s - loss: 1.7157 - acc: 0.3600
    epochs : 34
    Epoch 1/1
    0s - loss: 1.8813 - acc: 0.3600
    epochs : 35
    Epoch 1/1
    0s - loss: 1.7886 - acc: 0.4000
    epochs : 36
    Epoch 1/1
    0s - loss: 1.7950 - acc: 0.3200
    epochs : 37
    Epoch 1/1
    0s - loss: 1.7097 - acc: 0.4200
    epochs : 38
    Epoch 1/1
    0s - loss: 1.6844 - acc: 0.4200
    epochs : 39
    Epoch 1/1
    0s - loss: 1.6260 - acc: 0.4000
    epochs : 40
    Epoch 1/1
    0s - loss: 1.6549 - acc: 0.4600
    epochs : 41
    Epoch 1/1
    0s - loss: 1.6249 - acc: 0.3800
    epochs : 42
    Epoch 1/1
    0s - loss: 2.0695 - acc: 0.2600
    epochs : 43
    Epoch 1/1
    0s - loss: 1.6133 - acc: 0.4000
    epochs : 44
    Epoch 1/1
    0s - loss: 1.6592 - acc: 0.4000
    epochs : 45
    Epoch 1/1
    0s - loss: 1.6300 - acc: 0.4000
    epochs : 46
    Epoch 1/1
    0s - loss: 1.5646 - acc: 0.4400
    epochs : 47
    Epoch 1/1
    0s - loss: 1.6283 - acc: 0.4000
    epochs : 48
    Epoch 1/1
    0s - loss: 1.6154 - acc: 0.4200
    epochs : 49
    Epoch 1/1
    0s - loss: 1.5378 - acc: 0.3800
    epochs : 50
    Epoch 1/1
    0s - loss: 1.5571 - acc: 0.4400
    epochs : 51
    Epoch 1/1
    0s - loss: 1.5601 - acc: 0.4000
    epochs : 52
    Epoch 1/1
    0s - loss: 1.4982 - acc: 0.4600
    epochs : 53
    Epoch 1/1
    0s - loss: 1.4341 - acc: 0.4800
    epochs : 54
    Epoch 1/1
    0s - loss: 1.4465 - acc: 0.4600
    epochs : 55
    Epoch 1/1
    0s - loss: 1.4704 - acc: 0.4600
    epochs : 56
    Epoch 1/1
    0s - loss: 1.5120 - acc: 0.4000
    epochs : 57
    Epoch 1/1
    0s - loss: 1.4041 - acc: 0.5400
    epochs : 58
    Epoch 1/1
    0s - loss: 1.3394 - acc: 0.4800
    epochs : 59
    Epoch 1/1
    0s - loss: 1.4483 - acc: 0.5200
    epochs : 60
    Epoch 1/1
    0s - loss: 1.5172 - acc: 0.4400
    epochs : 61
    Epoch 1/1
    0s - loss: 1.4404 - acc: 0.4200
    epochs : 62
    Epoch 1/1
    0s - loss: 1.3351 - acc: 0.5200
    epochs : 63
    Epoch 1/1
    0s - loss: 1.2374 - acc: 0.5600
    epochs : 64
    Epoch 1/1
    0s - loss: 1.4581 - acc: 0.4600
    epochs : 65
    Epoch 1/1
    0s - loss: 1.3446 - acc: 0.4600
    epochs : 66
    Epoch 1/1
    0s - loss: 1.2374 - acc: 0.6000
    epochs : 67
    Epoch 1/1
    0s - loss: 1.3145 - acc: 0.5000
    epochs : 68
    Epoch 1/1
    0s - loss: 1.3460 - acc: 0.4800
    epochs : 69
    Epoch 1/1
    0s - loss: 1.2119 - acc: 0.5200
    epochs : 70
    Epoch 1/1
    0s - loss: 1.2341 - acc: 0.5400
    epochs : 71
    Epoch 1/1
    0s - loss: 1.1911 - acc: 0.5600
    epochs : 72
    Epoch 1/1
    0s - loss: 1.2116 - acc: 0.5400
    epochs : 73
    Epoch 1/1
    0s - loss: 1.1436 - acc: 0.5800
    epochs : 74
    Epoch 1/1
    0s - loss: 1.1134 - acc: 0.5600
    epochs : 75
    Epoch 1/1
    0s - loss: 1.2878 - acc: 0.5400
    epochs : 76
    Epoch 1/1
    0s - loss: 1.1820 - acc: 0.5800
    epochs : 77
    Epoch 1/1
    0s - loss: 1.4045 - acc: 0.4600
    epochs : 78
    Epoch 1/1
    0s - loss: 1.2677 - acc: 0.5400
    epochs : 79
    Epoch 1/1
    0s - loss: 1.2084 - acc: 0.6000
    epochs : 80
    Epoch 1/1
    0s - loss: 1.1495 - acc: 0.6200
    epochs : 81
    Epoch 1/1
    0s - loss: 1.2684 - acc: 0.4800
    epochs : 82
    Epoch 1/1
    0s - loss: 1.1223 - acc: 0.5800
    epochs : 83
    Epoch 1/1
    0s - loss: 1.1465 - acc: 0.5800
    epochs : 84
    Epoch 1/1
    0s - loss: 1.2300 - acc: 0.5200
    epochs : 85
    Epoch 1/1
    0s - loss: 1.0301 - acc: 0.6200
    epochs : 86
    Epoch 1/1
    0s - loss: 1.1275 - acc: 0.6000
    epochs : 87
    Epoch 1/1
    0s - loss: 1.1203 - acc: 0.5600
    epochs : 88
    Epoch 1/1
    0s - loss: 1.0207 - acc: 0.6000
    epochs : 89
    Epoch 1/1
    0s - loss: 1.0210 - acc: 0.6000
    epochs : 90
    Epoch 1/1
    0s - loss: 0.9947 - acc: 0.6400
    epochs : 91
    Epoch 1/1
    0s - loss: 1.1896 - acc: 0.5600
    epochs : 92
    Epoch 1/1
    0s - loss: 0.9782 - acc: 0.6000
    epochs : 93
    Epoch 1/1
    0s - loss: 1.0428 - acc: 0.6200
    epochs : 94
    Epoch 1/1
    0s - loss: 1.0751 - acc: 0.6000
    epochs : 95
    Epoch 1/1
    0s - loss: 0.9361 - acc: 0.5400
    epochs : 96
    Epoch 1/1
    0s - loss: 0.8880 - acc: 0.7000
    epochs : 97
    Epoch 1/1
    0s - loss: 1.3380 - acc: 0.5000
    epochs : 98
    Epoch 1/1
    0s - loss: 1.2836 - acc: 0.5000
    epochs : 99
    Epoch 1/1
    0s - loss: 1.2418 - acc: 0.4800
    epochs : 100
    Epoch 1/1
    0s - loss: 1.0698 - acc: 0.5600
    epochs : 101
    Epoch 1/1
    0s - loss: 0.8374 - acc: 0.7000
    epochs : 102
    Epoch 1/1
    0s - loss: 0.7920 - acc: 0.7200
    epochs : 103
    Epoch 1/1
    0s - loss: 0.8410 - acc: 0.7200
    epochs : 104
    Epoch 1/1
    0s - loss: 1.0148 - acc: 0.5600
    epochs : 105
    Epoch 1/1
    0s - loss: 1.2032 - acc: 0.4800
    epochs : 106
    Epoch 1/1
    0s - loss: 0.9537 - acc: 0.6200
    epochs : 107
    Epoch 1/1
    0s - loss: 0.7960 - acc: 0.7200
    epochs : 108
    Epoch 1/1
    0s - loss: 1.0109 - acc: 0.6000
    epochs : 109
    Epoch 1/1
    0s - loss: 0.8325 - acc: 0.6800
    epochs : 110
    Epoch 1/1
    0s - loss: 0.8800 - acc: 0.6600
    epochs : 111
    Epoch 1/1
    0s - loss: 0.9389 - acc: 0.6000
    epochs : 112
    Epoch 1/1
    0s - loss: 1.2555 - acc: 0.5600
    epochs : 113
    Epoch 1/1
    0s - loss: 1.0369 - acc: 0.5600
    epochs : 114
    Epoch 1/1
    0s - loss: 1.0460 - acc: 0.5200
    epochs : 115
    Epoch 1/1
    0s - loss: 1.0479 - acc: 0.5800
    epochs : 116
    Epoch 1/1
    0s - loss: 0.9475 - acc: 0.6000
    epochs : 117
    Epoch 1/1
    0s - loss: 1.3904 - acc: 0.4600
    epochs : 118
    Epoch 1/1
    0s - loss: 0.9302 - acc: 0.7400
    epochs : 119
    Epoch 1/1
    0s - loss: 1.1820 - acc: 0.6000
    epochs : 120
    Epoch 1/1
    0s - loss: 0.9724 - acc: 0.6200
    epochs : 121
    Epoch 1/1
    0s - loss: 0.9418 - acc: 0.6200
    epochs : 122
    Epoch 1/1
    0s - loss: 0.7945 - acc: 0.7200
    epochs : 123
    Epoch 1/1
    0s - loss: 1.0654 - acc: 0.6400
    epochs : 124
    Epoch 1/1
    0s - loss: 0.9669 - acc: 0.5600
    epochs : 125
    Epoch 1/1
    0s - loss: 0.8725 - acc: 0.7200
    epochs : 126
    Epoch 1/1
    0s - loss: 0.6585 - acc: 0.7800
    epochs : 127
    Epoch 1/1
    0s - loss: 0.7537 - acc: 0.6600
    epochs : 128
    Epoch 1/1
    0s - loss: 1.0641 - acc: 0.5600
    epochs : 129
    Epoch 1/1
    0s - loss: 0.9098 - acc: 0.5800
    epochs : 130
    Epoch 1/1
    0s - loss: 0.6653 - acc: 0.7600
    epochs : 131
    Epoch 1/1
    0s - loss: 0.5854 - acc: 0.8000
    epochs : 132
    Epoch 1/1
    0s - loss: 1.4878 - acc: 0.4400
    epochs : 133
    Epoch 1/1
    0s - loss: 1.3193 - acc: 0.5200
    epochs : 134
    Epoch 1/1
    0s - loss: 1.0939 - acc: 0.5800
    epochs : 135
    Epoch 1/1
    0s - loss: 1.0989 - acc: 0.5800
    epochs : 136
    Epoch 1/1
    0s - loss: 0.8139 - acc: 0.6600
    epochs : 137
    Epoch 1/1
    0s - loss: 0.6621 - acc: 0.8400
    epochs : 138
    Epoch 1/1
    0s - loss: 0.7614 - acc: 0.7200
    epochs : 139
    Epoch 1/1
    0s - loss: 0.8862 - acc: 0.6200
    epochs : 140
    Epoch 1/1
    0s - loss: 0.7379 - acc: 0.7200
    epochs : 141
    Epoch 1/1
    0s - loss: 0.6328 - acc: 0.8400
    epochs : 142
    Epoch 1/1
    0s - loss: 0.8670 - acc: 0.6000
    epochs : 143
    Epoch 1/1
    0s - loss: 1.0915 - acc: 0.6200
    epochs : 144
    Epoch 1/1
    0s - loss: 0.6937 - acc: 0.7400
    epochs : 145
    Epoch 1/1
    0s - loss: 0.4308 - acc: 0.8800
    epochs : 146
    Epoch 1/1
    0s - loss: 0.4869 - acc: 0.8800
    epochs : 147
    Epoch 1/1
    0s - loss: 0.8522 - acc: 0.7000
    epochs : 148
    Epoch 1/1
    0s - loss: 0.7078 - acc: 0.8000
    epochs : 149
    Epoch 1/1
    0s - loss: 0.7842 - acc: 0.7000
    epochs : 150
    Epoch 1/1
    0s - loss: 0.8269 - acc: 0.6600
    epochs : 151
    Epoch 1/1
    0s - loss: 0.9440 - acc: 0.5800
    epochs : 152
    Epoch 1/1
    0s - loss: 1.1151 - acc: 0.6000
    epochs : 153
    Epoch 1/1
    0s - loss: 0.5674 - acc: 0.7800
    epochs : 154
    Epoch 1/1
    0s - loss: 0.5918 - acc: 0.8000
    epochs : 155
    Epoch 1/1
    0s - loss: 0.6360 - acc: 0.7200
    epochs : 156
    Epoch 1/1
    0s - loss: 0.5255 - acc: 0.8400
    epochs : 157
    Epoch 1/1
    0s - loss: 0.4626 - acc: 0.8200
    epochs : 158
    Epoch 1/1
    0s - loss: 0.6635 - acc: 0.7800
    epochs : 159
    Epoch 1/1
    0s - loss: 0.4487 - acc: 0.9000
    epochs : 160
    Epoch 1/1
    0s - loss: 0.6036 - acc: 0.8000
    epochs : 161
    Epoch 1/1
    0s - loss: 0.5740 - acc: 0.7800
    epochs : 162
    Epoch 1/1
    0s - loss: 0.6661 - acc: 0.7200
    epochs : 163
    Epoch 1/1
    0s - loss: 0.3185 - acc: 0.9400
    epochs : 164
    Epoch 1/1
    0s - loss: 0.2383 - acc: 1.0000
    epochs : 165
    Epoch 1/1
    0s - loss: 0.2035 - acc: 0.9600
    epochs : 166
    Epoch 1/1
    0s - loss: 0.1461 - acc: 0.9800
    epochs : 167
    Epoch 1/1
    0s - loss: 0.1440 - acc: 0.9800
    epochs : 168
    Epoch 1/1
    0s - loss: 0.1925 - acc: 0.9600
    epochs : 169
    Epoch 1/1
    0s - loss: 0.2436 - acc: 0.9200
    epochs : 170
    Epoch 1/1
    0s - loss: 0.5228 - acc: 0.8200
    epochs : 171
    Epoch 1/1
    0s - loss: 0.4240 - acc: 0.8800
    epochs : 172
    Epoch 1/1
    0s - loss: 0.3760 - acc: 0.8600
    epochs : 173
    Epoch 1/1
    0s - loss: 0.4261 - acc: 0.8800
    epochs : 174
    Epoch 1/1
    0s - loss: 0.9488 - acc: 0.6000
    epochs : 175
    Epoch 1/1
    0s - loss: 1.1243 - acc: 0.6400
    epochs : 176
    Epoch 1/1
    0s - loss: 1.8498 - acc: 0.3800
    epochs : 177
    Epoch 1/1
    0s - loss: 1.0756 - acc: 0.6000
    epochs : 178
    Epoch 1/1
    0s - loss: 0.8424 - acc: 0.7000
    epochs : 179
    Epoch 1/1
    0s - loss: 0.8562 - acc: 0.6200
    epochs : 180
    Epoch 1/1
    0s - loss: 0.6869 - acc: 0.7400
    epochs : 181
    Epoch 1/1
    0s - loss: 0.8557 - acc: 0.6000
    epochs : 182
    Epoch 1/1
    0s - loss: 0.3628 - acc: 0.9000
    epochs : 183
    Epoch 1/1
    0s - loss: 0.3253 - acc: 0.9400
    epochs : 184
    Epoch 1/1
    0s - loss: 0.2061 - acc: 0.9600
    epochs : 185
    Epoch 1/1
    0s - loss: 0.1690 - acc: 0.9800
    epochs : 186
    Epoch 1/1
    0s - loss: 0.1426 - acc: 1.0000
    epochs : 187
    Epoch 1/1
    0s - loss: 0.1081 - acc: 1.0000
    epochs : 188
    Epoch 1/1
    0s - loss: 0.0958 - acc: 1.0000
    epochs : 189
    Epoch 1/1
    0s - loss: 0.0623 - acc: 1.0000
    epochs : 190
    Epoch 1/1
    0s - loss: 0.0593 - acc: 1.0000
    epochs : 191
    Epoch 1/1
    0s - loss: 0.0411 - acc: 1.0000
    epochs : 192
    Epoch 1/1
    0s - loss: 0.0343 - acc: 1.0000
    epochs : 193
    Epoch 1/1
    0s - loss: 0.0288 - acc: 1.0000
    epochs : 194
    Epoch 1/1
    0s - loss: 0.0259 - acc: 1.0000
    epochs : 195
    Epoch 1/1
    0s - loss: 0.0243 - acc: 1.0000
    epochs : 196
    Epoch 1/1
    0s - loss: 0.0208 - acc: 1.0000
    epochs : 197
    Epoch 1/1
    0s - loss: 0.0197 - acc: 1.0000
    epochs : 198
    Epoch 1/1
    0s - loss: 0.0176 - acc: 1.0000
    epochs : 199
    Epoch 1/1
    0s - loss: 0.0169 - acc: 1.0000
    epochs : 200
    Epoch 1/1
    0s - loss: 0.0147 - acc: 1.0000
    epochs : 201
    Epoch 1/1
    0s - loss: 0.0190 - acc: 1.0000
    epochs : 202
    Epoch 1/1
    0s - loss: 0.0502 - acc: 1.0000
    epochs : 203
    Epoch 1/1
    0s - loss: 0.6839 - acc: 0.8000
    epochs : 204
    Epoch 1/1
    0s - loss: 1.6899 - acc: 0.6200
    epochs : 205
    Epoch 1/1
    0s - loss: 1.5993 - acc: 0.4800
    epochs : 206
    Epoch 1/1
    0s - loss: 0.8757 - acc: 0.5800
    epochs : 207
    Epoch 1/1
    0s - loss: 1.0933 - acc: 0.7200
    epochs : 208
    Epoch 1/1
    0s - loss: 0.5000 - acc: 0.8000
    epochs : 209
    Epoch 1/1
    0s - loss: 0.5947 - acc: 0.8400
    epochs : 210
    Epoch 1/1
    0s - loss: 0.4250 - acc: 0.8800
    epochs : 211
    Epoch 1/1
    0s - loss: 0.2800 - acc: 0.9400
    epochs : 212
    Epoch 1/1
    0s - loss: 0.2680 - acc: 0.9000
    epochs : 213
    Epoch 1/1
    0s - loss: 0.2044 - acc: 0.9600
    epochs : 214
    Epoch 1/1
    0s - loss: 0.0864 - acc: 1.0000
    epochs : 215
    Epoch 1/1
    0s - loss: 0.0801 - acc: 1.0000
    epochs : 216
    Epoch 1/1
    0s - loss: 0.1282 - acc: 0.9800
    epochs : 217
    Epoch 1/1
    0s - loss: 0.8209 - acc: 0.7200
    epochs : 218
    Epoch 1/1
    0s - loss: 1.5435 - acc: 0.5400
    epochs : 219
    Epoch 1/1
    0s - loss: 1.1389 - acc: 0.4400
    epochs : 220
    Epoch 1/1
    0s - loss: 0.8027 - acc: 0.6800
    epochs : 221
    Epoch 1/1
    0s - loss: 0.7829 - acc: 0.7200
    epochs : 222
    Epoch 1/1
    0s - loss: 0.5679 - acc: 0.7600
    epochs : 223
    Epoch 1/1
    0s - loss: 0.4555 - acc: 0.8000
    epochs : 224
    Epoch 1/1
    0s - loss: 0.4595 - acc: 0.7600
    epochs : 225
    Epoch 1/1
    0s - loss: 0.3121 - acc: 0.8400
    epochs : 226
    Epoch 1/1
    0s - loss: 0.3122 - acc: 0.8600
    epochs : 227
    Epoch 1/1
    0s - loss: 0.3453 - acc: 0.9200
    epochs : 228
    Epoch 1/1
    0s - loss: 0.3412 - acc: 0.9000
    epochs : 229
    Epoch 1/1
    0s - loss: 0.1507 - acc: 0.9600
    epochs : 230
    Epoch 1/1
    0s - loss: 0.1691 - acc: 0.9600
    epochs : 231
    Epoch 1/1
    0s - loss: 0.0652 - acc: 1.0000
    epochs : 232
    Epoch 1/1
    0s - loss: 0.0459 - acc: 1.0000
    epochs : 233
    Epoch 1/1
    0s - loss: 0.0441 - acc: 1.0000
    epochs : 234
    Epoch 1/1
    0s - loss: 0.0343 - acc: 1.0000
    epochs : 235
    Epoch 1/1
    0s - loss: 0.0279 - acc: 1.0000
    epochs : 236
    Epoch 1/1
    0s - loss: 0.0237 - acc: 1.0000
    epochs : 237
    Epoch 1/1
    0s - loss: 0.0209 - acc: 1.0000
    epochs : 238
    Epoch 1/1
    0s - loss: 0.0189 - acc: 1.0000
    epochs : 239
    Epoch 1/1
    0s - loss: 0.0172 - acc: 1.0000
    epochs : 240
    Epoch 1/1
    0s - loss: 0.0157 - acc: 1.0000
    epochs : 241
    Epoch 1/1
    0s - loss: 0.0144 - acc: 1.0000
    epochs : 242
    Epoch 1/1
    0s - loss: 0.0132 - acc: 1.0000
    epochs : 243
    Epoch 1/1
    0s - loss: 0.0123 - acc: 1.0000
    epochs : 244
    Epoch 1/1
    0s - loss: 0.0116 - acc: 1.0000
    epochs : 245
    Epoch 1/1
    0s - loss: 0.0108 - acc: 1.0000
    epochs : 246
    Epoch 1/1
    0s - loss: 0.0101 - acc: 1.0000
    epochs : 247
    Epoch 1/1
    0s - loss: 0.0095 - acc: 1.0000
    epochs : 248
    Epoch 1/1
    0s - loss: 0.0090 - acc: 1.0000
    epochs : 249
    Epoch 1/1
    0s - loss: 0.0085 - acc: 1.0000
    epochs : 250
    Epoch 1/1
    0s - loss: 0.0081 - acc: 1.0000
    epochs : 251
    Epoch 1/1
    0s - loss: 0.0077 - acc: 1.0000
    epochs : 252
    Epoch 1/1
    0s - loss: 0.0073 - acc: 1.0000
    epochs : 253
    Epoch 1/1
    0s - loss: 0.0069 - acc: 1.0000
    epochs : 254
    Epoch 1/1
    0s - loss: 0.0066 - acc: 1.0000
    epochs : 255
    Epoch 1/1
    0s - loss: 0.0063 - acc: 1.0000
    epochs : 256
    Epoch 1/1
    0s - loss: 0.0060 - acc: 1.0000
    epochs : 257
    Epoch 1/1
    0s - loss: 0.0057 - acc: 1.0000
    epochs : 258
    Epoch 1/1
    0s - loss: 0.0054 - acc: 1.0000
    epochs : 259
    Epoch 1/1
    0s - loss: 0.0052 - acc: 1.0000
    epochs : 260
    Epoch 1/1
    0s - loss: 0.0049 - acc: 1.0000
    epochs : 261
    Epoch 1/1
    0s - loss: 0.0047 - acc: 1.0000
    epochs : 262
    Epoch 1/1
    0s - loss: 0.0045 - acc: 1.0000
    epochs : 263
    Epoch 1/1
    0s - loss: 0.0043 - acc: 1.0000
    epochs : 264
    Epoch 1/1
    0s - loss: 0.0041 - acc: 1.0000
    epochs : 265
    Epoch 1/1
    0s - loss: 0.0040 - acc: 1.0000
    epochs : 266
    Epoch 1/1
    0s - loss: 0.0038 - acc: 1.0000
    epochs : 267
    Epoch 1/1
    0s - loss: 0.0037 - acc: 1.0000
    epochs : 268
    Epoch 1/1
    0s - loss: 0.0035 - acc: 1.0000
    epochs : 269
    Epoch 1/1
    0s - loss: 0.0034 - acc: 1.0000
    epochs : 270
    Epoch 1/1
    0s - loss: 0.0032 - acc: 1.0000
    epochs : 271
    Epoch 1/1
    0s - loss: 0.0032 - acc: 1.0000
    epochs : 272
    Epoch 1/1
    0s - loss: 0.0030 - acc: 1.0000
    epochs : 273
    Epoch 1/1
    0s - loss: 0.0029 - acc: 1.0000
    epochs : 274
    Epoch 1/1
    0s - loss: 0.0028 - acc: 1.0000
    epochs : 275
    Epoch 1/1
    0s - loss: 0.0027 - acc: 1.0000
    epochs : 276
    Epoch 1/1
    0s - loss: 0.0026 - acc: 1.0000
    epochs : 277
    Epoch 1/1
    0s - loss: 0.0025 - acc: 1.0000
    epochs : 278
    Epoch 1/1
    0s - loss: 0.0024 - acc: 1.0000
    epochs : 279
    Epoch 1/1
    0s - loss: 0.0024 - acc: 1.0000
    epochs : 280
    Epoch 1/1
    0s - loss: 0.0023 - acc: 1.0000
    epochs : 281
    Epoch 1/1
    0s - loss: 0.0022 - acc: 1.0000
    epochs : 282
    Epoch 1/1
    0s - loss: 0.0021 - acc: 1.0000
    epochs : 283
    Epoch 1/1
    0s - loss: 0.0021 - acc: 1.0000
    epochs : 284
    Epoch 1/1
    0s - loss: 0.0020 - acc: 1.0000
    epochs : 285
    Epoch 1/1
    0s - loss: 0.0019 - acc: 1.0000
    epochs : 286
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 287
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 288
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 289
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 290
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 291
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 292
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 293
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 294
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 295
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 296
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 297
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 298
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 299
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 300
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 301
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 302
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 303
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 304
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 305
    Epoch 1/1
    0s - loss: 0.0010 - acc: 1.0000
    epochs : 306
    Epoch 1/1
    0s - loss: 9.9271e-04 - acc: 1.0000
    epochs : 307
    Epoch 1/1
    0s - loss: 9.7596e-04 - acc: 1.0000
    epochs : 308
    Epoch 1/1
    0s - loss: 9.4267e-04 - acc: 1.0000
    epochs : 309
    Epoch 1/1
    0s - loss: 9.1432e-04 - acc: 1.0000
    epochs : 310
    Epoch 1/1
    0s - loss: 8.7729e-04 - acc: 1.0000
    epochs : 311
    Epoch 1/1
    0s - loss: 8.5034e-04 - acc: 1.0000
    epochs : 312
    Epoch 1/1
    0s - loss: 8.2925e-04 - acc: 1.0000
    epochs : 313
    Epoch 1/1
    0s - loss: 8.1086e-04 - acc: 1.0000
    epochs : 314
    Epoch 1/1
    0s - loss: 7.8009e-04 - acc: 1.0000
    epochs : 315
    Epoch 1/1
    0s - loss: 7.6433e-04 - acc: 1.0000
    epochs : 316
    Epoch 1/1
    0s - loss: 7.4477e-04 - acc: 1.0000
    epochs : 317
    Epoch 1/1
    0s - loss: 7.2169e-04 - acc: 1.0000
    epochs : 318
    Epoch 1/1
    0s - loss: 7.0017e-04 - acc: 1.0000
    epochs : 319
    Epoch 1/1
    0s - loss: 6.7678e-04 - acc: 1.0000
    epochs : 320
    Epoch 1/1
    0s - loss: 6.6442e-04 - acc: 1.0000
    epochs : 321
    Epoch 1/1
    0s - loss: 6.4031e-04 - acc: 1.0000
    epochs : 322
    Epoch 1/1
    0s - loss: 6.2756e-04 - acc: 1.0000
    epochs : 323
    Epoch 1/1
    0s - loss: 6.1067e-04 - acc: 1.0000
    epochs : 324
    Epoch 1/1
    0s - loss: 5.9065e-04 - acc: 1.0000
    epochs : 325
    Epoch 1/1
    0s - loss: 5.7282e-04 - acc: 1.0000
    epochs : 326
    Epoch 1/1
    0s - loss: 5.5457e-04 - acc: 1.0000
    epochs : 327
    Epoch 1/1
    0s - loss: 5.3790e-04 - acc: 1.0000
    epochs : 328
    Epoch 1/1
    0s - loss: 5.2191e-04 - acc: 1.0000
    epochs : 329
    Epoch 1/1
    0s - loss: 5.0438e-04 - acc: 1.0000
    epochs : 330
    Epoch 1/1
    0s - loss: 4.8857e-04 - acc: 1.0000
    epochs : 331
    Epoch 1/1
    0s - loss: 4.7258e-04 - acc: 1.0000
    epochs : 332
    Epoch 1/1
    0s - loss: 4.5932e-04 - acc: 1.0000
    epochs : 333
    Epoch 1/1
    0s - loss: 4.4624e-04 - acc: 1.0000
    epochs : 334
    Epoch 1/1
    0s - loss: 4.3248e-04 - acc: 1.0000
    epochs : 335
    Epoch 1/1
    0s - loss: 4.1782e-04 - acc: 1.0000
    epochs : 336
    Epoch 1/1
    0s - loss: 4.0847e-04 - acc: 1.0000
    epochs : 337
    Epoch 1/1
    0s - loss: 3.9278e-04 - acc: 1.0000
    epochs : 338
    Epoch 1/1
    0s - loss: 3.8338e-04 - acc: 1.0000
    epochs : 339
    Epoch 1/1
    0s - loss: 3.7131e-04 - acc: 1.0000
    epochs : 340
    Epoch 1/1
    0s - loss: 3.6118e-04 - acc: 1.0000
    epochs : 341
    Epoch 1/1
    0s - loss: 3.5068e-04 - acc: 1.0000
    epochs : 342
    Epoch 1/1
    0s - loss: 3.4107e-04 - acc: 1.0000
    epochs : 343
    Epoch 1/1
    0s - loss: 3.3152e-04 - acc: 1.0000
    epochs : 344
    Epoch 1/1
    0s - loss: 3.2179e-04 - acc: 1.0000
    epochs : 345
    Epoch 1/1
    0s - loss: 3.1297e-04 - acc: 1.0000
    epochs : 346
    Epoch 1/1
    0s - loss: 3.0472e-04 - acc: 1.0000
    epochs : 347
    Epoch 1/1
    0s - loss: 2.9516e-04 - acc: 1.0000
    epochs : 348
    Epoch 1/1
    0s - loss: 2.8798e-04 - acc: 1.0000
    epochs : 349
    Epoch 1/1
    0s - loss: 2.7858e-04 - acc: 1.0000
    epochs : 350
    Epoch 1/1
    0s - loss: 2.7325e-04 - acc: 1.0000
    epochs : 351
    Epoch 1/1
    0s - loss: 2.6259e-04 - acc: 1.0000
    epochs : 352
    Epoch 1/1
    0s - loss: 2.5702e-04 - acc: 1.0000
    epochs : 353
    Epoch 1/1
    0s - loss: 2.4699e-04 - acc: 1.0000
    epochs : 354
    Epoch 1/1
    0s - loss: 2.4194e-04 - acc: 1.0000
    epochs : 355
    Epoch 1/1
    0s - loss: 2.3242e-04 - acc: 1.0000
    epochs : 356
    Epoch 1/1
    0s - loss: 2.2790e-04 - acc: 1.0000
    epochs : 357
    Epoch 1/1
    0s - loss: 2.1887e-04 - acc: 1.0000
    epochs : 358
    Epoch 1/1
    0s - loss: 2.1466e-04 - acc: 1.0000
    epochs : 359
    Epoch 1/1
    0s - loss: 2.0699e-04 - acc: 1.0000
    epochs : 360
    Epoch 1/1
    0s - loss: 2.0151e-04 - acc: 1.0000
    epochs : 361
    Epoch 1/1
    0s - loss: 1.9475e-04 - acc: 1.0000
    epochs : 362
    Epoch 1/1
    0s - loss: 1.8994e-04 - acc: 1.0000
    epochs : 363
    Epoch 1/1
    0s - loss: 1.8358e-04 - acc: 1.0000
    epochs : 364
    Epoch 1/1
    0s - loss: 1.7968e-04 - acc: 1.0000
    epochs : 365
    Epoch 1/1
    0s - loss: 1.7377e-04 - acc: 1.0000
    epochs : 366
    Epoch 1/1
    0s - loss: 1.6892e-04 - acc: 1.0000
    epochs : 367
    Epoch 1/1
    0s - loss: 1.6502e-04 - acc: 1.0000
    epochs : 368
    Epoch 1/1
    0s - loss: 1.5925e-04 - acc: 1.0000
    epochs : 369
    Epoch 1/1
    0s - loss: 1.5637e-04 - acc: 1.0000
    epochs : 370
    Epoch 1/1
    0s - loss: 1.5097e-04 - acc: 1.0000
    epochs : 371
    Epoch 1/1
    0s - loss: 1.4709e-04 - acc: 1.0000
    epochs : 372
    Epoch 1/1
    0s - loss: 1.4342e-04 - acc: 1.0000
    epochs : 373
    Epoch 1/1
    0s - loss: 1.3889e-04 - acc: 1.0000
    epochs : 374
    Epoch 1/1
    0s - loss: 1.3677e-04 - acc: 1.0000
    epochs : 375
    Epoch 1/1
    0s - loss: 1.3188e-04 - acc: 1.0000
    epochs : 376
    Epoch 1/1
    0s - loss: 1.2964e-04 - acc: 1.0000
    epochs : 377
    Epoch 1/1
    0s - loss: 1.2456e-04 - acc: 1.0000
    epochs : 378
    Epoch 1/1
    0s - loss: 1.2191e-04 - acc: 1.0000
    epochs : 379
    Epoch 1/1
    0s - loss: 1.1762e-04 - acc: 1.0000
    epochs : 380
    Epoch 1/1
    0s - loss: 1.1555e-04 - acc: 1.0000
    epochs : 381
    Epoch 1/1
    0s - loss: 1.1526e-04 - acc: 1.0000
    epochs : 382
    Epoch 1/1
    0s - loss: 1.1287e-04 - acc: 1.0000
    epochs : 383
    Epoch 1/1
    0s - loss: 1.1194e-04 - acc: 1.0000
    epochs : 384
    Epoch 1/1
    0s - loss: 1.0882e-04 - acc: 1.0000
    epochs : 385
    Epoch 1/1
    0s - loss: 1.0760e-04 - acc: 1.0000
    epochs : 386
    Epoch 1/1
    0s - loss: 1.0495e-04 - acc: 1.0000
    epochs : 387
    Epoch 1/1
    0s - loss: 1.0209e-04 - acc: 1.0000
    epochs : 388
    Epoch 1/1
    0s - loss: 9.9721e-05 - acc: 1.0000
    epochs : 389
    Epoch 1/1
    0s - loss: 9.5490e-05 - acc: 1.0000
    epochs : 390
    Epoch 1/1
    0s - loss: 9.0264e-05 - acc: 1.0000
    epochs : 391
    Epoch 1/1
    0s - loss: 8.6603e-05 - acc: 1.0000
    epochs : 392
    Epoch 1/1
    0s - loss: 8.3500e-05 - acc: 1.0000
    epochs : 393
    Epoch 1/1
    0s - loss: 8.1627e-05 - acc: 1.0000
    epochs : 394
    Epoch 1/1
    0s - loss: 8.0136e-05 - acc: 1.0000
    epochs : 395
    Epoch 1/1
    0s - loss: 7.8138e-05 - acc: 1.0000
    epochs : 396
    Epoch 1/1
    0s - loss: 7.6059e-05 - acc: 1.0000
    epochs : 397
    Epoch 1/1
    0s - loss: 7.3845e-05 - acc: 1.0000
    epochs : 398
    Epoch 1/1
    0s - loss: 7.1751e-05 - acc: 1.0000
    epochs : 399
    Epoch 1/1
    0s - loss: 6.9777e-05 - acc: 1.0000
    epochs : 400
    Epoch 1/1
    0s - loss: 6.7586e-05 - acc: 1.0000
    epochs : 401
    Epoch 1/1
    0s - loss: 6.5572e-05 - acc: 1.0000
    epochs : 402
    Epoch 1/1
    0s - loss: 6.3527e-05 - acc: 1.0000
    epochs : 403
    Epoch 1/1
    0s - loss: 6.1841e-05 - acc: 1.0000
    epochs : 404
    Epoch 1/1
    0s - loss: 6.0112e-05 - acc: 1.0000
    epochs : 405
    Epoch 1/1
    0s - loss: 5.8557e-05 - acc: 1.0000
    epochs : 406
    Epoch 1/1
    0s - loss: 5.6770e-05 - acc: 1.0000
    epochs : 407
    Epoch 1/1
    0s - loss: 5.5060e-05 - acc: 1.0000
    epochs : 408
    Epoch 1/1
    0s - loss: 5.3522e-05 - acc: 1.0000
    epochs : 409
    Epoch 1/1
    0s - loss: 5.1983e-05 - acc: 1.0000
    epochs : 410
    Epoch 1/1
    0s - loss: 5.0417e-05 - acc: 1.0000
    epochs : 411
    Epoch 1/1
    0s - loss: 4.8837e-05 - acc: 1.0000
    epochs : 412
    Epoch 1/1
    0s - loss: 4.7696e-05 - acc: 1.0000
    epochs : 413
    Epoch 1/1
    0s - loss: 4.6388e-05 - acc: 1.0000
    epochs : 414
    Epoch 1/1
    0s - loss: 4.5435e-05 - acc: 1.0000
    epochs : 415
    Epoch 1/1
    0s - loss: 4.3882e-05 - acc: 1.0000
    epochs : 416
    Epoch 1/1
    0s - loss: 4.2637e-05 - acc: 1.0000
    epochs : 417
    Epoch 1/1
    0s - loss: 4.1791e-05 - acc: 1.0000
    epochs : 418
    Epoch 1/1
    0s - loss: 4.0380e-05 - acc: 1.0000
    epochs : 419
    Epoch 1/1
    0s - loss: 3.9527e-05 - acc: 1.0000
    epochs : 420
    Epoch 1/1
    0s - loss: 3.8078e-05 - acc: 1.0000
    epochs : 421
    Epoch 1/1
    0s - loss: 3.7480e-05 - acc: 1.0000
    epochs : 422
    Epoch 1/1
    0s - loss: 3.6280e-05 - acc: 1.0000
    epochs : 423
    Epoch 1/1
    0s - loss: 3.5656e-05 - acc: 1.0000
    epochs : 424
    Epoch 1/1
    0s - loss: 3.4337e-05 - acc: 1.0000
    epochs : 425
    Epoch 1/1
    0s - loss: 3.3958e-05 - acc: 1.0000
    epochs : 426
    Epoch 1/1
    0s - loss: 3.2275e-05 - acc: 1.0000
    epochs : 427
    Epoch 1/1
    0s - loss: 3.2378e-05 - acc: 1.0000
    epochs : 428
    Epoch 1/1
    0s - loss: 3.0863e-05 - acc: 1.0000
    epochs : 429
    Epoch 1/1
    0s - loss: 2.9803e-05 - acc: 1.0000
    epochs : 430
    Epoch 1/1
    0s - loss: 2.9611e-05 - acc: 1.0000
    epochs : 431
    Epoch 1/1
    0s - loss: 2.8653e-05 - acc: 1.0000
    epochs : 432
    Epoch 1/1
    0s - loss: 3.0654e-05 - acc: 1.0000
    epochs : 433
    Epoch 1/1
    0s - loss: 4.3429e-05 - acc: 1.0000
    epochs : 434
    Epoch 1/1
    0s - loss: 1.5979e-04 - acc: 1.0000
    epochs : 435
    Epoch 1/1
    0s - loss: 1.8255 - acc: 0.7600
    epochs : 436
    Epoch 1/1
    0s - loss: 2.6301 - acc: 0.2800
    epochs : 437
    Epoch 1/1
    0s - loss: 2.0289 - acc: 0.3000
    epochs : 438
    Epoch 1/1
    0s - loss: 1.7491 - acc: 0.4200
    epochs : 439
    Epoch 1/1
    0s - loss: 1.5782 - acc: 0.4800
    epochs : 440
    Epoch 1/1
    0s - loss: 1.4806 - acc: 0.4200
    epochs : 441
    Epoch 1/1
    0s - loss: 1.3811 - acc: 0.4600
    epochs : 442
    Epoch 1/1
    0s - loss: 1.3656 - acc: 0.4800
    epochs : 443
    Epoch 1/1
    0s - loss: 1.3263 - acc: 0.4600
    epochs : 444
    Epoch 1/1
    0s - loss: 1.3068 - acc: 0.4400
    epochs : 445
    Epoch 1/1
    0s - loss: 1.2417 - acc: 0.4800
    epochs : 446
    Epoch 1/1
    0s - loss: 1.2043 - acc: 0.4800
    epochs : 447
    Epoch 1/1
    0s - loss: 1.1662 - acc: 0.5400
    epochs : 448
    Epoch 1/1
    0s - loss: 1.1522 - acc: 0.5000
    epochs : 449
    Epoch 1/1
    0s - loss: 1.1476 - acc: 0.5000
    epochs : 450
    Epoch 1/1
    0s - loss: 1.1454 - acc: 0.5200
    epochs : 451
    Epoch 1/1
    0s - loss: 1.0991 - acc: 0.5400
    epochs : 452
    Epoch 1/1
    0s - loss: 1.0455 - acc: 0.5400
    epochs : 453
    Epoch 1/1
    0s - loss: 1.0130 - acc: 0.5600
    epochs : 454
    Epoch 1/1
    0s - loss: 1.0278 - acc: 0.5800
    epochs : 455
    Epoch 1/1
    0s - loss: 1.0178 - acc: 0.5800
    epochs : 456
    Epoch 1/1
    0s - loss: 1.0184 - acc: 0.5600
    epochs : 457
    Epoch 1/1
    0s - loss: 0.9438 - acc: 0.6200
    epochs : 458
    Epoch 1/1
    0s - loss: 0.9081 - acc: 0.6400
    epochs : 459
    Epoch 1/1
    0s - loss: 0.9123 - acc: 0.5800
    epochs : 460
    Epoch 1/1
    0s - loss: 0.9040 - acc: 0.6200
    epochs : 461
    Epoch 1/1
    0s - loss: 0.9031 - acc: 0.6400
    epochs : 462
    Epoch 1/1
    0s - loss: 1.2169 - acc: 0.5600
    epochs : 463
    Epoch 1/1
    0s - loss: 1.1292 - acc: 0.5800
    epochs : 464
    Epoch 1/1
    0s - loss: 1.0175 - acc: 0.6000
    epochs : 465
    Epoch 1/1
    0s - loss: 0.9882 - acc: 0.5600
    epochs : 466
    Epoch 1/1
    0s - loss: 0.8317 - acc: 0.6600
    epochs : 467
    Epoch 1/1
    0s - loss: 1.1298 - acc: 0.5400
    epochs : 468
    Epoch 1/1
    0s - loss: 1.0381 - acc: 0.5800
    epochs : 469
    Epoch 1/1
    0s - loss: 0.8656 - acc: 0.6200
    epochs : 470
    Epoch 1/1
    0s - loss: 0.8511 - acc: 0.6400
    epochs : 471
    Epoch 1/1
    0s - loss: 0.8140 - acc: 0.7200
    epochs : 472
    Epoch 1/1
    0s - loss: 0.7991 - acc: 0.6200
    epochs : 473
    Epoch 1/1
    0s - loss: 0.9040 - acc: 0.5400
    epochs : 474
    Epoch 1/1
    0s - loss: 0.8163 - acc: 0.6600
    epochs : 475
    Epoch 1/1
    0s - loss: 0.7562 - acc: 0.6600
    epochs : 476
    Epoch 1/1
    0s - loss: 0.8686 - acc: 0.6400
    epochs : 477
    Epoch 1/1
    0s - loss: 0.7541 - acc: 0.7000
    epochs : 478
    Epoch 1/1
    0s - loss: 0.8331 - acc: 0.6400
    epochs : 479
    Epoch 1/1
    0s - loss: 0.8976 - acc: 0.6800
    epochs : 480
    Epoch 1/1
    0s - loss: 0.8776 - acc: 0.6400
    epochs : 481
    Epoch 1/1
    0s - loss: 0.8284 - acc: 0.6200
    epochs : 482
    Epoch 1/1
    0s - loss: 0.8826 - acc: 0.6400
    epochs : 483
    Epoch 1/1
    0s - loss: 0.8612 - acc: 0.6200
    epochs : 484
    Epoch 1/1
    0s - loss: 1.0908 - acc: 0.6400
    epochs : 485
    Epoch 1/1
    0s - loss: 0.9170 - acc: 0.6000
    epochs : 486
    Epoch 1/1
    0s - loss: 0.7973 - acc: 0.6400
    epochs : 487
    Epoch 1/1
    0s - loss: 0.7711 - acc: 0.6800
    epochs : 488
    Epoch 1/1
    0s - loss: 0.8058 - acc: 0.6800
    epochs : 489
    Epoch 1/1
    0s - loss: 1.1156 - acc: 0.5800
    epochs : 490
    Epoch 1/1
    0s - loss: 0.7707 - acc: 0.6400
    epochs : 491
    Epoch 1/1
    0s - loss: 1.0215 - acc: 0.6000
    epochs : 492
    Epoch 1/1
    0s - loss: 0.8603 - acc: 0.6600
    epochs : 493
    Epoch 1/1
    0s - loss: 0.8181 - acc: 0.6600
    epochs : 494
    Epoch 1/1
    0s - loss: 0.8004 - acc: 0.6600
    epochs : 495
    Epoch 1/1
    0s - loss: 0.6375 - acc: 0.7600
    epochs : 496
    Epoch 1/1
    0s - loss: 0.8720 - acc: 0.5800
    epochs : 497
    Epoch 1/1
    0s - loss: 0.8583 - acc: 0.6600
    epochs : 498
    Epoch 1/1
    0s - loss: 0.9681 - acc: 0.5600
    epochs : 499
    Epoch 1/1
    0s - loss: 0.6908 - acc: 0.7200
    epochs : 500
    Epoch 1/1
    0s - loss: 0.6693 - acc: 0.7600
    epochs : 501
    Epoch 1/1
    0s - loss: 0.6275 - acc: 0.7400
    epochs : 502
    Epoch 1/1
    0s - loss: 0.6065 - acc: 0.7400
    epochs : 503
    Epoch 1/1
    0s - loss: 0.5331 - acc: 0.7600
    epochs : 504
    Epoch 1/1
    0s - loss: 0.6632 - acc: 0.7600
    epochs : 505
    Epoch 1/1
    0s - loss: 0.8938 - acc: 0.6400
    epochs : 506
    Epoch 1/1
    0s - loss: 0.6025 - acc: 0.7800
    epochs : 507
    Epoch 1/1
    0s - loss: 0.5874 - acc: 0.7600
    epochs : 508
    Epoch 1/1
    0s - loss: 0.5201 - acc: 0.7800
    epochs : 509
    Epoch 1/1
    0s - loss: 0.6311 - acc: 0.7200
    epochs : 510
    Epoch 1/1
    0s - loss: 0.6721 - acc: 0.7200
    epochs : 511
    Epoch 1/1
    0s - loss: 0.7598 - acc: 0.7600
    epochs : 512
    Epoch 1/1
    0s - loss: 0.4609 - acc: 0.8800
    epochs : 513
    Epoch 1/1
    0s - loss: 0.5260 - acc: 0.7600
    epochs : 514
    Epoch 1/1
    0s - loss: 0.6688 - acc: 0.7600
    epochs : 515
    Epoch 1/1
    0s - loss: 1.0440 - acc: 0.6600
    epochs : 516
    Epoch 1/1
    0s - loss: 0.6633 - acc: 0.8000
    epochs : 517
    Epoch 1/1
    0s - loss: 0.6094 - acc: 0.7600
    epochs : 518
    Epoch 1/1
    0s - loss: 0.4405 - acc: 0.8000
    epochs : 519
    Epoch 1/1
    0s - loss: 0.3938 - acc: 0.8000
    epochs : 520
    Epoch 1/1
    0s - loss: 0.3888 - acc: 0.8000
    epochs : 521
    Epoch 1/1
    0s - loss: 0.7401 - acc: 0.7600
    epochs : 522
    Epoch 1/1
    0s - loss: 1.0862 - acc: 0.6600
    epochs : 523
    Epoch 1/1
    0s - loss: 1.2623 - acc: 0.5400
    epochs : 524
    Epoch 1/1
    0s - loss: 0.8384 - acc: 0.6200
    epochs : 525
    Epoch 1/1
    0s - loss: 0.6399 - acc: 0.7800
    epochs : 526
    Epoch 1/1
    0s - loss: 0.6416 - acc: 0.7400
    epochs : 527
    Epoch 1/1
    0s - loss: 0.7969 - acc: 0.7600
    epochs : 528
    Epoch 1/1
    0s - loss: 0.6428 - acc: 0.7000
    epochs : 529
    Epoch 1/1
    0s - loss: 0.8732 - acc: 0.6200
    epochs : 530
    Epoch 1/1
    0s - loss: 0.6947 - acc: 0.6800
    epochs : 531
    Epoch 1/1
    0s - loss: 0.7533 - acc: 0.7000
    epochs : 532
    Epoch 1/1
    0s - loss: 0.4362 - acc: 0.8600
    epochs : 533
    Epoch 1/1
    0s - loss: 0.5769 - acc: 0.7600
    epochs : 534
    Epoch 1/1
    0s - loss: 0.3255 - acc: 0.9400
    epochs : 535
    Epoch 1/1
    0s - loss: 0.2485 - acc: 0.9800
    epochs : 536
    Epoch 1/1
    0s - loss: 0.2528 - acc: 0.9200
    epochs : 537
    Epoch 1/1
    0s - loss: 0.3131 - acc: 0.9200
    epochs : 538
    Epoch 1/1
    0s - loss: 0.7263 - acc: 0.6600
    epochs : 539
    Epoch 1/1
    0s - loss: 0.9211 - acc: 0.7200
    epochs : 540
    Epoch 1/1
    0s - loss: 0.6190 - acc: 0.7400
    epochs : 541
    Epoch 1/1
    0s - loss: 0.6345 - acc: 0.7400
    epochs : 542
    Epoch 1/1
    0s - loss: 0.5150 - acc: 0.8200
    epochs : 543
    Epoch 1/1
    0s - loss: 0.2668 - acc: 0.8600
    epochs : 544
    Epoch 1/1
    0s - loss: 0.3076 - acc: 0.9200
    epochs : 545
    Epoch 1/1
    0s - loss: 0.4140 - acc: 0.8000
    epochs : 546
    Epoch 1/1
    0s - loss: 1.2432 - acc: 0.6800
    epochs : 547
    Epoch 1/1
    0s - loss: 0.8839 - acc: 0.6400
    epochs : 548
    Epoch 1/1
    0s - loss: 0.7337 - acc: 0.6400
    epochs : 549
    Epoch 1/1
    0s - loss: 0.7355 - acc: 0.7400
    epochs : 550
    Epoch 1/1
    0s - loss: 0.9124 - acc: 0.6600
    epochs : 551
    Epoch 1/1
    0s - loss: 1.8845 - acc: 0.5000
    epochs : 552
    Epoch 1/1
    0s - loss: 0.7822 - acc: 0.7400
    epochs : 553
    Epoch 1/1
    0s - loss: 0.6093 - acc: 0.7400
    epochs : 554
    Epoch 1/1
    0s - loss: 0.6631 - acc: 0.7200
    epochs : 555
    Epoch 1/1
    0s - loss: 1.0264 - acc: 0.6200
    epochs : 556
    Epoch 1/1
    0s - loss: 0.7138 - acc: 0.6800
    epochs : 557
    Epoch 1/1
    0s - loss: 0.6857 - acc: 0.7000
    epochs : 558
    Epoch 1/1
    0s - loss: 0.7170 - acc: 0.7200
    epochs : 559
    Epoch 1/1
    0s - loss: 0.7986 - acc: 0.6800
    epochs : 560
    Epoch 1/1
    0s - loss: 0.6366 - acc: 0.7400
    epochs : 561
    Epoch 1/1
    0s - loss: 0.3346 - acc: 0.9200
    epochs : 562
    Epoch 1/1
    0s - loss: 0.4981 - acc: 0.7800
    epochs : 563
    Epoch 1/1
    0s - loss: 0.3882 - acc: 0.8600
    epochs : 564
    Epoch 1/1
    0s - loss: 0.2622 - acc: 0.8600
    epochs : 565
    Epoch 1/1
    0s - loss: 0.4885 - acc: 0.8400
    epochs : 566
    Epoch 1/1
    0s - loss: 0.3556 - acc: 0.9000
    epochs : 567
    Epoch 1/1
    0s - loss: 0.6488 - acc: 0.8400
    epochs : 568
    Epoch 1/1
    0s - loss: 0.2506 - acc: 0.9200
    epochs : 569
    Epoch 1/1
    0s - loss: 0.1874 - acc: 0.9800
    epochs : 570
    Epoch 1/1
    0s - loss: 0.2154 - acc: 0.9200
    epochs : 571
    Epoch 1/1
    0s - loss: 0.2411 - acc: 0.9400
    epochs : 572
    Epoch 1/1
    0s - loss: 0.5241 - acc: 0.8000
    epochs : 573
    Epoch 1/1
    0s - loss: 0.6929 - acc: 0.7600
    epochs : 574
    Epoch 1/1
    0s - loss: 0.4831 - acc: 0.7800
    epochs : 575
    Epoch 1/1
    0s - loss: 0.2343 - acc: 0.9200
    epochs : 576
    Epoch 1/1
    0s - loss: 0.3845 - acc: 0.8800
    epochs : 577
    Epoch 1/1
    0s - loss: 0.3748 - acc: 0.9000
    epochs : 578
    Epoch 1/1
    0s - loss: 0.2576 - acc: 0.9400
    epochs : 579
    Epoch 1/1
    0s - loss: 0.1412 - acc: 0.9800
    epochs : 580
    Epoch 1/1
    0s - loss: 0.0965 - acc: 0.9800
    epochs : 581
    Epoch 1/1
    0s - loss: 0.0737 - acc: 1.0000
    epochs : 582
    Epoch 1/1
    0s - loss: 0.0734 - acc: 0.9800
    epochs : 583
    Epoch 1/1
    0s - loss: 0.0566 - acc: 1.0000
    epochs : 584
    Epoch 1/1
    0s - loss: 0.0435 - acc: 1.0000
    epochs : 585
    Epoch 1/1
    0s - loss: 0.0351 - acc: 1.0000
    epochs : 586
    Epoch 1/1
    0s - loss: 0.0308 - acc: 1.0000
    epochs : 587
    Epoch 1/1
    0s - loss: 0.0432 - acc: 1.0000
    epochs : 588
    Epoch 1/1
    0s - loss: 0.0363 - acc: 1.0000
    epochs : 589
    Epoch 1/1
    0s - loss: 0.0257 - acc: 1.0000
    epochs : 590
    Epoch 1/1
    0s - loss: 0.0220 - acc: 1.0000
    epochs : 591
    Epoch 1/1
    0s - loss: 0.0616 - acc: 0.9600
    epochs : 592
    Epoch 1/1
    0s - loss: 0.0800 - acc: 0.9800
    epochs : 593
    Epoch 1/1
    0s - loss: 0.0775 - acc: 1.0000
    epochs : 594
    Epoch 1/1
    0s - loss: 1.0967 - acc: 0.7200
    epochs : 595
    Epoch 1/1
    0s - loss: 1.0072 - acc: 0.6800
    epochs : 596
    Epoch 1/1
    0s - loss: 1.0471 - acc: 0.7000
    epochs : 597
    Epoch 1/1
    0s - loss: 0.8654 - acc: 0.7200
    epochs : 598
    Epoch 1/1
    0s - loss: 0.5787 - acc: 0.7000
    epochs : 599
    Epoch 1/1
    0s - loss: 0.5886 - acc: 0.7400
    epochs : 600
    Epoch 1/1
    0s - loss: 0.4694 - acc: 0.8000
    epochs : 601
    Epoch 1/1
    0s - loss: 0.4832 - acc: 0.7400
    epochs : 602
    Epoch 1/1
    0s - loss: 0.3161 - acc: 0.8800
    epochs : 603
    Epoch 1/1
    0s - loss: 0.4135 - acc: 0.8800
    epochs : 604
    Epoch 1/1
    0s - loss: 0.1724 - acc: 0.9600
    epochs : 605
    Epoch 1/1
    0s - loss: 0.1239 - acc: 0.9800
    epochs : 606
    Epoch 1/1
    0s - loss: 0.0614 - acc: 1.0000
    epochs : 607
    Epoch 1/1
    0s - loss: 0.0477 - acc: 1.0000
    epochs : 608
    Epoch 1/1
    0s - loss: 0.0400 - acc: 1.0000
    epochs : 609
    Epoch 1/1
    0s - loss: 0.0343 - acc: 1.0000
    epochs : 610
    Epoch 1/1
    0s - loss: 0.0269 - acc: 1.0000
    epochs : 611
    Epoch 1/1
    0s - loss: 0.0242 - acc: 1.0000
    epochs : 612
    Epoch 1/1
    0s - loss: 0.0221 - acc: 1.0000
    epochs : 613
    Epoch 1/1
    0s - loss: 0.0205 - acc: 1.0000
    epochs : 614
    Epoch 1/1
    0s - loss: 0.0195 - acc: 1.0000
    epochs : 615
    Epoch 1/1
    0s - loss: 0.0180 - acc: 1.0000
    epochs : 616
    Epoch 1/1
    0s - loss: 0.0172 - acc: 1.0000
    epochs : 617
    Epoch 1/1
    0s - loss: 0.0161 - acc: 1.0000
    epochs : 618
    Epoch 1/1
    0s - loss: 0.0155 - acc: 1.0000
    epochs : 619
    Epoch 1/1
    0s - loss: 0.0155 - acc: 1.0000
    epochs : 620
    Epoch 1/1
    0s - loss: 0.0151 - acc: 1.0000
    epochs : 621
    Epoch 1/1
    0s - loss: 0.0149 - acc: 1.0000
    epochs : 622
    Epoch 1/1
    0s - loss: 0.0124 - acc: 1.0000
    epochs : 623
    Epoch 1/1
    0s - loss: 0.0118 - acc: 1.0000
    epochs : 624
    Epoch 1/1
    0s - loss: 0.0113 - acc: 1.0000
    epochs : 625
    Epoch 1/1
    0s - loss: 0.0123 - acc: 1.0000
    epochs : 626
    Epoch 1/1
    0s - loss: 0.0709 - acc: 0.9600
    epochs : 627
    Epoch 1/1
    0s - loss: 0.1954 - acc: 0.9600
    epochs : 628
    Epoch 1/1
    0s - loss: 0.6225 - acc: 0.7400
    epochs : 629
    Epoch 1/1
    0s - loss: 1.5863 - acc: 0.5600
    epochs : 630
    Epoch 1/1
    0s - loss: 0.9945 - acc: 0.6200
    epochs : 631
    Epoch 1/1
    0s - loss: 1.1161 - acc: 0.5400
    epochs : 632
    Epoch 1/1
    0s - loss: 0.9314 - acc: 0.5800
    epochs : 633
    Epoch 1/1
    0s - loss: 0.7922 - acc: 0.7000
    epochs : 634
    Epoch 1/1
    0s - loss: 0.4506 - acc: 0.8400
    epochs : 635
    Epoch 1/1
    0s - loss: 0.3163 - acc: 0.8800
    epochs : 636
    Epoch 1/1
    0s - loss: 0.3438 - acc: 0.8800
    epochs : 637
    Epoch 1/1
    0s - loss: 0.3324 - acc: 0.8400
    epochs : 638
    Epoch 1/1
    0s - loss: 0.2418 - acc: 0.9000
    epochs : 639
    Epoch 1/1
    0s - loss: 0.1752 - acc: 0.9200
    epochs : 640
    Epoch 1/1
    0s - loss: 0.1152 - acc: 0.9600
    epochs : 641
    Epoch 1/1
    0s - loss: 0.0892 - acc: 1.0000
    epochs : 642
    Epoch 1/1
    0s - loss: 0.0697 - acc: 1.0000
    epochs : 643
    Epoch 1/1
    0s - loss: 0.0573 - acc: 1.0000
    epochs : 644
    Epoch 1/1
    0s - loss: 0.0547 - acc: 1.0000
    epochs : 645
    Epoch 1/1
    0s - loss: 0.0515 - acc: 1.0000
    epochs : 646
    Epoch 1/1
    0s - loss: 0.0428 - acc: 1.0000
    epochs : 647
    Epoch 1/1
    0s - loss: 0.0400 - acc: 1.0000
    epochs : 648
    Epoch 1/1
    0s - loss: 0.0397 - acc: 1.0000
    epochs : 649
    Epoch 1/1
    0s - loss: 0.1281 - acc: 0.9400
    epochs : 650
    Epoch 1/1
    0s - loss: 0.3269 - acc: 0.9200
    epochs : 651
    Epoch 1/1
    0s - loss: 0.4637 - acc: 0.8600
    epochs : 652
    Epoch 1/1
    0s - loss: 1.6348 - acc: 0.5200
    epochs : 653
    Epoch 1/1
    0s - loss: 1.2383 - acc: 0.5600
    epochs : 654
    Epoch 1/1
    0s - loss: 0.6423 - acc: 0.7600
    epochs : 655
    Epoch 1/1
    0s - loss: 0.4284 - acc: 0.8000
    epochs : 656
    Epoch 1/1
    0s - loss: 0.2901 - acc: 0.9200
    epochs : 657
    Epoch 1/1
    0s - loss: 0.2389 - acc: 0.9400
    epochs : 658
    Epoch 1/1
    0s - loss: 0.2609 - acc: 0.8800
    epochs : 659
    Epoch 1/1
    0s - loss: 0.2326 - acc: 0.9200
    epochs : 660
    Epoch 1/1
    0s - loss: 0.4189 - acc: 0.8200
    epochs : 661
    Epoch 1/1
    0s - loss: 0.2156 - acc: 0.9800
    epochs : 662
    Epoch 1/1
    0s - loss: 0.0915 - acc: 1.0000
    epochs : 663
    Epoch 1/1
    0s - loss: 0.0569 - acc: 1.0000
    epochs : 664
    Epoch 1/1
    0s - loss: 0.0550 - acc: 1.0000
    epochs : 665
    Epoch 1/1
    0s - loss: 0.0404 - acc: 1.0000
    epochs : 666
    Epoch 1/1
    0s - loss: 0.0347 - acc: 1.0000
    epochs : 667
    Epoch 1/1
    0s - loss: 0.0303 - acc: 1.0000
    epochs : 668
    Epoch 1/1
    0s - loss: 0.0275 - acc: 1.0000
    epochs : 669
    Epoch 1/1
    0s - loss: 0.0259 - acc: 1.0000
    epochs : 670
    Epoch 1/1
    0s - loss: 0.0246 - acc: 1.0000
    epochs : 671
    Epoch 1/1
    0s - loss: 0.0218 - acc: 1.0000
    epochs : 672
    Epoch 1/1
    0s - loss: 0.0186 - acc: 1.0000
    epochs : 673
    Epoch 1/1
    0s - loss: 0.0167 - acc: 1.0000
    epochs : 674
    Epoch 1/1
    0s - loss: 0.0154 - acc: 1.0000
    epochs : 675
    Epoch 1/1
    0s - loss: 0.0143 - acc: 1.0000
    epochs : 676
    Epoch 1/1
    0s - loss: 0.0133 - acc: 1.0000
    epochs : 677
    Epoch 1/1
    0s - loss: 0.0128 - acc: 1.0000
    epochs : 678
    Epoch 1/1
    0s - loss: 0.0124 - acc: 1.0000
    epochs : 679
    Epoch 1/1
    0s - loss: 0.0116 - acc: 1.0000
    epochs : 680
    Epoch 1/1
    0s - loss: 0.0108 - acc: 1.0000
    epochs : 681
    Epoch 1/1
    0s - loss: 0.0100 - acc: 1.0000
    epochs : 682
    Epoch 1/1
    0s - loss: 0.0093 - acc: 1.0000
    epochs : 683
    Epoch 1/1
    0s - loss: 0.0088 - acc: 1.0000
    epochs : 684
    Epoch 1/1
    0s - loss: 0.0083 - acc: 1.0000
    epochs : 685
    Epoch 1/1
    0s - loss: 0.0078 - acc: 1.0000
    epochs : 686
    Epoch 1/1
    0s - loss: 0.0073 - acc: 1.0000
    epochs : 687
    Epoch 1/1
    0s - loss: 0.0069 - acc: 1.0000
    epochs : 688
    Epoch 1/1
    0s - loss: 0.0065 - acc: 1.0000
    epochs : 689
    Epoch 1/1
    0s - loss: 0.0061 - acc: 1.0000
    epochs : 690
    Epoch 1/1
    0s - loss: 0.0057 - acc: 1.0000
    epochs : 691
    Epoch 1/1
    0s - loss: 0.0054 - acc: 1.0000
    epochs : 692
    Epoch 1/1
    0s - loss: 0.0051 - acc: 1.0000
    epochs : 693
    Epoch 1/1
    0s - loss: 0.0048 - acc: 1.0000
    epochs : 694
    Epoch 1/1
    0s - loss: 0.0046 - acc: 1.0000
    epochs : 695
    Epoch 1/1
    0s - loss: 0.0044 - acc: 1.0000
    epochs : 696
    Epoch 1/1
    0s - loss: 0.0042 - acc: 1.0000
    epochs : 697
    Epoch 1/1
    0s - loss: 0.0041 - acc: 1.0000
    epochs : 698
    Epoch 1/1
    0s - loss: 0.0040 - acc: 1.0000
    epochs : 699
    Epoch 1/1
    0s - loss: 0.0042 - acc: 1.0000
    epochs : 700
    Epoch 1/1
    0s - loss: 0.0053 - acc: 1.0000
    epochs : 701
    Epoch 1/1
    0s - loss: 0.0140 - acc: 1.0000
    epochs : 702
    Epoch 1/1
    0s - loss: 0.7369 - acc: 0.8000
    epochs : 703
    Epoch 1/1
    0s - loss: 1.0815 - acc: 0.7400
    epochs : 704
    Epoch 1/1
    0s - loss: 2.4531 - acc: 0.3000
    epochs : 705
    Epoch 1/1
    0s - loss: 1.7171 - acc: 0.4000
    epochs : 706
    Epoch 1/1
    0s - loss: 1.7062 - acc: 0.4400
    epochs : 707
    Epoch 1/1
    0s - loss: 1.2960 - acc: 0.4800
    epochs : 708
    Epoch 1/1
    0s - loss: 1.0641 - acc: 0.5600
    epochs : 709
    Epoch 1/1
    0s - loss: 0.9618 - acc: 0.6600
    epochs : 710
    Epoch 1/1
    0s - loss: 0.8526 - acc: 0.6800
    epochs : 711
    Epoch 1/1
    0s - loss: 0.9053 - acc: 0.6800
    epochs : 712
    Epoch 1/1
    0s - loss: 0.7270 - acc: 0.7200
    epochs : 713
    Epoch 1/1
    0s - loss: 0.6746 - acc: 0.7400
    epochs : 714
    Epoch 1/1
    0s - loss: 0.6144 - acc: 0.8000
    epochs : 715
    Epoch 1/1
    0s - loss: 0.4944 - acc: 0.8200
    epochs : 716
    Epoch 1/1
    0s - loss: 0.4070 - acc: 0.9400
    epochs : 717
    Epoch 1/1
    0s - loss: 0.4968 - acc: 0.8400
    epochs : 718
    Epoch 1/1
    0s - loss: 0.4095 - acc: 0.8000
    epochs : 719
    Epoch 1/1
    0s - loss: 0.5715 - acc: 0.8000
    epochs : 720
    Epoch 1/1
    0s - loss: 0.6920 - acc: 0.7200
    epochs : 721
    Epoch 1/1
    0s - loss: 0.2484 - acc: 0.9200
    epochs : 722
    Epoch 1/1
    0s - loss: 0.2183 - acc: 0.9600
    epochs : 723
    Epoch 1/1
    0s - loss: 0.0986 - acc: 0.9800
    epochs : 724
    Epoch 1/1
    0s - loss: 0.2288 - acc: 0.9400
    epochs : 725
    Epoch 1/1
    0s - loss: 0.7648 - acc: 0.7200
    epochs : 726
    Epoch 1/1
    0s - loss: 0.7239 - acc: 0.7400
    epochs : 727
    Epoch 1/1
    0s - loss: 0.3425 - acc: 0.8400
    epochs : 728
    Epoch 1/1
    0s - loss: 0.0930 - acc: 1.0000
    epochs : 729
    Epoch 1/1
    0s - loss: 0.4379 - acc: 0.8800
    epochs : 730
    Epoch 1/1
    0s - loss: 0.2400 - acc: 0.9000
    epochs : 731
    Epoch 1/1
    0s - loss: 0.2115 - acc: 0.9200
    epochs : 732
    Epoch 1/1
    0s - loss: 0.0570 - acc: 1.0000
    epochs : 733
    Epoch 1/1
    0s - loss: 0.0537 - acc: 1.0000
    epochs : 734
    Epoch 1/1
    0s - loss: 0.0484 - acc: 1.0000
    epochs : 735
    Epoch 1/1
    0s - loss: 0.0357 - acc: 1.0000
    epochs : 736
    Epoch 1/1
    0s - loss: 0.0358 - acc: 1.0000
    epochs : 737
    Epoch 1/1
    0s - loss: 0.0262 - acc: 1.0000
    epochs : 738
    Epoch 1/1
    0s - loss: 0.0226 - acc: 1.0000
    epochs : 739
    Epoch 1/1
    0s - loss: 0.0267 - acc: 1.0000
    epochs : 740
    Epoch 1/1
    0s - loss: 0.0274 - acc: 1.0000
    epochs : 741
    Epoch 1/1
    0s - loss: 0.0179 - acc: 1.0000
    epochs : 742
    Epoch 1/1
    0s - loss: 0.0129 - acc: 1.0000
    epochs : 743
    Epoch 1/1
    0s - loss: 0.0119 - acc: 1.0000
    epochs : 744
    Epoch 1/1
    0s - loss: 0.0112 - acc: 1.0000
    epochs : 745
    Epoch 1/1
    0s - loss: 0.0110 - acc: 1.0000
    epochs : 746
    Epoch 1/1
    0s - loss: 0.0106 - acc: 1.0000
    epochs : 747
    Epoch 1/1
    0s - loss: 0.0100 - acc: 1.0000
    epochs : 748
    Epoch 1/1
    0s - loss: 0.0089 - acc: 1.0000
    epochs : 749
    Epoch 1/1
    0s - loss: 0.0081 - acc: 1.0000
    epochs : 750
    Epoch 1/1
    0s - loss: 0.0078 - acc: 1.0000
    epochs : 751
    Epoch 1/1
    0s - loss: 0.0075 - acc: 1.0000
    epochs : 752
    Epoch 1/1
    0s - loss: 0.0075 - acc: 1.0000
    epochs : 753
    Epoch 1/1
    0s - loss: 0.0072 - acc: 1.0000
    epochs : 754
    Epoch 1/1
    0s - loss: 0.0070 - acc: 1.0000
    epochs : 755
    Epoch 1/1
    0s - loss: 0.0074 - acc: 1.0000
    epochs : 756
    Epoch 1/1
    0s - loss: 0.0072 - acc: 1.0000
    epochs : 757
    Epoch 1/1
    0s - loss: 0.0073 - acc: 1.0000
    epochs : 758
    Epoch 1/1
    0s - loss: 0.0060 - acc: 1.0000
    epochs : 759
    Epoch 1/1
    0s - loss: 0.0055 - acc: 1.0000
    epochs : 760
    Epoch 1/1
    0s - loss: 0.0053 - acc: 1.0000
    epochs : 761
    Epoch 1/1
    0s - loss: 0.0050 - acc: 1.0000
    epochs : 762
    Epoch 1/1
    0s - loss: 0.0048 - acc: 1.0000
    epochs : 763
    Epoch 1/1
    0s - loss: 0.0046 - acc: 1.0000
    epochs : 764
    Epoch 1/1
    0s - loss: 0.0043 - acc: 1.0000
    epochs : 765
    Epoch 1/1
    0s - loss: 0.0040 - acc: 1.0000
    epochs : 766
    Epoch 1/1
    0s - loss: 0.0037 - acc: 1.0000
    epochs : 767
    Epoch 1/1
    0s - loss: 0.0035 - acc: 1.0000
    epochs : 768
    Epoch 1/1
    0s - loss: 0.0033 - acc: 1.0000
    epochs : 769
    Epoch 1/1
    0s - loss: 0.0032 - acc: 1.0000
    epochs : 770
    Epoch 1/1
    0s - loss: 0.0030 - acc: 1.0000
    epochs : 771
    Epoch 1/1
    0s - loss: 0.0029 - acc: 1.0000
    epochs : 772
    Epoch 1/1
    0s - loss: 0.0028 - acc: 1.0000
    epochs : 773
    Epoch 1/1
    0s - loss: 0.0027 - acc: 1.0000
    epochs : 774
    Epoch 1/1
    0s - loss: 0.0025 - acc: 1.0000
    epochs : 775
    Epoch 1/1
    0s - loss: 0.0024 - acc: 1.0000
    epochs : 776
    Epoch 1/1
    0s - loss: 0.0024 - acc: 1.0000
    epochs : 777
    Epoch 1/1
    0s - loss: 0.0023 - acc: 1.0000
    epochs : 778
    Epoch 1/1
    0s - loss: 0.0022 - acc: 1.0000
    epochs : 779
    Epoch 1/1
    0s - loss: 0.0021 - acc: 1.0000
    epochs : 780
    Epoch 1/1
    0s - loss: 0.0020 - acc: 1.0000
    epochs : 781
    Epoch 1/1
    0s - loss: 0.0020 - acc: 1.0000
    epochs : 782
    Epoch 1/1
    0s - loss: 0.0019 - acc: 1.0000
    epochs : 783
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 784
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 785
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 786
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 787
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 788
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 789
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 790
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 791
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 792
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 793
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 794
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 795
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 796
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 797
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 798
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 799
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 800
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 801
    Epoch 1/1
    0s - loss: 0.0010 - acc: 1.0000
    epochs : 802
    Epoch 1/1
    0s - loss: 0.0010 - acc: 1.0000
    epochs : 803
    Epoch 1/1
    0s - loss: 9.7699e-04 - acc: 1.0000
    epochs : 804
    Epoch 1/1
    0s - loss: 9.5463e-04 - acc: 1.0000
    epochs : 805
    Epoch 1/1
    0s - loss: 9.3343e-04 - acc: 1.0000
    epochs : 806
    Epoch 1/1
    0s - loss: 9.1585e-04 - acc: 1.0000
    epochs : 807
    Epoch 1/1
    0s - loss: 8.9587e-04 - acc: 1.0000
    epochs : 808
    Epoch 1/1
    0s - loss: 8.7200e-04 - acc: 1.0000
    epochs : 809
    Epoch 1/1
    0s - loss: 8.4878e-04 - acc: 1.0000
    epochs : 810
    Epoch 1/1
    0s - loss: 8.3539e-04 - acc: 1.0000
    epochs : 811
    Epoch 1/1
    0s - loss: 8.1946e-04 - acc: 1.0000
    epochs : 812
    Epoch 1/1
    0s - loss: 8.4101e-04 - acc: 1.0000
    epochs : 813
    Epoch 1/1
    0s - loss: 8.0379e-04 - acc: 1.0000
    epochs : 814
    Epoch 1/1
    0s - loss: 7.9850e-04 - acc: 1.0000
    epochs : 815
    Epoch 1/1
    0s - loss: 7.6260e-04 - acc: 1.0000
    epochs : 816
    Epoch 1/1
    0s - loss: 7.8459e-04 - acc: 1.0000
    epochs : 817
    Epoch 1/1
    0s - loss: 7.7435e-04 - acc: 1.0000
    epochs : 818
    Epoch 1/1
    0s - loss: 7.3461e-04 - acc: 1.0000
    epochs : 819
    Epoch 1/1
    0s - loss: 7.0705e-04 - acc: 1.0000
    epochs : 820
    Epoch 1/1
    0s - loss: 6.7604e-04 - acc: 1.0000
    epochs : 821
    Epoch 1/1
    0s - loss: 6.4830e-04 - acc: 1.0000
    epochs : 822
    Epoch 1/1
    0s - loss: 6.3416e-04 - acc: 1.0000
    epochs : 823
    Epoch 1/1
    0s - loss: 6.2342e-04 - acc: 1.0000
    epochs : 824
    Epoch 1/1
    0s - loss: 6.0476e-04 - acc: 1.0000
    epochs : 825
    Epoch 1/1
    0s - loss: 5.7970e-04 - acc: 1.0000
    epochs : 826
    Epoch 1/1
    0s - loss: 5.5105e-04 - acc: 1.0000
    epochs : 827
    Epoch 1/1
    0s - loss: 5.3213e-04 - acc: 1.0000
    epochs : 828
    Epoch 1/1
    0s - loss: 5.1329e-04 - acc: 1.0000
    epochs : 829
    Epoch 1/1
    0s - loss: 4.9346e-04 - acc: 1.0000
    epochs : 830
    Epoch 1/1
    0s - loss: 4.7611e-04 - acc: 1.0000
    epochs : 831
    Epoch 1/1
    0s - loss: 4.5754e-04 - acc: 1.0000
    epochs : 832
    Epoch 1/1
    0s - loss: 4.4017e-04 - acc: 1.0000
    epochs : 833
    Epoch 1/1
    0s - loss: 4.2210e-04 - acc: 1.0000
    epochs : 834
    Epoch 1/1
    0s - loss: 4.0581e-04 - acc: 1.0000
    epochs : 835
    Epoch 1/1
    0s - loss: 3.8952e-04 - acc: 1.0000
    epochs : 836
    Epoch 1/1
    0s - loss: 3.7579e-04 - acc: 1.0000
    epochs : 837
    Epoch 1/1
    0s - loss: 3.6043e-04 - acc: 1.0000
    epochs : 838
    Epoch 1/1
    0s - loss: 3.4602e-04 - acc: 1.0000
    epochs : 839
    Epoch 1/1
    0s - loss: 3.3337e-04 - acc: 1.0000
    epochs : 840
    Epoch 1/1
    0s - loss: 3.2296e-04 - acc: 1.0000
    epochs : 841
    Epoch 1/1
    0s - loss: 3.1188e-04 - acc: 1.0000
    epochs : 842
    Epoch 1/1
    0s - loss: 3.0350e-04 - acc: 1.0000
    epochs : 843
    Epoch 1/1
    0s - loss: 2.9468e-04 - acc: 1.0000
    epochs : 844
    Epoch 1/1
    0s - loss: 2.8564e-04 - acc: 1.0000
    epochs : 845
    Epoch 1/1
    0s - loss: 2.7238e-04 - acc: 1.0000
    epochs : 846
    Epoch 1/1
    0s - loss: 2.6223e-04 - acc: 1.0000
    epochs : 847
    Epoch 1/1
    0s - loss: 2.5331e-04 - acc: 1.0000
    epochs : 848
    Epoch 1/1
    0s - loss: 2.4500e-04 - acc: 1.0000
    epochs : 849
    Epoch 1/1
    0s - loss: 2.3789e-04 - acc: 1.0000
    epochs : 850
    Epoch 1/1
    0s - loss: 2.3170e-04 - acc: 1.0000
    epochs : 851
    Epoch 1/1
    0s - loss: 2.2636e-04 - acc: 1.0000
    epochs : 852
    Epoch 1/1
    0s - loss: 2.2012e-04 - acc: 1.0000
    epochs : 853
    Epoch 1/1
    0s - loss: 2.1566e-04 - acc: 1.0000
    epochs : 854
    Epoch 1/1
    0s - loss: 2.1182e-04 - acc: 1.0000
    epochs : 855
    Epoch 1/1
    0s - loss: 2.0826e-04 - acc: 1.0000
    epochs : 856
    Epoch 1/1
    0s - loss: 2.0486e-04 - acc: 1.0000
    epochs : 857
    Epoch 1/1
    0s - loss: 2.0327e-04 - acc: 1.0000
    epochs : 858
    Epoch 1/1
    0s - loss: 1.9379e-04 - acc: 1.0000
    epochs : 859
    Epoch 1/1
    0s - loss: 1.9059e-04 - acc: 1.0000
    epochs : 860
    Epoch 1/1
    0s - loss: 1.8231e-04 - acc: 1.0000
    epochs : 861
    Epoch 1/1
    0s - loss: 1.7724e-04 - acc: 1.0000
    epochs : 862
    Epoch 1/1
    0s - loss: 1.7480e-04 - acc: 1.0000
    epochs : 863
    Epoch 1/1
    0s - loss: 1.6890e-04 - acc: 1.0000
    epochs : 864
    Epoch 1/1
    0s - loss: 1.6616e-04 - acc: 1.0000
    epochs : 865
    Epoch 1/1
    0s - loss: 1.6192e-04 - acc: 1.0000
    epochs : 866
    Epoch 1/1
    0s - loss: 1.5685e-04 - acc: 1.0000
    epochs : 867
    Epoch 1/1
    0s - loss: 1.5087e-04 - acc: 1.0000
    epochs : 868
    Epoch 1/1
    0s - loss: 1.4584e-04 - acc: 1.0000
    epochs : 869
    Epoch 1/1
    0s - loss: 1.4131e-04 - acc: 1.0000
    epochs : 870
    Epoch 1/1
    0s - loss: 1.3731e-04 - acc: 1.0000
    epochs : 871
    Epoch 1/1
    0s - loss: 1.3352e-04 - acc: 1.0000
    epochs : 872
    Epoch 1/1
    0s - loss: 1.2909e-04 - acc: 1.0000
    epochs : 873
    Epoch 1/1
    0s - loss: 1.2560e-04 - acc: 1.0000
    epochs : 874
    Epoch 1/1
    0s - loss: 1.2152e-04 - acc: 1.0000
    epochs : 875
    Epoch 1/1
    0s - loss: 1.1878e-04 - acc: 1.0000
    epochs : 876
    Epoch 1/1
    0s - loss: 1.1571e-04 - acc: 1.0000
    epochs : 877
    Epoch 1/1
    0s - loss: 1.1328e-04 - acc: 1.0000
    epochs : 878
    Epoch 1/1
    0s - loss: 1.0921e-04 - acc: 1.0000
    epochs : 879
    Epoch 1/1
    0s - loss: 1.0987e-04 - acc: 1.0000
    epochs : 880
    Epoch 1/1
    0s - loss: 1.0482e-04 - acc: 1.0000
    epochs : 881
    Epoch 1/1
    0s - loss: 9.8649e-05 - acc: 1.0000
    epochs : 882
    Epoch 1/1
    0s - loss: 1.0271e-04 - acc: 1.0000
    epochs : 883
    Epoch 1/1
    0s - loss: 9.7396e-05 - acc: 1.0000
    epochs : 884
    Epoch 1/1
    0s - loss: 9.3854e-05 - acc: 1.0000
    epochs : 885
    Epoch 1/1
    0s - loss: 8.9132e-05 - acc: 1.0000
    epochs : 886
    Epoch 1/1
    0s - loss: 8.6192e-05 - acc: 1.0000
    epochs : 887
    Epoch 1/1
    0s - loss: 8.4249e-05 - acc: 1.0000
    epochs : 888
    Epoch 1/1
    0s - loss: 8.2914e-05 - acc: 1.0000
    epochs : 889
    Epoch 1/1
    0s - loss: 8.3664e-05 - acc: 1.0000
    epochs : 890
    Epoch 1/1
    0s - loss: 8.3595e-05 - acc: 1.0000
    epochs : 891
    Epoch 1/1
    0s - loss: 8.0615e-05 - acc: 1.0000
    epochs : 892
    Epoch 1/1
    0s - loss: 7.5127e-05 - acc: 1.0000
    epochs : 893
    Epoch 1/1
    0s - loss: 7.1666e-05 - acc: 1.0000
    epochs : 894
    Epoch 1/1
    0s - loss: 6.9261e-05 - acc: 1.0000
    epochs : 895
    Epoch 1/1
    0s - loss: 6.8038e-05 - acc: 1.0000
    epochs : 896
    Epoch 1/1
    0s - loss: 6.7165e-05 - acc: 1.0000
    epochs : 897
    Epoch 1/1
    0s - loss: 6.6617e-05 - acc: 1.0000
    epochs : 898
    Epoch 1/1
    0s - loss: 6.6148e-05 - acc: 1.0000
    epochs : 899
    Epoch 1/1
    0s - loss: 6.2407e-05 - acc: 1.0000
    epochs : 900
    Epoch 1/1
    0s - loss: 6.0165e-05 - acc: 1.0000
    epochs : 901
    Epoch 1/1
    0s - loss: 5.7980e-05 - acc: 1.0000
    epochs : 902
    Epoch 1/1
    0s - loss: 5.6608e-05 - acc: 1.0000
    epochs : 903
    Epoch 1/1
    0s - loss: 5.6620e-05 - acc: 1.0000
    epochs : 904
    Epoch 1/1
    0s - loss: 5.5846e-05 - acc: 1.0000
    epochs : 905
    Epoch 1/1
    0s - loss: 5.5781e-05 - acc: 1.0000
    epochs : 906
    Epoch 1/1
    0s - loss: 5.4619e-05 - acc: 1.0000
    epochs : 907
    Epoch 1/1
    0s - loss: 5.1156e-05 - acc: 1.0000
    epochs : 908
    Epoch 1/1
    0s - loss: 4.9122e-05 - acc: 1.0000
    epochs : 909
    Epoch 1/1
    0s - loss: 4.7673e-05 - acc: 1.0000
    epochs : 910
    Epoch 1/1
    0s - loss: 4.5443e-05 - acc: 1.0000
    epochs : 911
    Epoch 1/1
    0s - loss: 4.4778e-05 - acc: 1.0000
    epochs : 912
    Epoch 1/1
    0s - loss: 4.4089e-05 - acc: 1.0000
    epochs : 913
    Epoch 1/1
    0s - loss: 4.3263e-05 - acc: 1.0000
    epochs : 914
    Epoch 1/1
    0s - loss: 4.2267e-05 - acc: 1.0000
    epochs : 915
    Epoch 1/1
    0s - loss: 4.0613e-05 - acc: 1.0000
    epochs : 916
    Epoch 1/1
    0s - loss: 3.8593e-05 - acc: 1.0000
    epochs : 917
    Epoch 1/1
    0s - loss: 3.7531e-05 - acc: 1.0000
    epochs : 918
    Epoch 1/1
    0s - loss: 3.6212e-05 - acc: 1.0000
    epochs : 919
    Epoch 1/1
    0s - loss: 3.5291e-05 - acc: 1.0000
    epochs : 920
    Epoch 1/1
    0s - loss: 3.4244e-05 - acc: 1.0000
    epochs : 921
    Epoch 1/1
    0s - loss: 3.3157e-05 - acc: 1.0000
    epochs : 922
    Epoch 1/1
    0s - loss: 3.2093e-05 - acc: 1.0000
    epochs : 923
    Epoch 1/1
    0s - loss: 3.1432e-05 - acc: 1.0000
    epochs : 924
    Epoch 1/1
    0s - loss: 3.0448e-05 - acc: 1.0000
    epochs : 925
    Epoch 1/1
    0s - loss: 2.9497e-05 - acc: 1.0000
    epochs : 926
    Epoch 1/1
    0s - loss: 2.8706e-05 - acc: 1.0000
    epochs : 927
    Epoch 1/1
    0s - loss: 2.7728e-05 - acc: 1.0000
    epochs : 928
    Epoch 1/1
    0s - loss: 2.7155e-05 - acc: 1.0000
    epochs : 929
    Epoch 1/1
    0s - loss: 2.6229e-05 - acc: 1.0000
    epochs : 930
    Epoch 1/1
    0s - loss: 2.5544e-05 - acc: 1.0000
    epochs : 931
    Epoch 1/1
    0s - loss: 2.4808e-05 - acc: 1.0000
    epochs : 932
    Epoch 1/1
    0s - loss: 2.4024e-05 - acc: 1.0000
    epochs : 933
    Epoch 1/1
    0s - loss: 2.3474e-05 - acc: 1.0000
    epochs : 934
    Epoch 1/1
    0s - loss: 2.2924e-05 - acc: 1.0000
    epochs : 935
    Epoch 1/1
    0s - loss: 2.2303e-05 - acc: 1.0000
    epochs : 936
    Epoch 1/1
    0s - loss: 2.1494e-05 - acc: 1.0000
    epochs : 937
    Epoch 1/1
    0s - loss: 2.0923e-05 - acc: 1.0000
    epochs : 938
    Epoch 1/1
    0s - loss: 2.0335e-05 - acc: 1.0000
    epochs : 939
    Epoch 1/1
    0s - loss: 1.9715e-05 - acc: 1.0000
    epochs : 940
    Epoch 1/1
    0s - loss: 1.9149e-05 - acc: 1.0000
    epochs : 941
    Epoch 1/1
    0s - loss: 1.8798e-05 - acc: 1.0000
    epochs : 942
    Epoch 1/1
    0s - loss: 1.8189e-05 - acc: 1.0000
    epochs : 943
    Epoch 1/1
    0s - loss: 1.7637e-05 - acc: 1.0000
    epochs : 944
    Epoch 1/1
    0s - loss: 1.7342e-05 - acc: 1.0000
    epochs : 945
    Epoch 1/1
    0s - loss: 1.6682e-05 - acc: 1.0000
    epochs : 946
    Epoch 1/1
    0s - loss: 1.6321e-05 - acc: 1.0000
    epochs : 947
    Epoch 1/1
    0s - loss: 1.5685e-05 - acc: 1.0000
    epochs : 948
    Epoch 1/1
    0s - loss: 1.5066e-05 - acc: 1.0000
    epochs : 949
    Epoch 1/1
    0s - loss: 1.4618e-05 - acc: 1.0000
    epochs : 950
    Epoch 1/1
    0s - loss: 1.4657e-05 - acc: 1.0000
    epochs : 951
    Epoch 1/1
    0s - loss: 1.6101e-05 - acc: 1.0000
    epochs : 952
    Epoch 1/1
    0s - loss: 1.9448e-05 - acc: 1.0000
    epochs : 953
    Epoch 1/1
    0s - loss: 1.6654e-05 - acc: 1.0000
    epochs : 954
    Epoch 1/1
    0s - loss: 1.5694e-05 - acc: 1.0000
    epochs : 955
    Epoch 1/1
    0s - loss: 1.3577e-05 - acc: 1.0000
    epochs : 956
    Epoch 1/1
    0s - loss: 1.3236e-05 - acc: 1.0000
    epochs : 957
    Epoch 1/1
    0s - loss: 1.2760e-05 - acc: 1.0000
    epochs : 958
    Epoch 1/1
    0s - loss: 1.2374e-05 - acc: 1.0000
    epochs : 959
    Epoch 1/1
    0s - loss: 1.2001e-05 - acc: 1.0000
    epochs : 960
    Epoch 1/1
    0s - loss: 1.1708e-05 - acc: 1.0000
    epochs : 961
    Epoch 1/1
    0s - loss: 1.1420e-05 - acc: 1.0000
    epochs : 962
    Epoch 1/1
    0s - loss: 1.1188e-05 - acc: 1.0000
    epochs : 963
    Epoch 1/1
    0s - loss: 1.0802e-05 - acc: 1.0000
    epochs : 964
    Epoch 1/1
    0s - loss: 1.0606e-05 - acc: 1.0000
    epochs : 965
    Epoch 1/1
    0s - loss: 1.0125e-05 - acc: 1.0000
    epochs : 966
    Epoch 1/1
    0s - loss: 9.8491e-06 - acc: 1.0000
    epochs : 967
    Epoch 1/1
    0s - loss: 9.5654e-06 - acc: 1.0000
    epochs : 968
    Epoch 1/1
    0s - loss: 9.2197e-06 - acc: 1.0000
    epochs : 969
    Epoch 1/1
    0s - loss: 8.9777e-06 - acc: 1.0000
    epochs : 970
    Epoch 1/1
    0s - loss: 8.7643e-06 - acc: 1.0000
    epochs : 971
    Epoch 1/1
    0s - loss: 8.6141e-06 - acc: 1.0000
    epochs : 972
    Epoch 1/1
    0s - loss: 8.5414e-06 - acc: 1.0000
    epochs : 973
    Epoch 1/1
    0s - loss: 8.2052e-06 - acc: 1.0000
    epochs : 974
    Epoch 1/1
    0s - loss: 8.2279e-06 - acc: 1.0000
    epochs : 975
    Epoch 1/1
    0s - loss: 7.8416e-06 - acc: 1.0000
    epochs : 976
    Epoch 1/1
    0s - loss: 7.7725e-06 - acc: 1.0000
    epochs : 977
    Epoch 1/1
    0s - loss: 7.7117e-06 - acc: 1.0000
    epochs : 978
    Epoch 1/1
    0s - loss: 7.7439e-06 - acc: 1.0000
    epochs : 979
    Epoch 1/1
    0s - loss: 7.0978e-06 - acc: 1.0000
    epochs : 980
    Epoch 1/1
    0s - loss: 7.5174e-06 - acc: 1.0000
    epochs : 981
    Epoch 1/1
    0s - loss: 6.6519e-06 - acc: 1.0000
    epochs : 982
    Epoch 1/1
    0s - loss: 7.7117e-06 - acc: 1.0000
    epochs : 983
    Epoch 1/1
    0s - loss: 6.2991e-06 - acc: 1.0000
    epochs : 984
    Epoch 1/1
    0s - loss: 6.7509e-06 - acc: 1.0000
    epochs : 985
    Epoch 1/1
    0s - loss: 6.4981e-06 - acc: 1.0000
    epochs : 986
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 987
    Epoch 1/1
    0s - loss: 3.4904 - acc: 0.4400
    epochs : 988
    Epoch 1/1
    0s - loss: 1.7947 - acc: 0.4000
    epochs : 989
    Epoch 1/1
    0s - loss: 1.5221 - acc: 0.4400
    epochs : 990
    Epoch 1/1
    0s - loss: 1.5055 - acc: 0.4600
    epochs : 991
    Epoch 1/1
    0s - loss: 1.2835 - acc: 0.4800
    epochs : 992
    Epoch 1/1
    0s - loss: 1.2393 - acc: 0.4800
    epochs : 993
    Epoch 1/1
    0s - loss: 1.2162 - acc: 0.5000
    epochs : 994
    Epoch 1/1
    0s - loss: 1.0769 - acc: 0.5400
    epochs : 995
    Epoch 1/1
    0s - loss: 1.0356 - acc: 0.5400
    epochs : 996
    Epoch 1/1
    0s - loss: 0.9990 - acc: 0.6400
    epochs : 997
    Epoch 1/1
    0s - loss: 0.9957 - acc: 0.6200
    epochs : 998
    Epoch 1/1
    0s - loss: 0.9323 - acc: 0.6400
    epochs : 999
    Epoch 1/1
    0s - loss: 0.9947 - acc: 0.6000
    epochs : 1000
    Epoch 1/1
    0s - loss: 1.0990 - acc: 0.5600
    epochs : 1001
    Epoch 1/1
    0s - loss: 1.0180 - acc: 0.6800
    epochs : 1002
    Epoch 1/1
    0s - loss: 0.9428 - acc: 0.6600
    epochs : 1003
    Epoch 1/1
    0s - loss: 0.9046 - acc: 0.6400
    epochs : 1004
    Epoch 1/1
    0s - loss: 0.7600 - acc: 0.7600
    epochs : 1005
    Epoch 1/1
    0s - loss: 0.7208 - acc: 0.6600
    epochs : 1006
    Epoch 1/1
    0s - loss: 0.6578 - acc: 0.7800
    epochs : 1007
    Epoch 1/1
    0s - loss: 0.5607 - acc: 0.8400
    epochs : 1008
    Epoch 1/1
    0s - loss: 0.7488 - acc: 0.7200
    epochs : 1009
    Epoch 1/1
    0s - loss: 0.5640 - acc: 0.8000
    epochs : 1010
    Epoch 1/1
    0s - loss: 1.1467 - acc: 0.5200
    epochs : 1011
    Epoch 1/1
    0s - loss: 0.9705 - acc: 0.7000
    epochs : 1012
    Epoch 1/1
    0s - loss: 0.7409 - acc: 0.7200
    epochs : 1013
    Epoch 1/1
    0s - loss: 0.5616 - acc: 0.7800
    epochs : 1014
    Epoch 1/1
    0s - loss: 0.6627 - acc: 0.7600
    epochs : 1015
    Epoch 1/1
    0s - loss: 0.5911 - acc: 0.8200
    epochs : 1016
    Epoch 1/1
    0s - loss: 0.5676 - acc: 0.8200
    epochs : 1017
    Epoch 1/1
    0s - loss: 0.4428 - acc: 0.8200
    epochs : 1018
    Epoch 1/1
    0s - loss: 0.4278 - acc: 0.8600
    epochs : 1019
    Epoch 1/1
    0s - loss: 0.4384 - acc: 0.8600
    epochs : 1020
    Epoch 1/1
    0s - loss: 0.7076 - acc: 0.7200
    epochs : 1021
    Epoch 1/1
    0s - loss: 0.7295 - acc: 0.7000
    epochs : 1022
    Epoch 1/1
    0s - loss: 0.4233 - acc: 0.8400
    epochs : 1023
    Epoch 1/1
    0s - loss: 0.4236 - acc: 0.8800
    epochs : 1024
    Epoch 1/1
    0s - loss: 0.4103 - acc: 0.8600
    epochs : 1025
    Epoch 1/1
    0s - loss: 0.6371 - acc: 0.7400
    epochs : 1026
    Epoch 1/1
    0s - loss: 0.3154 - acc: 0.9000
    epochs : 1027
    Epoch 1/1
    0s - loss: 0.2435 - acc: 0.9400
    epochs : 1028
    Epoch 1/1
    0s - loss: 0.2119 - acc: 0.9200
    epochs : 1029
    Epoch 1/1
    0s - loss: 0.1691 - acc: 0.9800
    epochs : 1030
    Epoch 1/1
    0s - loss: 0.1055 - acc: 1.0000
    epochs : 1031
    Epoch 1/1
    0s - loss: 0.0854 - acc: 1.0000
    epochs : 1032
    Epoch 1/1
    0s - loss: 0.2223 - acc: 0.9400
    epochs : 1033
    Epoch 1/1
    0s - loss: 0.6328 - acc: 0.8200
    epochs : 1034
    Epoch 1/1
    0s - loss: 0.7767 - acc: 0.7000
    epochs : 1035
    Epoch 1/1
    0s - loss: 0.7247 - acc: 0.7200
    epochs : 1036
    Epoch 1/1
    0s - loss: 0.3635 - acc: 0.8600
    epochs : 1037
    Epoch 1/1
    0s - loss: 0.2423 - acc: 0.9000
    epochs : 1038
    Epoch 1/1
    0s - loss: 0.1177 - acc: 0.9800
    epochs : 1039
    Epoch 1/1
    0s - loss: 0.1267 - acc: 0.9600
    epochs : 1040
    Epoch 1/1
    0s - loss: 0.1808 - acc: 0.9200
    epochs : 1041
    Epoch 1/1
    0s - loss: 0.2993 - acc: 0.8800
    epochs : 1042
    Epoch 1/1
    0s - loss: 0.3392 - acc: 0.9000
    epochs : 1043
    Epoch 1/1
    0s - loss: 0.1890 - acc: 0.9000
    epochs : 1044
    Epoch 1/1
    0s - loss: 0.5064 - acc: 0.8400
    epochs : 1045
    Epoch 1/1
    0s - loss: 0.3941 - acc: 0.8600
    epochs : 1046
    Epoch 1/1
    0s - loss: 0.2993 - acc: 0.9000
    epochs : 1047
    Epoch 1/1
    0s - loss: 0.6034 - acc: 0.7800
    epochs : 1048
    Epoch 1/1
    0s - loss: 0.4318 - acc: 0.8200
    epochs : 1049
    Epoch 1/1
    0s - loss: 0.5947 - acc: 0.7800
    epochs : 1050
    Epoch 1/1
    0s - loss: 0.7394 - acc: 0.7200
    epochs : 1051
    Epoch 1/1
    0s - loss: 0.8907 - acc: 0.7000
    epochs : 1052
    Epoch 1/1
    0s - loss: 0.7931 - acc: 0.6400
    epochs : 1053
    Epoch 1/1
    0s - loss: 0.4538 - acc: 0.8200
    epochs : 1054
    Epoch 1/1
    0s - loss: 0.4600 - acc: 0.8400
    epochs : 1055
    Epoch 1/1
    0s - loss: 0.3892 - acc: 0.8400
    epochs : 1056
    Epoch 1/1
    0s - loss: 0.2154 - acc: 0.9200
    epochs : 1057
    Epoch 1/1
    0s - loss: 0.1873 - acc: 0.9000
    epochs : 1058
    Epoch 1/1
    0s - loss: 0.1293 - acc: 0.9800
    epochs : 1059
    Epoch 1/1
    0s - loss: 0.1084 - acc: 1.0000
    epochs : 1060
    Epoch 1/1
    0s - loss: 0.3900 - acc: 0.8200
    epochs : 1061
    Epoch 1/1
    0s - loss: 0.1436 - acc: 0.9800
    epochs : 1062
    Epoch 1/1
    0s - loss: 0.1382 - acc: 1.0000
    epochs : 1063
    Epoch 1/1
    0s - loss: 0.1049 - acc: 0.9600
    epochs : 1064
    Epoch 1/1
    0s - loss: 0.2238 - acc: 0.9400
    epochs : 1065
    Epoch 1/1
    0s - loss: 0.1359 - acc: 0.9800
    epochs : 1066
    Epoch 1/1
    0s - loss: 0.1504 - acc: 0.9800
    epochs : 1067
    Epoch 1/1
    0s - loss: 0.1847 - acc: 0.9400
    epochs : 1068
    Epoch 1/1
    0s - loss: 0.3861 - acc: 0.8000
    epochs : 1069
    Epoch 1/1
    0s - loss: 0.1989 - acc: 0.9200
    epochs : 1070
    Epoch 1/1
    0s - loss: 0.1353 - acc: 0.9600
    epochs : 1071
    Epoch 1/1
    0s - loss: 0.1895 - acc: 0.9000
    epochs : 1072
    Epoch 1/1
    0s - loss: 0.1913 - acc: 0.9400
    epochs : 1073
    Epoch 1/1
    0s - loss: 0.2381 - acc: 0.9200
    epochs : 1074
    Epoch 1/1
    0s - loss: 0.0995 - acc: 0.9800
    epochs : 1075
    Epoch 1/1
    0s - loss: 0.0765 - acc: 1.0000
    epochs : 1076
    Epoch 1/1
    0s - loss: 0.0566 - acc: 1.0000
    epochs : 1077
    Epoch 1/1
    0s - loss: 0.0443 - acc: 1.0000
    epochs : 1078
    Epoch 1/1
    0s - loss: 0.0364 - acc: 1.0000
    epochs : 1079
    Epoch 1/1
    0s - loss: 0.0296 - acc: 1.0000
    epochs : 1080
    Epoch 1/1
    0s - loss: 0.0240 - acc: 1.0000
    epochs : 1081
    Epoch 1/1
    0s - loss: 0.0211 - acc: 1.0000
    epochs : 1082
    Epoch 1/1
    0s - loss: 0.0191 - acc: 1.0000
    epochs : 1083
    Epoch 1/1
    0s - loss: 0.0171 - acc: 1.0000
    epochs : 1084
    Epoch 1/1
    0s - loss: 0.0151 - acc: 1.0000
    epochs : 1085
    Epoch 1/1
    0s - loss: 0.0134 - acc: 1.0000
    epochs : 1086
    Epoch 1/1
    0s - loss: 0.0120 - acc: 1.0000
    epochs : 1087
    Epoch 1/1
    0s - loss: 0.0107 - acc: 1.0000
    epochs : 1088
    Epoch 1/1
    0s - loss: 0.0096 - acc: 1.0000
    epochs : 1089
    Epoch 1/1
    0s - loss: 0.0088 - acc: 1.0000
    epochs : 1090
    Epoch 1/1
    0s - loss: 0.0081 - acc: 1.0000
    epochs : 1091
    Epoch 1/1
    0s - loss: 0.0076 - acc: 1.0000
    epochs : 1092
    Epoch 1/1
    0s - loss: 0.0071 - acc: 1.0000
    epochs : 1093
    Epoch 1/1
    0s - loss: 0.0066 - acc: 1.0000
    epochs : 1094
    Epoch 1/1
    0s - loss: 0.0062 - acc: 1.0000
    epochs : 1095
    Epoch 1/1
    0s - loss: 0.0058 - acc: 1.0000
    epochs : 1096
    Epoch 1/1
    0s - loss: 0.0055 - acc: 1.0000
    epochs : 1097
    Epoch 1/1
    0s - loss: 0.0053 - acc: 1.0000
    epochs : 1098
    Epoch 1/1
    0s - loss: 0.0051 - acc: 1.0000
    epochs : 1099
    Epoch 1/1
    0s - loss: 0.0048 - acc: 1.0000
    epochs : 1100
    Epoch 1/1
    0s - loss: 0.0046 - acc: 1.0000
    epochs : 1101
    Epoch 1/1
    0s - loss: 0.0044 - acc: 1.0000
    epochs : 1102
    Epoch 1/1
    0s - loss: 0.0042 - acc: 1.0000
    epochs : 1103
    Epoch 1/1
    0s - loss: 0.0040 - acc: 1.0000
    epochs : 1104
    Epoch 1/1
    0s - loss: 0.0039 - acc: 1.0000
    epochs : 1105
    Epoch 1/1
    0s - loss: 0.0045 - acc: 1.0000
    epochs : 1106
    Epoch 1/1
    0s - loss: 0.0065 - acc: 1.0000
    epochs : 1107
    Epoch 1/1
    0s - loss: 0.0384 - acc: 1.0000
    epochs : 1108
    Epoch 1/1
    0s - loss: 0.6855 - acc: 0.7000
    epochs : 1109
    Epoch 1/1
    0s - loss: 2.7461 - acc: 0.3400
    epochs : 1110
    Epoch 1/1
    0s - loss: 1.1076 - acc: 0.6000
    epochs : 1111
    Epoch 1/1
    0s - loss: 0.9254 - acc: 0.5600
    epochs : 1112
    Epoch 1/1
    0s - loss: 0.5606 - acc: 0.7800
    epochs : 1113
    Epoch 1/1
    0s - loss: 0.3549 - acc: 0.9200
    epochs : 1114
    Epoch 1/1
    0s - loss: 0.3815 - acc: 0.8400
    epochs : 1115
    Epoch 1/1
    0s - loss: 0.1632 - acc: 0.9400
    epochs : 1116
    Epoch 1/1
    0s - loss: 0.1146 - acc: 0.9400
    epochs : 1117
    Epoch 1/1
    0s - loss: 0.0855 - acc: 0.9800
    epochs : 1118
    Epoch 1/1
    0s - loss: 0.0772 - acc: 0.9600
    epochs : 1119
    Epoch 1/1
    0s - loss: 0.0656 - acc: 1.0000
    epochs : 1120
    Epoch 1/1
    0s - loss: 0.0567 - acc: 1.0000
    epochs : 1121
    Epoch 1/1
    0s - loss: 0.0476 - acc: 1.0000
    epochs : 1122
    Epoch 1/1
    0s - loss: 0.0386 - acc: 1.0000
    epochs : 1123
    Epoch 1/1
    0s - loss: 0.0295 - acc: 1.0000
    epochs : 1124
    Epoch 1/1
    0s - loss: 0.0256 - acc: 1.0000
    epochs : 1125
    Epoch 1/1
    0s - loss: 0.0224 - acc: 1.0000
    epochs : 1126
    Epoch 1/1
    0s - loss: 0.0244 - acc: 1.0000
    epochs : 1127
    Epoch 1/1
    0s - loss: 0.0202 - acc: 1.0000
    epochs : 1128
    Epoch 1/1
    0s - loss: 0.0176 - acc: 1.0000
    epochs : 1129
    Epoch 1/1
    0s - loss: 0.0898 - acc: 0.9800
    epochs : 1130
    Epoch 1/1
    0s - loss: 0.0786 - acc: 0.9800
    epochs : 1131
    Epoch 1/1
    0s - loss: 0.1381 - acc: 0.9800
    epochs : 1132
    Epoch 1/1
    0s - loss: 0.8053 - acc: 0.7000
    epochs : 1133
    Epoch 1/1
    0s - loss: 0.7304 - acc: 0.8000
    epochs : 1134
    Epoch 1/1
    0s - loss: 0.5813 - acc: 0.7800
    epochs : 1135
    Epoch 1/1
    0s - loss: 0.5074 - acc: 0.8400
    epochs : 1136
    Epoch 1/1
    0s - loss: 0.2036 - acc: 0.9400
    epochs : 1137
    Epoch 1/1
    0s - loss: 0.1502 - acc: 0.9000
    epochs : 1138
    Epoch 1/1
    0s - loss: 0.0795 - acc: 1.0000
    epochs : 1139
    Epoch 1/1
    0s - loss: 0.0449 - acc: 1.0000
    epochs : 1140
    Epoch 1/1
    0s - loss: 0.0421 - acc: 1.0000
    epochs : 1141
    Epoch 1/1
    0s - loss: 0.0311 - acc: 1.0000
    epochs : 1142
    Epoch 1/1
    0s - loss: 0.0263 - acc: 1.0000
    epochs : 1143
    Epoch 1/1
    0s - loss: 0.0209 - acc: 1.0000
    epochs : 1144
    Epoch 1/1
    0s - loss: 0.0174 - acc: 1.0000
    epochs : 1145
    Epoch 1/1
    0s - loss: 0.0155 - acc: 1.0000
    epochs : 1146
    Epoch 1/1
    0s - loss: 0.0139 - acc: 1.0000
    epochs : 1147
    Epoch 1/1
    0s - loss: 0.0126 - acc: 1.0000
    epochs : 1148
    Epoch 1/1
    0s - loss: 0.0116 - acc: 1.0000
    epochs : 1149
    Epoch 1/1
    0s - loss: 0.0106 - acc: 1.0000
    epochs : 1150
    Epoch 1/1
    0s - loss: 0.0099 - acc: 1.0000
    epochs : 1151
    Epoch 1/1
    0s - loss: 0.0091 - acc: 1.0000
    epochs : 1152
    Epoch 1/1
    0s - loss: 0.0085 - acc: 1.0000
    epochs : 1153
    Epoch 1/1
    0s - loss: 0.0079 - acc: 1.0000
    epochs : 1154
    Epoch 1/1
    0s - loss: 0.0074 - acc: 1.0000
    epochs : 1155
    Epoch 1/1
    0s - loss: 0.0069 - acc: 1.0000
    epochs : 1156
    Epoch 1/1
    0s - loss: 0.0065 - acc: 1.0000
    epochs : 1157
    Epoch 1/1
    0s - loss: 0.0061 - acc: 1.0000
    epochs : 1158
    Epoch 1/1
    0s - loss: 0.0057 - acc: 1.0000
    epochs : 1159
    Epoch 1/1
    0s - loss: 0.0054 - acc: 1.0000
    epochs : 1160
    Epoch 1/1
    0s - loss: 0.0051 - acc: 1.0000
    epochs : 1161
    Epoch 1/1
    0s - loss: 0.0048 - acc: 1.0000
    epochs : 1162
    Epoch 1/1
    0s - loss: 0.0045 - acc: 1.0000
    epochs : 1163
    Epoch 1/1
    0s - loss: 0.0043 - acc: 1.0000
    epochs : 1164
    Epoch 1/1
    0s - loss: 0.0042 - acc: 1.0000
    epochs : 1165
    Epoch 1/1
    0s - loss: 0.0040 - acc: 1.0000
    epochs : 1166
    Epoch 1/1
    0s - loss: 0.0038 - acc: 1.0000
    epochs : 1167
    Epoch 1/1
    0s - loss: 0.0036 - acc: 1.0000
    epochs : 1168
    Epoch 1/1
    0s - loss: 0.0035 - acc: 1.0000
    epochs : 1169
    Epoch 1/1
    0s - loss: 0.0034 - acc: 1.0000
    epochs : 1170
    Epoch 1/1
    0s - loss: 0.0033 - acc: 1.0000
    epochs : 1171
    Epoch 1/1
    0s - loss: 0.0031 - acc: 1.0000
    epochs : 1172
    Epoch 1/1
    0s - loss: 0.0030 - acc: 1.0000
    epochs : 1173
    Epoch 1/1
    0s - loss: 0.0029 - acc: 1.0000
    epochs : 1174
    Epoch 1/1
    0s - loss: 0.0028 - acc: 1.0000
    epochs : 1175
    Epoch 1/1
    0s - loss: 0.0027 - acc: 1.0000
    epochs : 1176
    Epoch 1/1
    0s - loss: 0.0026 - acc: 1.0000
    epochs : 1177
    Epoch 1/1
    0s - loss: 0.0025 - acc: 1.0000
    epochs : 1178
    Epoch 1/1
    0s - loss: 0.0024 - acc: 1.0000
    epochs : 1179
    Epoch 1/1
    0s - loss: 0.0023 - acc: 1.0000
    epochs : 1180
    Epoch 1/1
    0s - loss: 0.0023 - acc: 1.0000
    epochs : 1181
    Epoch 1/1
    0s - loss: 0.0022 - acc: 1.0000
    epochs : 1182
    Epoch 1/1
    0s - loss: 0.0021 - acc: 1.0000
    epochs : 1183
    Epoch 1/1
    0s - loss: 0.0021 - acc: 1.0000
    epochs : 1184
    Epoch 1/1
    0s - loss: 0.0020 - acc: 1.0000
    epochs : 1185
    Epoch 1/1
    0s - loss: 0.0019 - acc: 1.0000
    epochs : 1186
    Epoch 1/1
    0s - loss: 0.0019 - acc: 1.0000
    epochs : 1187
    Epoch 1/1
    0s - loss: 0.0019 - acc: 1.0000
    epochs : 1188
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 1189
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 1190
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 1191
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 1192
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 1193
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 1194
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 1195
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1196
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1197
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1198
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1199
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1200
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1201
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1202
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1203
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1204
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1205
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1206
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1207
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1208
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1209
    Epoch 1/1
    0s - loss: 0.0010 - acc: 1.0000
    epochs : 1210
    Epoch 1/1
    0s - loss: 0.0010 - acc: 1.0000
    epochs : 1211
    Epoch 1/1
    0s - loss: 9.7338e-04 - acc: 1.0000
    epochs : 1212
    Epoch 1/1
    0s - loss: 9.4009e-04 - acc: 1.0000
    epochs : 1213
    Epoch 1/1
    0s - loss: 9.1489e-04 - acc: 1.0000
    epochs : 1214
    Epoch 1/1
    0s - loss: 8.8450e-04 - acc: 1.0000
    epochs : 1215
    Epoch 1/1
    0s - loss: 8.6172e-04 - acc: 1.0000
    epochs : 1216
    Epoch 1/1
    0s - loss: 8.3369e-04 - acc: 1.0000
    epochs : 1217
    Epoch 1/1
    0s - loss: 8.1071e-04 - acc: 1.0000
    epochs : 1218
    Epoch 1/1
    0s - loss: 7.8656e-04 - acc: 1.0000
    epochs : 1219
    Epoch 1/1
    0s - loss: 7.6606e-04 - acc: 1.0000
    epochs : 1220
    Epoch 1/1
    0s - loss: 7.4927e-04 - acc: 1.0000
    epochs : 1221
    Epoch 1/1
    0s - loss: 7.3037e-04 - acc: 1.0000
    epochs : 1222
    Epoch 1/1
    0s - loss: 7.1804e-04 - acc: 1.0000
    epochs : 1223
    Epoch 1/1
    0s - loss: 6.9047e-04 - acc: 1.0000
    epochs : 1224
    Epoch 1/1
    0s - loss: 7.4023e-04 - acc: 1.0000
    epochs : 1225
    Epoch 1/1
    0s - loss: 7.1344e-04 - acc: 1.0000
    epochs : 1226
    Epoch 1/1
    0s - loss: 9.6925e-04 - acc: 1.0000
    epochs : 1227
    Epoch 1/1
    0s - loss: 9.1933e-04 - acc: 1.0000
    epochs : 1228
    Epoch 1/1
    0s - loss: 0.0023 - acc: 1.0000
    epochs : 1229
    Epoch 1/1
    0s - loss: 0.5761 - acc: 0.7800
    epochs : 1230
    Epoch 1/1
    0s - loss: 1.6760 - acc: 0.6400
    epochs : 1231
    Epoch 1/1
    0s - loss: 2.6962 - acc: 0.3000
    epochs : 1232
    Epoch 1/1
    0s - loss: 1.6756 - acc: 0.5000
    epochs : 1233
    Epoch 1/1
    0s - loss: 1.5844 - acc: 0.4400
    epochs : 1234
    Epoch 1/1
    0s - loss: 1.3652 - acc: 0.5200
    epochs : 1235
    Epoch 1/1
    0s - loss: 1.1757 - acc: 0.5400
    epochs : 1236
    Epoch 1/1
    0s - loss: 1.3485 - acc: 0.4600
    epochs : 1237
    Epoch 1/1
    0s - loss: 1.0746 - acc: 0.6000
    epochs : 1238
    Epoch 1/1
    0s - loss: 0.9230 - acc: 0.6400
    epochs : 1239
    Epoch 1/1
    0s - loss: 1.1195 - acc: 0.5600
    epochs : 1240
    Epoch 1/1
    0s - loss: 1.0190 - acc: 0.6000
    epochs : 1241
    Epoch 1/1
    0s - loss: 0.8867 - acc: 0.6400
    epochs : 1242
    Epoch 1/1
    0s - loss: 0.7283 - acc: 0.7000
    epochs : 1243
    Epoch 1/1
    0s - loss: 0.7397 - acc: 0.6800
    epochs : 1244
    Epoch 1/1
    0s - loss: 0.5848 - acc: 0.6800
    epochs : 1245
    Epoch 1/1
    0s - loss: 0.5999 - acc: 0.8000
    epochs : 1246
    Epoch 1/1
    0s - loss: 0.5486 - acc: 0.7000
    epochs : 1247
    Epoch 1/1
    0s - loss: 0.3991 - acc: 0.8000
    epochs : 1248
    Epoch 1/1
    0s - loss: 0.3331 - acc: 0.8800
    epochs : 1249
    Epoch 1/1
    0s - loss: 0.2281 - acc: 0.9000
    epochs : 1250
    Epoch 1/1
    0s - loss: 0.1898 - acc: 0.9200
    epochs : 1251
    Epoch 1/1
    0s - loss: 0.1633 - acc: 0.9400
    epochs : 1252
    Epoch 1/1
    0s - loss: 0.1357 - acc: 0.9800
    epochs : 1253
    Epoch 1/1
    0s - loss: 0.1194 - acc: 0.9800
    epochs : 1254
    Epoch 1/1
    0s - loss: 0.0976 - acc: 1.0000
    epochs : 1255
    Epoch 1/1
    0s - loss: 0.0770 - acc: 1.0000
    epochs : 1256
    Epoch 1/1
    0s - loss: 0.0615 - acc: 1.0000
    epochs : 1257
    Epoch 1/1
    0s - loss: 0.1978 - acc: 0.9200
    epochs : 1258
    Epoch 1/1
    0s - loss: 0.8727 - acc: 0.8000
    epochs : 1259
    Epoch 1/1
    0s - loss: 0.2820 - acc: 0.8600
    epochs : 1260
    Epoch 1/1
    0s - loss: 0.2899 - acc: 0.8800
    epochs : 1261
    Epoch 1/1
    0s - loss: 0.7103 - acc: 0.7400
    epochs : 1262
    Epoch 1/1
    0s - loss: 0.4239 - acc: 0.8600
    epochs : 1263
    Epoch 1/1
    0s - loss: 0.3049 - acc: 0.8800
    epochs : 1264
    Epoch 1/1
    0s - loss: 0.1240 - acc: 0.9600
    epochs : 1265
    Epoch 1/1
    0s - loss: 0.0667 - acc: 1.0000
    epochs : 1266
    Epoch 1/1
    0s - loss: 0.0778 - acc: 1.0000
    epochs : 1267
    Epoch 1/1
    0s - loss: 0.0543 - acc: 1.0000
    epochs : 1268
    Epoch 1/1
    0s - loss: 0.0944 - acc: 0.9600
    epochs : 1269
    Epoch 1/1
    0s - loss: 0.3341 - acc: 0.8600
    epochs : 1270
    Epoch 1/1
    0s - loss: 0.1248 - acc: 0.9800
    epochs : 1271
    Epoch 1/1
    0s - loss: 0.3195 - acc: 0.8800
    epochs : 1272
    Epoch 1/1
    0s - loss: 0.1904 - acc: 0.9200
    epochs : 1273
    Epoch 1/1
    0s - loss: 0.0797 - acc: 1.0000
    epochs : 1274
    Epoch 1/1
    0s - loss: 0.0301 - acc: 1.0000
    epochs : 1275
    Epoch 1/1
    0s - loss: 0.0198 - acc: 1.0000
    epochs : 1276
    Epoch 1/1
    0s - loss: 0.0157 - acc: 1.0000
    epochs : 1277
    Epoch 1/1
    0s - loss: 0.0142 - acc: 1.0000
    epochs : 1278
    Epoch 1/1
    0s - loss: 0.0133 - acc: 1.0000
    epochs : 1279
    Epoch 1/1
    0s - loss: 0.0125 - acc: 1.0000
    epochs : 1280
    Epoch 1/1
    0s - loss: 0.0111 - acc: 1.0000
    epochs : 1281
    Epoch 1/1
    0s - loss: 0.0099 - acc: 1.0000
    epochs : 1282
    Epoch 1/1
    0s - loss: 0.0090 - acc: 1.0000
    epochs : 1283
    Epoch 1/1
    0s - loss: 0.0081 - acc: 1.0000
    epochs : 1284
    Epoch 1/1
    0s - loss: 0.0074 - acc: 1.0000
    epochs : 1285
    Epoch 1/1
    0s - loss: 0.0067 - acc: 1.0000
    epochs : 1286
    Epoch 1/1
    0s - loss: 0.0062 - acc: 1.0000
    epochs : 1287
    Epoch 1/1
    0s - loss: 0.0058 - acc: 1.0000
    epochs : 1288
    Epoch 1/1
    0s - loss: 0.0055 - acc: 1.0000
    epochs : 1289
    Epoch 1/1
    0s - loss: 0.0052 - acc: 1.0000
    epochs : 1290
    Epoch 1/1
    0s - loss: 0.0049 - acc: 1.0000
    epochs : 1291
    Epoch 1/1
    0s - loss: 0.0047 - acc: 1.0000
    epochs : 1292
    Epoch 1/1
    0s - loss: 0.0044 - acc: 1.0000
    epochs : 1293
    Epoch 1/1
    0s - loss: 0.0042 - acc: 1.0000
    epochs : 1294
    Epoch 1/1
    0s - loss: 0.0040 - acc: 1.0000
    epochs : 1295
    Epoch 1/1
    0s - loss: 0.0039 - acc: 1.0000
    epochs : 1296
    Epoch 1/1
    0s - loss: 0.0037 - acc: 1.0000
    epochs : 1297
    Epoch 1/1
    0s - loss: 0.0036 - acc: 1.0000
    epochs : 1298
    Epoch 1/1
    0s - loss: 0.0034 - acc: 1.0000
    epochs : 1299
    Epoch 1/1
    0s - loss: 0.0033 - acc: 1.0000
    epochs : 1300
    Epoch 1/1
    0s - loss: 0.0031 - acc: 1.0000
    epochs : 1301
    Epoch 1/1
    0s - loss: 0.0030 - acc: 1.0000
    epochs : 1302
    Epoch 1/1
    0s - loss: 0.0028 - acc: 1.0000
    epochs : 1303
    Epoch 1/1
    0s - loss: 0.0027 - acc: 1.0000
    epochs : 1304
    Epoch 1/1
    0s - loss: 0.0026 - acc: 1.0000
    epochs : 1305
    Epoch 1/1
    0s - loss: 0.0025 - acc: 1.0000
    epochs : 1306
    Epoch 1/1
    0s - loss: 0.0024 - acc: 1.0000
    epochs : 1307
    Epoch 1/1
    0s - loss: 0.0024 - acc: 1.0000
    epochs : 1308
    Epoch 1/1
    0s - loss: 0.0023 - acc: 1.0000
    epochs : 1309
    Epoch 1/1
    0s - loss: 0.0022 - acc: 1.0000
    epochs : 1310
    Epoch 1/1
    0s - loss: 0.0021 - acc: 1.0000
    epochs : 1311
    Epoch 1/1
    0s - loss: 0.0020 - acc: 1.0000
    epochs : 1312
    Epoch 1/1
    0s - loss: 0.0020 - acc: 1.0000
    epochs : 1313
    Epoch 1/1
    0s - loss: 0.0019 - acc: 1.0000
    epochs : 1314
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 1315
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 1316
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 1317
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 1318
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 1319
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 1320
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1321
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1322
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1323
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1324
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1325
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1326
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1327
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1328
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1329
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1330
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1331
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1332
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1333
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1334
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1335
    Epoch 1/1
    0s - loss: 0.0010 - acc: 1.0000
    epochs : 1336
    Epoch 1/1
    0s - loss: 0.0010 - acc: 1.0000
    epochs : 1337
    Epoch 1/1
    0s - loss: 9.7151e-04 - acc: 1.0000
    epochs : 1338
    Epoch 1/1
    0s - loss: 9.3755e-04 - acc: 1.0000
    epochs : 1339
    Epoch 1/1
    0s - loss: 9.0644e-04 - acc: 1.0000
    epochs : 1340
    Epoch 1/1
    0s - loss: 8.7469e-04 - acc: 1.0000
    epochs : 1341
    Epoch 1/1
    0s - loss: 8.5519e-04 - acc: 1.0000
    epochs : 1342
    Epoch 1/1
    0s - loss: 8.3839e-04 - acc: 1.0000
    epochs : 1343
    Epoch 1/1
    0s - loss: 8.2284e-04 - acc: 1.0000
    epochs : 1344
    Epoch 1/1
    0s - loss: 8.0636e-04 - acc: 1.0000
    epochs : 1345
    Epoch 1/1
    0s - loss: 7.6628e-04 - acc: 1.0000
    epochs : 1346
    Epoch 1/1
    0s - loss: 7.1038e-04 - acc: 1.0000
    epochs : 1347
    Epoch 1/1
    0s - loss: 6.8757e-04 - acc: 1.0000
    epochs : 1348
    Epoch 1/1
    0s - loss: 6.5321e-04 - acc: 1.0000
    epochs : 1349
    Epoch 1/1
    0s - loss: 6.1616e-04 - acc: 1.0000
    epochs : 1350
    Epoch 1/1
    0s - loss: 6.0202e-04 - acc: 1.0000
    epochs : 1351
    Epoch 1/1
    0s - loss: 6.1555e-04 - acc: 1.0000
    epochs : 1352
    Epoch 1/1
    0s - loss: 0.0664 - acc: 0.9800
    epochs : 1353
    Epoch 1/1
    0s - loss: 1.8611 - acc: 0.5800
    epochs : 1354
    Epoch 1/1
    0s - loss: 1.6482 - acc: 0.5000
    epochs : 1355
    Epoch 1/1
    0s - loss: 1.8889 - acc: 0.4800
    epochs : 1356
    Epoch 1/1
    0s - loss: 0.6231 - acc: 0.7800
    epochs : 1357
    Epoch 1/1
    0s - loss: 0.5451 - acc: 0.8200
    epochs : 1358
    Epoch 1/1
    0s - loss: 1.1654 - acc: 0.7400
    epochs : 1359
    Epoch 1/1
    0s - loss: 0.9210 - acc: 0.6200
    epochs : 1360
    Epoch 1/1
    0s - loss: 0.5834 - acc: 0.8000
    epochs : 1361
    Epoch 1/1
    0s - loss: 0.9729 - acc: 0.8000
    epochs : 1362
    Epoch 1/1
    0s - loss: 0.3645 - acc: 0.8400
    epochs : 1363
    Epoch 1/1
    0s - loss: 0.2641 - acc: 0.9200
    epochs : 1364
    Epoch 1/1
    0s - loss: 0.2744 - acc: 0.9000
    epochs : 1365
    Epoch 1/1
    0s - loss: 0.1374 - acc: 1.0000
    epochs : 1366
    Epoch 1/1
    0s - loss: 0.1090 - acc: 0.9800
    epochs : 1367
    Epoch 1/1
    0s - loss: 0.1061 - acc: 1.0000
    epochs : 1368
    Epoch 1/1
    0s - loss: 0.0549 - acc: 1.0000
    epochs : 1369
    Epoch 1/1
    0s - loss: 0.0471 - acc: 1.0000
    epochs : 1370
    Epoch 1/1
    0s - loss: 0.0365 - acc: 1.0000
    epochs : 1371
    Epoch 1/1
    0s - loss: 0.0341 - acc: 1.0000
    epochs : 1372
    Epoch 1/1
    0s - loss: 0.0303 - acc: 1.0000
    epochs : 1373
    Epoch 1/1
    0s - loss: 0.0259 - acc: 1.0000
    epochs : 1374
    Epoch 1/1
    0s - loss: 0.0259 - acc: 1.0000
    epochs : 1375
    Epoch 1/1
    0s - loss: 0.0235 - acc: 1.0000
    epochs : 1376
    Epoch 1/1
    0s - loss: 0.0197 - acc: 1.0000
    epochs : 1377
    Epoch 1/1
    0s - loss: 0.0169 - acc: 1.0000
    epochs : 1378
    Epoch 1/1
    0s - loss: 0.0167 - acc: 1.0000
    epochs : 1379
    Epoch 1/1
    0s - loss: 0.0141 - acc: 1.0000
    epochs : 1380
    Epoch 1/1
    0s - loss: 0.0134 - acc: 1.0000
    epochs : 1381
    Epoch 1/1
    0s - loss: 0.0128 - acc: 1.0000
    epochs : 1382
    Epoch 1/1
    0s - loss: 0.0121 - acc: 1.0000
    epochs : 1383
    Epoch 1/1
    0s - loss: 0.0111 - acc: 1.0000
    epochs : 1384
    Epoch 1/1
    0s - loss: 0.0104 - acc: 1.0000
    epochs : 1385
    Epoch 1/1
    0s - loss: 0.0098 - acc: 1.0000
    epochs : 1386
    Epoch 1/1
    0s - loss: 0.0090 - acc: 1.0000
    epochs : 1387
    Epoch 1/1
    0s - loss: 0.0083 - acc: 1.0000
    epochs : 1388
    Epoch 1/1
    0s - loss: 0.0076 - acc: 1.0000
    epochs : 1389
    Epoch 1/1
    0s - loss: 0.0070 - acc: 1.0000
    epochs : 1390
    Epoch 1/1
    0s - loss: 0.0066 - acc: 1.0000
    epochs : 1391
    Epoch 1/1
    0s - loss: 0.0062 - acc: 1.0000
    epochs : 1392
    Epoch 1/1
    0s - loss: 0.0059 - acc: 1.0000
    epochs : 1393
    Epoch 1/1
    0s - loss: 0.0056 - acc: 1.0000
    epochs : 1394
    Epoch 1/1
    0s - loss: 0.0053 - acc: 1.0000
    epochs : 1395
    Epoch 1/1
    0s - loss: 0.0051 - acc: 1.0000
    epochs : 1396
    Epoch 1/1
    0s - loss: 0.0048 - acc: 1.0000
    epochs : 1397
    Epoch 1/1
    0s - loss: 0.0046 - acc: 1.0000
    epochs : 1398
    Epoch 1/1
    0s - loss: 0.0043 - acc: 1.0000
    epochs : 1399
    Epoch 1/1
    0s - loss: 0.0042 - acc: 1.0000
    epochs : 1400
    Epoch 1/1
    0s - loss: 0.0039 - acc: 1.0000
    epochs : 1401
    Epoch 1/1
    0s - loss: 0.0038 - acc: 1.0000
    epochs : 1402
    Epoch 1/1
    0s - loss: 0.0036 - acc: 1.0000
    epochs : 1403
    Epoch 1/1
    0s - loss: 0.0034 - acc: 1.0000
    epochs : 1404
    Epoch 1/1
    0s - loss: 0.0033 - acc: 1.0000
    epochs : 1405
    Epoch 1/1
    0s - loss: 0.0032 - acc: 1.0000
    epochs : 1406
    Epoch 1/1
    0s - loss: 0.0030 - acc: 1.0000
    epochs : 1407
    Epoch 1/1
    0s - loss: 0.0029 - acc: 1.0000
    epochs : 1408
    Epoch 1/1
    0s - loss: 0.0028 - acc: 1.0000
    epochs : 1409
    Epoch 1/1
    0s - loss: 0.0027 - acc: 1.0000
    epochs : 1410
    Epoch 1/1
    0s - loss: 0.0026 - acc: 1.0000
    epochs : 1411
    Epoch 1/1
    0s - loss: 0.0025 - acc: 1.0000
    epochs : 1412
    Epoch 1/1
    0s - loss: 0.0024 - acc: 1.0000
    epochs : 1413
    Epoch 1/1
    0s - loss: 0.0023 - acc: 1.0000
    epochs : 1414
    Epoch 1/1
    0s - loss: 0.0022 - acc: 1.0000
    epochs : 1415
    Epoch 1/1
    0s - loss: 0.0021 - acc: 1.0000
    epochs : 1416
    Epoch 1/1
    0s - loss: 0.0021 - acc: 1.0000
    epochs : 1417
    Epoch 1/1
    0s - loss: 0.0020 - acc: 1.0000
    epochs : 1418
    Epoch 1/1
    0s - loss: 0.0020 - acc: 1.0000
    epochs : 1419
    Epoch 1/1
    0s - loss: 0.0019 - acc: 1.0000
    epochs : 1420
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 1421
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 1422
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 1423
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 1424
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 1425
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 1426
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1427
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1428
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1429
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1430
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1431
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1432
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1433
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1434
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1435
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1436
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1437
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1438
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1439
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1440
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1441
    Epoch 1/1
    0s - loss: 0.0010 - acc: 1.0000
    epochs : 1442
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1443
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1444
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1445
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1446
    Epoch 1/1
    0s - loss: 9.5604e-04 - acc: 1.0000
    epochs : 1447
    Epoch 1/1
    0s - loss: 9.8795e-04 - acc: 1.0000
    epochs : 1448
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1449
    Epoch 1/1
    0s - loss: 9.1345e-04 - acc: 1.0000
    epochs : 1450
    Epoch 1/1
    0s - loss: 8.6601e-04 - acc: 1.0000
    epochs : 1451
    Epoch 1/1
    0s - loss: 8.3767e-04 - acc: 1.0000
    epochs : 1452
    Epoch 1/1
    0s - loss: 8.4475e-04 - acc: 1.0000
    epochs : 1453
    Epoch 1/1
    0s - loss: 7.8500e-04 - acc: 1.0000
    epochs : 1454
    Epoch 1/1
    0s - loss: 7.7939e-04 - acc: 1.0000
    epochs : 1455
    Epoch 1/1
    0s - loss: 7.3049e-04 - acc: 1.0000
    epochs : 1456
    Epoch 1/1
    0s - loss: 7.3134e-04 - acc: 1.0000
    epochs : 1457
    Epoch 1/1
    0s - loss: 6.8119e-04 - acc: 1.0000
    epochs : 1458
    Epoch 1/1
    0s - loss: 6.3448e-04 - acc: 1.0000
    epochs : 1459
    Epoch 1/1
    0s - loss: 6.1577e-04 - acc: 1.0000
    epochs : 1460
    Epoch 1/1
    0s - loss: 5.9232e-04 - acc: 1.0000
    epochs : 1461
    Epoch 1/1
    0s - loss: 5.6611e-04 - acc: 1.0000
    epochs : 1462
    Epoch 1/1
    0s - loss: 5.4498e-04 - acc: 1.0000
    epochs : 1463
    Epoch 1/1
    0s - loss: 5.2128e-04 - acc: 1.0000
    epochs : 1464
    Epoch 1/1
    0s - loss: 5.0735e-04 - acc: 1.0000
    epochs : 1465
    Epoch 1/1
    0s - loss: 4.8488e-04 - acc: 1.0000
    epochs : 1466
    Epoch 1/1
    0s - loss: 4.7578e-04 - acc: 1.0000
    epochs : 1467
    Epoch 1/1
    0s - loss: 4.5443e-04 - acc: 1.0000
    epochs : 1468
    Epoch 1/1
    0s - loss: 4.4260e-04 - acc: 1.0000
    epochs : 1469
    Epoch 1/1
    0s - loss: 4.2633e-04 - acc: 1.0000
    epochs : 1470
    Epoch 1/1
    0s - loss: 4.1153e-04 - acc: 1.0000
    epochs : 1471
    Epoch 1/1
    0s - loss: 3.9604e-04 - acc: 1.0000
    epochs : 1472
    Epoch 1/1
    0s - loss: 3.8089e-04 - acc: 1.0000
    epochs : 1473
    Epoch 1/1
    0s - loss: 3.6651e-04 - acc: 1.0000
    epochs : 1474
    Epoch 1/1
    0s - loss: 3.5385e-04 - acc: 1.0000
    epochs : 1475
    Epoch 1/1
    0s - loss: 3.4123e-04 - acc: 1.0000
    epochs : 1476
    Epoch 1/1
    0s - loss: 3.3133e-04 - acc: 1.0000
    epochs : 1477
    Epoch 1/1
    0s - loss: 3.2013e-04 - acc: 1.0000
    epochs : 1478
    Epoch 1/1
    0s - loss: 3.1026e-04 - acc: 1.0000
    epochs : 1479
    Epoch 1/1
    0s - loss: 2.9902e-04 - acc: 1.0000
    epochs : 1480
    Epoch 1/1
    0s - loss: 2.8977e-04 - acc: 1.0000
    epochs : 1481
    Epoch 1/1
    0s - loss: 2.8361e-04 - acc: 1.0000
    epochs : 1482
    Epoch 1/1
    0s - loss: 2.7676e-04 - acc: 1.0000
    epochs : 1483
    Epoch 1/1
    0s - loss: 2.7961e-04 - acc: 1.0000
    epochs : 1484
    Epoch 1/1
    0s - loss: 2.7226e-04 - acc: 1.0000
    epochs : 1485
    Epoch 1/1
    0s - loss: 2.8488e-04 - acc: 1.0000
    epochs : 1486
    Epoch 1/1
    0s - loss: 2.6641e-04 - acc: 1.0000
    epochs : 1487
    Epoch 1/1
    0s - loss: 2.3784e-04 - acc: 1.0000
    epochs : 1488
    Epoch 1/1
    0s - loss: 2.3179e-04 - acc: 1.0000
    epochs : 1489
    Epoch 1/1
    0s - loss: 2.2128e-04 - acc: 1.0000
    epochs : 1490
    Epoch 1/1
    0s - loss: 2.1410e-04 - acc: 1.0000
    epochs : 1491
    Epoch 1/1
    0s - loss: 2.1013e-04 - acc: 1.0000
    epochs : 1492
    Epoch 1/1
    0s - loss: 2.0383e-04 - acc: 1.0000
    epochs : 1493
    Epoch 1/1
    0s - loss: 2.0462e-04 - acc: 1.0000
    epochs : 1494
    Epoch 1/1
    0s - loss: 1.9577e-04 - acc: 1.0000
    epochs : 1495
    Epoch 1/1
    0s - loss: 2.0095e-04 - acc: 1.0000
    epochs : 1496
    Epoch 1/1
    0s - loss: 1.9035e-04 - acc: 1.0000
    epochs : 1497
    Epoch 1/1
    0s - loss: 1.8765e-04 - acc: 1.0000
    epochs : 1498
    Epoch 1/1
    0s - loss: 1.7843e-04 - acc: 1.0000
    epochs : 1499
    Epoch 1/1
    0s - loss: 1.7469e-04 - acc: 1.0000
    epochs : 1500
    Epoch 1/1
    0s - loss: 1.6626e-04 - acc: 1.0000
    epochs : 1501
    Epoch 1/1
    0s - loss: 1.6366e-04 - acc: 1.0000
    epochs : 1502
    Epoch 1/1
    0s - loss: 1.5587e-04 - acc: 1.0000
    epochs : 1503
    Epoch 1/1
    0s - loss: 1.5610e-04 - acc: 1.0000
    epochs : 1504
    Epoch 1/1
    0s - loss: 1.4734e-04 - acc: 1.0000
    epochs : 1505
    Epoch 1/1
    0s - loss: 1.4461e-04 - acc: 1.0000
    epochs : 1506
    Epoch 1/1
    0s - loss: 1.4102e-04 - acc: 1.0000
    epochs : 1507
    Epoch 1/1
    0s - loss: 1.3697e-04 - acc: 1.0000
    epochs : 1508
    Epoch 1/1
    0s - loss: 1.3299e-04 - acc: 1.0000
    epochs : 1509
    Epoch 1/1
    0s - loss: 1.2793e-04 - acc: 1.0000
    epochs : 1510
    Epoch 1/1
    0s - loss: 1.2600e-04 - acc: 1.0000
    epochs : 1511
    Epoch 1/1
    0s - loss: 1.2186e-04 - acc: 1.0000
    epochs : 1512
    Epoch 1/1
    0s - loss: 1.2081e-04 - acc: 1.0000
    epochs : 1513
    Epoch 1/1
    0s - loss: 1.2345e-04 - acc: 1.0000
    epochs : 1514
    Epoch 1/1
    0s - loss: 1.2236e-04 - acc: 1.0000
    epochs : 1515
    Epoch 1/1
    0s - loss: 0.1055 - acc: 0.9600
    epochs : 1516
    Epoch 1/1
    0s - loss: 1.6470 - acc: 0.5400
    epochs : 1517
    Epoch 1/1
    0s - loss: 1.6735 - acc: 0.6000
    epochs : 1518
    Epoch 1/1
    0s - loss: 1.4801 - acc: 0.5400
    epochs : 1519
    Epoch 1/1
    0s - loss: 1.2805 - acc: 0.5600
    epochs : 1520
    Epoch 1/1
    0s - loss: 1.2159 - acc: 0.4800
    epochs : 1521
    Epoch 1/1
    0s - loss: 0.9041 - acc: 0.6400
    epochs : 1522
    Epoch 1/1
    0s - loss: 0.8472 - acc: 0.7000
    epochs : 1523
    Epoch 1/1
    0s - loss: 0.7663 - acc: 0.7200
    epochs : 1524
    Epoch 1/1
    0s - loss: 0.4919 - acc: 0.8400
    epochs : 1525
    Epoch 1/1
    0s - loss: 0.4524 - acc: 0.8200
    epochs : 1526
    Epoch 1/1
    0s - loss: 0.4447 - acc: 0.7800
    epochs : 1527
    Epoch 1/1
    0s - loss: 0.8564 - acc: 0.7000
    epochs : 1528
    Epoch 1/1
    0s - loss: 0.4543 - acc: 0.8800
    epochs : 1529
    Epoch 1/1
    0s - loss: 0.4921 - acc: 0.8200
    epochs : 1530
    Epoch 1/1
    0s - loss: 0.4681 - acc: 0.7600
    epochs : 1531
    Epoch 1/1
    0s - loss: 0.3551 - acc: 0.8600
    epochs : 1532
    Epoch 1/1
    0s - loss: 0.2901 - acc: 0.9400
    epochs : 1533
    Epoch 1/1
    0s - loss: 0.2627 - acc: 0.9000
    epochs : 1534
    Epoch 1/1
    0s - loss: 0.2055 - acc: 0.9400
    epochs : 1535
    Epoch 1/1
    0s - loss: 0.2657 - acc: 0.8800
    epochs : 1536
    Epoch 1/1
    0s - loss: 0.1862 - acc: 0.9200
    epochs : 1537
    Epoch 1/1
    0s - loss: 1.2431 - acc: 0.6400
    epochs : 1538
    Epoch 1/1
    0s - loss: 0.6319 - acc: 0.7600
    epochs : 1539
    Epoch 1/1
    0s - loss: 0.3153 - acc: 0.8800
    epochs : 1540
    Epoch 1/1
    0s - loss: 0.4549 - acc: 0.8600
    epochs : 1541
    Epoch 1/1
    0s - loss: 0.2998 - acc: 0.8800
    epochs : 1542
    Epoch 1/1
    0s - loss: 0.1122 - acc: 1.0000
    epochs : 1543
    Epoch 1/1
    0s - loss: 0.0997 - acc: 1.0000
    epochs : 1544
    Epoch 1/1
    0s - loss: 0.0809 - acc: 1.0000
    epochs : 1545
    Epoch 1/1
    0s - loss: 0.0655 - acc: 1.0000
    epochs : 1546
    Epoch 1/1
    0s - loss: 0.0519 - acc: 1.0000
    epochs : 1547
    Epoch 1/1
    0s - loss: 0.0391 - acc: 1.0000
    epochs : 1548
    Epoch 1/1
    0s - loss: 0.0322 - acc: 1.0000
    epochs : 1549
    Epoch 1/1
    0s - loss: 0.0284 - acc: 1.0000
    epochs : 1550
    Epoch 1/1
    0s - loss: 0.0252 - acc: 1.0000
    epochs : 1551
    Epoch 1/1
    0s - loss: 0.0225 - acc: 1.0000
    epochs : 1552
    Epoch 1/1
    0s - loss: 0.0202 - acc: 1.0000
    epochs : 1553
    Epoch 1/1
    0s - loss: 0.0183 - acc: 1.0000
    epochs : 1554
    Epoch 1/1
    0s - loss: 0.0169 - acc: 1.0000
    epochs : 1555
    Epoch 1/1
    0s - loss: 0.0154 - acc: 1.0000
    epochs : 1556
    Epoch 1/1
    0s - loss: 0.0142 - acc: 1.0000
    epochs : 1557
    Epoch 1/1
    0s - loss: 0.0132 - acc: 1.0000
    epochs : 1558
    Epoch 1/1
    0s - loss: 0.0120 - acc: 1.0000
    epochs : 1559
    Epoch 1/1
    0s - loss: 0.0107 - acc: 1.0000
    epochs : 1560
    Epoch 1/1
    0s - loss: 0.0096 - acc: 1.0000
    epochs : 1561
    Epoch 1/1
    0s - loss: 0.0087 - acc: 1.0000
    epochs : 1562
    Epoch 1/1
    0s - loss: 0.0079 - acc: 1.0000
    epochs : 1563
    Epoch 1/1
    0s - loss: 0.0072 - acc: 1.0000
    epochs : 1564
    Epoch 1/1
    0s - loss: 0.0066 - acc: 1.0000
    epochs : 1565
    Epoch 1/1
    0s - loss: 0.0062 - acc: 1.0000
    epochs : 1566
    Epoch 1/1
    0s - loss: 0.0058 - acc: 1.0000
    epochs : 1567
    Epoch 1/1
    0s - loss: 0.0055 - acc: 1.0000
    epochs : 1568
    Epoch 1/1
    0s - loss: 0.0052 - acc: 1.0000
    epochs : 1569
    Epoch 1/1
    0s - loss: 0.0050 - acc: 1.0000
    epochs : 1570
    Epoch 1/1
    0s - loss: 0.0048 - acc: 1.0000
    epochs : 1571
    Epoch 1/1
    0s - loss: 0.0046 - acc: 1.0000
    epochs : 1572
    Epoch 1/1
    0s - loss: 0.0044 - acc: 1.0000
    epochs : 1573
    Epoch 1/1
    0s - loss: 0.0043 - acc: 1.0000
    epochs : 1574
    Epoch 1/1
    0s - loss: 0.0041 - acc: 1.0000
    epochs : 1575
    Epoch 1/1
    0s - loss: 0.0039 - acc: 1.0000
    epochs : 1576
    Epoch 1/1
    0s - loss: 0.0037 - acc: 1.0000
    epochs : 1577
    Epoch 1/1
    0s - loss: 0.0035 - acc: 1.0000
    epochs : 1578
    Epoch 1/1
    0s - loss: 0.0033 - acc: 1.0000
    epochs : 1579
    Epoch 1/1
    0s - loss: 0.0031 - acc: 1.0000
    epochs : 1580
    Epoch 1/1
    0s - loss: 0.0029 - acc: 1.0000
    epochs : 1581
    Epoch 1/1
    0s - loss: 0.0028 - acc: 1.0000
    epochs : 1582
    Epoch 1/1
    0s - loss: 0.0027 - acc: 1.0000
    epochs : 1583
    Epoch 1/1
    0s - loss: 0.0026 - acc: 1.0000
    epochs : 1584
    Epoch 1/1
    0s - loss: 0.0026 - acc: 1.0000
    epochs : 1585
    Epoch 1/1
    0s - loss: 0.0026 - acc: 1.0000
    epochs : 1586
    Epoch 1/1
    0s - loss: 0.0026 - acc: 1.0000
    epochs : 1587
    Epoch 1/1
    0s - loss: 0.0026 - acc: 1.0000
    epochs : 1588
    Epoch 1/1
    0s - loss: 0.0025 - acc: 1.0000
    epochs : 1589
    Epoch 1/1
    0s - loss: 0.0023 - acc: 1.0000
    epochs : 1590
    Epoch 1/1
    0s - loss: 0.0021 - acc: 1.0000
    epochs : 1591
    Epoch 1/1
    0s - loss: 0.0019 - acc: 1.0000
    epochs : 1592
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 1593
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 1594
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 1595
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1596
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1597
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1598
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1599
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1600
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1601
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1602
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1603
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1604
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1605
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1606
    Epoch 1/1
    0s - loss: 0.0010 - acc: 1.0000
    epochs : 1607
    Epoch 1/1
    0s - loss: 9.8522e-04 - acc: 1.0000
    epochs : 1608
    Epoch 1/1
    0s - loss: 9.5640e-04 - acc: 1.0000
    epochs : 1609
    Epoch 1/1
    0s - loss: 9.2835e-04 - acc: 1.0000
    epochs : 1610
    Epoch 1/1
    0s - loss: 9.0077e-04 - acc: 1.0000
    epochs : 1611
    Epoch 1/1
    0s - loss: 8.7345e-04 - acc: 1.0000
    epochs : 1612
    Epoch 1/1
    0s - loss: 8.4645e-04 - acc: 1.0000
    epochs : 1613
    Epoch 1/1
    0s - loss: 8.2051e-04 - acc: 1.0000
    epochs : 1614
    Epoch 1/1
    0s - loss: 7.9580e-04 - acc: 1.0000
    epochs : 1615
    Epoch 1/1
    0s - loss: 7.7193e-04 - acc: 1.0000
    epochs : 1616
    Epoch 1/1
    0s - loss: 7.4952e-04 - acc: 1.0000
    epochs : 1617
    Epoch 1/1
    0s - loss: 7.2685e-04 - acc: 1.0000
    epochs : 1618
    Epoch 1/1
    0s - loss: 7.0438e-04 - acc: 1.0000
    epochs : 1619
    Epoch 1/1
    0s - loss: 6.8255e-04 - acc: 1.0000
    epochs : 1620
    Epoch 1/1
    0s - loss: 6.6248e-04 - acc: 1.0000
    epochs : 1621
    Epoch 1/1
    0s - loss: 6.4282e-04 - acc: 1.0000
    epochs : 1622
    Epoch 1/1
    0s - loss: 6.2390e-04 - acc: 1.0000
    epochs : 1623
    Epoch 1/1
    0s - loss: 6.0559e-04 - acc: 1.0000
    epochs : 1624
    Epoch 1/1
    0s - loss: 5.8800e-04 - acc: 1.0000
    epochs : 1625
    Epoch 1/1
    0s - loss: 5.7106e-04 - acc: 1.0000
    epochs : 1626
    Epoch 1/1
    0s - loss: 5.5479e-04 - acc: 1.0000
    epochs : 1627
    Epoch 1/1
    0s - loss: 5.3892e-04 - acc: 1.0000
    epochs : 1628
    Epoch 1/1
    0s - loss: 5.2429e-04 - acc: 1.0000
    epochs : 1629
    Epoch 1/1
    0s - loss: 5.1019e-04 - acc: 1.0000
    epochs : 1630
    Epoch 1/1
    0s - loss: 4.9739e-04 - acc: 1.0000
    epochs : 1631
    Epoch 1/1
    0s - loss: 4.8571e-04 - acc: 1.0000
    epochs : 1632
    Epoch 1/1
    0s - loss: 4.7412e-04 - acc: 1.0000
    epochs : 1633
    Epoch 1/1
    0s - loss: 4.6363e-04 - acc: 1.0000
    epochs : 1634
    Epoch 1/1
    0s - loss: 4.5392e-04 - acc: 1.0000
    epochs : 1635
    Epoch 1/1
    0s - loss: 4.4492e-04 - acc: 1.0000
    epochs : 1636
    Epoch 1/1
    0s - loss: 4.4215e-04 - acc: 1.0000
    epochs : 1637
    Epoch 1/1
    0s - loss: 4.2990e-04 - acc: 1.0000
    epochs : 1638
    Epoch 1/1
    0s - loss: 4.2564e-04 - acc: 1.0000
    epochs : 1639
    Epoch 1/1
    0s - loss: 4.2535e-04 - acc: 1.0000
    epochs : 1640
    Epoch 1/1
    0s - loss: 4.3003e-04 - acc: 1.0000
    epochs : 1641
    Epoch 1/1
    0s - loss: 4.0798e-04 - acc: 1.0000
    epochs : 1642
    Epoch 1/1
    0s - loss: 4.3034e-04 - acc: 1.0000
    epochs : 1643
    Epoch 1/1
    0s - loss: 4.1978e-04 - acc: 1.0000
    epochs : 1644
    Epoch 1/1
    0s - loss: 4.3963e-04 - acc: 1.0000
    epochs : 1645
    Epoch 1/1
    0s - loss: 5.2291e-04 - acc: 1.0000
    epochs : 1646
    Epoch 1/1
    0s - loss: 4.3820e-04 - acc: 1.0000
    epochs : 1647
    Epoch 1/1
    0s - loss: 6.9941e-04 - acc: 1.0000
    epochs : 1648
    Epoch 1/1
    0s - loss: 0.1029 - acc: 0.9600
    epochs : 1649
    Epoch 1/1
    0s - loss: 2.2793 - acc: 0.5200
    epochs : 1650
    Epoch 1/1
    0s - loss: 1.3817 - acc: 0.6000
    epochs : 1651
    Epoch 1/1
    0s - loss: 2.1992 - acc: 0.4400
    epochs : 1652
    Epoch 1/1
    0s - loss: 1.2902 - acc: 0.6200
    epochs : 1653
    Epoch 1/1
    0s - loss: 1.0793 - acc: 0.6600
    epochs : 1654
    Epoch 1/1
    0s - loss: 1.0209 - acc: 0.6200
    epochs : 1655
    Epoch 1/1
    0s - loss: 1.4676 - acc: 0.6000
    epochs : 1656
    Epoch 1/1
    0s - loss: 0.6134 - acc: 0.7800
    epochs : 1657
    Epoch 1/1
    0s - loss: 0.8946 - acc: 0.7000
    epochs : 1658
    Epoch 1/1
    0s - loss: 0.5378 - acc: 0.8000
    epochs : 1659
    Epoch 1/1
    0s - loss: 0.2660 - acc: 0.9400
    epochs : 1660
    Epoch 1/1
    0s - loss: 0.1976 - acc: 0.9600
    epochs : 1661
    Epoch 1/1
    0s - loss: 0.1205 - acc: 0.9800
    epochs : 1662
    Epoch 1/1
    0s - loss: 0.1572 - acc: 0.9800
    epochs : 1663
    Epoch 1/1
    0s - loss: 0.5323 - acc: 0.7600
    epochs : 1664
    Epoch 1/1
    0s - loss: 1.2572 - acc: 0.6000
    epochs : 1665
    Epoch 1/1
    0s - loss: 0.5434 - acc: 0.8200
    epochs : 1666
    Epoch 1/1
    0s - loss: 0.3222 - acc: 0.9000
    epochs : 1667
    Epoch 1/1
    0s - loss: 0.1343 - acc: 0.9800
    epochs : 1668
    Epoch 1/1
    0s - loss: 0.1630 - acc: 0.9800
    epochs : 1669
    Epoch 1/1
    0s - loss: 0.2503 - acc: 0.9400
    epochs : 1670
    Epoch 1/1
    0s - loss: 0.3039 - acc: 0.9200
    epochs : 1671
    Epoch 1/1
    0s - loss: 0.2851 - acc: 0.9000
    epochs : 1672
    Epoch 1/1
    0s - loss: 0.2717 - acc: 0.8800
    epochs : 1673
    Epoch 1/1
    0s - loss: 0.3856 - acc: 0.8400
    epochs : 1674
    Epoch 1/1
    0s - loss: 0.1277 - acc: 1.0000
    epochs : 1675
    Epoch 1/1
    0s - loss: 0.0963 - acc: 1.0000
    epochs : 1676
    Epoch 1/1
    0s - loss: 0.0605 - acc: 1.0000
    epochs : 1677
    Epoch 1/1
    0s - loss: 0.0303 - acc: 1.0000
    epochs : 1678
    Epoch 1/1
    0s - loss: 0.0533 - acc: 1.0000
    epochs : 1679
    Epoch 1/1
    0s - loss: 0.0265 - acc: 1.0000
    epochs : 1680
    Epoch 1/1
    0s - loss: 0.0265 - acc: 1.0000
    epochs : 1681
    Epoch 1/1
    0s - loss: 0.0135 - acc: 1.0000
    epochs : 1682
    Epoch 1/1
    0s - loss: 0.0119 - acc: 1.0000
    epochs : 1683
    Epoch 1/1
    0s - loss: 0.0115 - acc: 1.0000
    epochs : 1684
    Epoch 1/1
    0s - loss: 0.0127 - acc: 1.0000
    epochs : 1685
    Epoch 1/1
    0s - loss: 0.0131 - acc: 1.0000
    epochs : 1686
    Epoch 1/1
    0s - loss: 0.0110 - acc: 1.0000
    epochs : 1687
    Epoch 1/1
    0s - loss: 0.0088 - acc: 1.0000
    epochs : 1688
    Epoch 1/1
    0s - loss: 0.0080 - acc: 1.0000
    epochs : 1689
    Epoch 1/1
    0s - loss: 0.0086 - acc: 1.0000
    epochs : 1690
    Epoch 1/1
    0s - loss: 0.0078 - acc: 1.0000
    epochs : 1691
    Epoch 1/1
    0s - loss: 0.0086 - acc: 1.0000
    epochs : 1692
    Epoch 1/1
    0s - loss: 0.0067 - acc: 1.0000
    epochs : 1693
    Epoch 1/1
    0s - loss: 0.0076 - acc: 1.0000
    epochs : 1694
    Epoch 1/1
    0s - loss: 0.0059 - acc: 1.0000
    epochs : 1695
    Epoch 1/1
    0s - loss: 0.0051 - acc: 1.0000
    epochs : 1696
    Epoch 1/1
    0s - loss: 0.0048 - acc: 1.0000
    epochs : 1697
    Epoch 1/1
    0s - loss: 0.0048 - acc: 1.0000
    epochs : 1698
    Epoch 1/1
    0s - loss: 0.0043 - acc: 1.0000
    epochs : 1699
    Epoch 1/1
    0s - loss: 0.0042 - acc: 1.0000
    epochs : 1700
    Epoch 1/1
    0s - loss: 0.0039 - acc: 1.0000
    epochs : 1701
    Epoch 1/1
    0s - loss: 0.0038 - acc: 1.0000
    epochs : 1702
    Epoch 1/1
    0s - loss: 0.0035 - acc: 1.0000
    epochs : 1703
    Epoch 1/1
    0s - loss: 0.0033 - acc: 1.0000
    epochs : 1704
    Epoch 1/1
    0s - loss: 0.0032 - acc: 1.0000
    epochs : 1705
    Epoch 1/1
    0s - loss: 0.0030 - acc: 1.0000
    epochs : 1706
    Epoch 1/1
    0s - loss: 0.0029 - acc: 1.0000
    epochs : 1707
    Epoch 1/1
    0s - loss: 0.0027 - acc: 1.0000
    epochs : 1708
    Epoch 1/1
    0s - loss: 0.0026 - acc: 1.0000
    epochs : 1709
    Epoch 1/1
    0s - loss: 0.0024 - acc: 1.0000
    epochs : 1710
    Epoch 1/1
    0s - loss: 0.0024 - acc: 1.0000
    epochs : 1711
    Epoch 1/1
    0s - loss: 0.0022 - acc: 1.0000
    epochs : 1712
    Epoch 1/1
    0s - loss: 0.0022 - acc: 1.0000
    epochs : 1713
    Epoch 1/1
    0s - loss: 0.0020 - acc: 1.0000
    epochs : 1714
    Epoch 1/1
    0s - loss: 0.0020 - acc: 1.0000
    epochs : 1715
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 1716
    Epoch 1/1
    0s - loss: 0.0018 - acc: 1.0000
    epochs : 1717
    Epoch 1/1
    0s - loss: 0.0017 - acc: 1.0000
    epochs : 1718
    Epoch 1/1
    0s - loss: 0.0016 - acc: 1.0000
    epochs : 1719
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1720
    Epoch 1/1
    0s - loss: 0.0015 - acc: 1.0000
    epochs : 1721
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1722
    Epoch 1/1
    0s - loss: 0.0014 - acc: 1.0000
    epochs : 1723
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1724
    Epoch 1/1
    0s - loss: 0.0013 - acc: 1.0000
    epochs : 1725
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1726
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1727
    Epoch 1/1
    0s - loss: 0.0012 - acc: 1.0000
    epochs : 1728
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1729
    Epoch 1/1
    0s - loss: 0.0011 - acc: 1.0000
    epochs : 1730
    Epoch 1/1
    0s - loss: 0.0010 - acc: 1.0000
    epochs : 1731
    Epoch 1/1
    0s - loss: 9.8908e-04 - acc: 1.0000
    epochs : 1732
    Epoch 1/1
    0s - loss: 9.5410e-04 - acc: 1.0000
    epochs : 1733
    Epoch 1/1
    0s - loss: 9.1770e-04 - acc: 1.0000
    epochs : 1734
    Epoch 1/1
    0s - loss: 8.8636e-04 - acc: 1.0000
    epochs : 1735
    Epoch 1/1
    0s - loss: 8.5658e-04 - acc: 1.0000
    epochs : 1736
    Epoch 1/1
    0s - loss: 8.2965e-04 - acc: 1.0000
    epochs : 1737
    Epoch 1/1
    0s - loss: 8.0727e-04 - acc: 1.0000
    epochs : 1738
    Epoch 1/1
    0s - loss: 7.8488e-04 - acc: 1.0000
    epochs : 1739
    Epoch 1/1
    0s - loss: 7.6627e-04 - acc: 1.0000
    epochs : 1740
    Epoch 1/1
    0s - loss: 7.4340e-04 - acc: 1.0000
    epochs : 1741
    Epoch 1/1
    0s - loss: 7.3197e-04 - acc: 1.0000
    epochs : 1742
    Epoch 1/1
    0s - loss: 7.5870e-04 - acc: 1.0000
    epochs : 1743
    Epoch 1/1
    0s - loss: 7.2443e-04 - acc: 1.0000
    epochs : 1744
    Epoch 1/1
    0s - loss: 7.2240e-04 - acc: 1.0000
    epochs : 1745
    Epoch 1/1
    0s - loss: 7.2364e-04 - acc: 1.0000
    epochs : 1746
    Epoch 1/1
    0s - loss: 6.9996e-04 - acc: 1.0000
    epochs : 1747
    Epoch 1/1
    0s - loss: 7.0449e-04 - acc: 1.0000
    epochs : 1748
    Epoch 1/1
    0s - loss: 6.8451e-04 - acc: 1.0000
    epochs : 1749
    Epoch 1/1
    0s - loss: 6.7807e-04 - acc: 1.0000
    epochs : 1750
    Epoch 1/1
    0s - loss: 6.5010e-04 - acc: 1.0000
    epochs : 1751
    Epoch 1/1
    0s - loss: 6.1709e-04 - acc: 1.0000
    epochs : 1752
    Epoch 1/1
    0s - loss: 6.0043e-04 - acc: 1.0000
    epochs : 1753
    Epoch 1/1
    0s - loss: 5.7432e-04 - acc: 1.0000
    epochs : 1754
    Epoch 1/1
    0s - loss: 5.6193e-04 - acc: 1.0000
    epochs : 1755
    Epoch 1/1
    0s - loss: 5.2856e-04 - acc: 1.0000
    epochs : 1756
    Epoch 1/1
    0s - loss: 5.0841e-04 - acc: 1.0000
    epochs : 1757
    Epoch 1/1
    0s - loss: 4.8659e-04 - acc: 1.0000
    epochs : 1758
    Epoch 1/1
    0s - loss: 4.8232e-04 - acc: 1.0000
    epochs : 1759
    Epoch 1/1
    0s - loss: 4.5256e-04 - acc: 1.0000
    epochs : 1760
    Epoch 1/1
    0s - loss: 4.5036e-04 - acc: 1.0000
    epochs : 1761
    Epoch 1/1
    0s - loss: 4.1528e-04 - acc: 1.0000
    epochs : 1762
    Epoch 1/1
    0s - loss: 4.1474e-04 - acc: 1.0000
    epochs : 1763
    Epoch 1/1
    0s - loss: 3.8374e-04 - acc: 1.0000
    epochs : 1764
    Epoch 1/1
    0s - loss: 3.7660e-04 - acc: 1.0000
    epochs : 1765
    Epoch 1/1
    0s - loss: 3.5272e-04 - acc: 1.0000
    epochs : 1766
    Epoch 1/1
    0s - loss: 3.4463e-04 - acc: 1.0000
    epochs : 1767
    Epoch 1/1
    0s - loss: 3.2642e-04 - acc: 1.0000
    epochs : 1768
    Epoch 1/1
    0s - loss: 3.2095e-04 - acc: 1.0000
    epochs : 1769
    Epoch 1/1
    0s - loss: 3.0610e-04 - acc: 1.0000
    epochs : 1770
    Epoch 1/1
    0s - loss: 2.9729e-04 - acc: 1.0000
    epochs : 1771
    Epoch 1/1
    0s - loss: 2.8601e-04 - acc: 1.0000
    epochs : 1772
    Epoch 1/1
    0s - loss: 2.7775e-04 - acc: 1.0000
    epochs : 1773
    Epoch 1/1
    0s - loss: 2.6942e-04 - acc: 1.0000
    epochs : 1774
    Epoch 1/1
    0s - loss: 2.6164e-04 - acc: 1.0000
    epochs : 1775
    Epoch 1/1
    0s - loss: 2.5357e-04 - acc: 1.0000
    epochs : 1776
    Epoch 1/1
    0s - loss: 2.4545e-04 - acc: 1.0000
    epochs : 1777
    Epoch 1/1
    0s - loss: 2.3730e-04 - acc: 1.0000
    epochs : 1778
    Epoch 1/1
    0s - loss: 2.3024e-04 - acc: 1.0000
    epochs : 1779
    Epoch 1/1
    0s - loss: 2.2286e-04 - acc: 1.0000
    epochs : 1780
    Epoch 1/1
    0s - loss: 2.1640e-04 - acc: 1.0000
    epochs : 1781
    Epoch 1/1
    0s - loss: 2.1034e-04 - acc: 1.0000
    epochs : 1782
    Epoch 1/1
    0s - loss: 2.0423e-04 - acc: 1.0000
    epochs : 1783
    Epoch 1/1
    0s - loss: 1.9828e-04 - acc: 1.0000
    epochs : 1784
    Epoch 1/1
    0s - loss: 1.9274e-04 - acc: 1.0000
    epochs : 1785
    Epoch 1/1
    0s - loss: 1.8760e-04 - acc: 1.0000
    epochs : 1786
    Epoch 1/1
    0s - loss: 1.8265e-04 - acc: 1.0000
    epochs : 1787
    Epoch 1/1
    0s - loss: 1.7761e-04 - acc: 1.0000
    epochs : 1788
    Epoch 1/1
    0s - loss: 1.7272e-04 - acc: 1.0000
    epochs : 1789
    Epoch 1/1
    0s - loss: 1.6743e-04 - acc: 1.0000
    epochs : 1790
    Epoch 1/1
    0s - loss: 1.6142e-04 - acc: 1.0000
    epochs : 1791
    Epoch 1/1
    0s - loss: 1.5597e-04 - acc: 1.0000
    epochs : 1792
    Epoch 1/1
    0s - loss: 1.5090e-04 - acc: 1.0000
    epochs : 1793
    Epoch 1/1
    0s - loss: 1.4605e-04 - acc: 1.0000
    epochs : 1794
    Epoch 1/1
    0s - loss: 1.4155e-04 - acc: 1.0000
    epochs : 1795
    Epoch 1/1
    0s - loss: 1.3757e-04 - acc: 1.0000
    epochs : 1796
    Epoch 1/1
    0s - loss: 1.3380e-04 - acc: 1.0000
    epochs : 1797
    Epoch 1/1
    0s - loss: 1.3033e-04 - acc: 1.0000
    epochs : 1798
    Epoch 1/1
    0s - loss: 1.2700e-04 - acc: 1.0000
    epochs : 1799
    Epoch 1/1
    0s - loss: 1.2364e-04 - acc: 1.0000
    epochs : 1800
    Epoch 1/1
    0s - loss: 1.2036e-04 - acc: 1.0000
    epochs : 1801
    Epoch 1/1
    0s - loss: 1.1735e-04 - acc: 1.0000
    epochs : 1802
    Epoch 1/1
    0s - loss: 1.1446e-04 - acc: 1.0000
    epochs : 1803
    Epoch 1/1
    0s - loss: 1.1158e-04 - acc: 1.0000
    epochs : 1804
    Epoch 1/1
    0s - loss: 1.0887e-04 - acc: 1.0000
    epochs : 1805
    Epoch 1/1
    0s - loss: 1.0620e-04 - acc: 1.0000
    epochs : 1806
    Epoch 1/1
    0s - loss: 1.0373e-04 - acc: 1.0000
    epochs : 1807
    Epoch 1/1
    0s - loss: 1.0135e-04 - acc: 1.0000
    epochs : 1808
    Epoch 1/1
    0s - loss: 9.9045e-05 - acc: 1.0000
    epochs : 1809
    Epoch 1/1
    0s - loss: 9.6766e-05 - acc: 1.0000
    epochs : 1810
    Epoch 1/1
    0s - loss: 9.4627e-05 - acc: 1.0000
    epochs : 1811
    Epoch 1/1
    0s - loss: 9.2519e-05 - acc: 1.0000
    epochs : 1812
    Epoch 1/1
    0s - loss: 9.0461e-05 - acc: 1.0000
    epochs : 1813
    Epoch 1/1
    0s - loss: 8.8334e-05 - acc: 1.0000
    epochs : 1814
    Epoch 1/1
    0s - loss: 8.6194e-05 - acc: 1.0000
    epochs : 1815
    Epoch 1/1
    0s - loss: 8.4147e-05 - acc: 1.0000
    epochs : 1816
    Epoch 1/1
    0s - loss: 8.2157e-05 - acc: 1.0000
    epochs : 1817
    Epoch 1/1
    0s - loss: 8.0111e-05 - acc: 1.0000
    epochs : 1818
    Epoch 1/1
    0s - loss: 7.8153e-05 - acc: 1.0000
    epochs : 1819
    Epoch 1/1
    0s - loss: 7.6249e-05 - acc: 1.0000
    epochs : 1820
    Epoch 1/1
    0s - loss: 7.4278e-05 - acc: 1.0000
    epochs : 1821
    Epoch 1/1
    0s - loss: 7.2293e-05 - acc: 1.0000
    epochs : 1822
    Epoch 1/1
    0s - loss: 7.0404e-05 - acc: 1.0000
    epochs : 1823
    Epoch 1/1
    0s - loss: 6.8599e-05 - acc: 1.0000
    epochs : 1824
    Epoch 1/1
    0s - loss: 6.6807e-05 - acc: 1.0000
    epochs : 1825
    Epoch 1/1
    0s - loss: 6.5124e-05 - acc: 1.0000
    epochs : 1826
    Epoch 1/1
    0s - loss: 6.3514e-05 - acc: 1.0000
    epochs : 1827
    Epoch 1/1
    0s - loss: 6.1931e-05 - acc: 1.0000
    epochs : 1828
    Epoch 1/1
    0s - loss: 6.0462e-05 - acc: 1.0000
    epochs : 1829
    Epoch 1/1
    0s - loss: 5.8978e-05 - acc: 1.0000
    epochs : 1830
    Epoch 1/1
    0s - loss: 5.7565e-05 - acc: 1.0000
    epochs : 1831
    Epoch 1/1
    0s - loss: 5.6153e-05 - acc: 1.0000
    epochs : 1832
    Epoch 1/1
    0s - loss: 5.4720e-05 - acc: 1.0000
    epochs : 1833
    Epoch 1/1
    0s - loss: 5.3402e-05 - acc: 1.0000
    epochs : 1834
    Epoch 1/1
    0s - loss: 5.2092e-05 - acc: 1.0000
    epochs : 1835
    Epoch 1/1
    0s - loss: 5.0857e-05 - acc: 1.0000
    epochs : 1836
    Epoch 1/1
    0s - loss: 4.9634e-05 - acc: 1.0000
    epochs : 1837
    Epoch 1/1
    0s - loss: 4.8436e-05 - acc: 1.0000
    epochs : 1838
    Epoch 1/1
    0s - loss: 4.7290e-05 - acc: 1.0000
    epochs : 1839
    Epoch 1/1
    0s - loss: 4.6197e-05 - acc: 1.0000
    epochs : 1840
    Epoch 1/1
    0s - loss: 4.5080e-05 - acc: 1.0000
    epochs : 1841
    Epoch 1/1
    0s - loss: 4.3946e-05 - acc: 1.0000
    epochs : 1842
    Epoch 1/1
    0s - loss: 4.2849e-05 - acc: 1.0000
    epochs : 1843
    Epoch 1/1
    0s - loss: 4.1815e-05 - acc: 1.0000
    epochs : 1844
    Epoch 1/1
    0s - loss: 4.0756e-05 - acc: 1.0000
    epochs : 1845
    Epoch 1/1
    0s - loss: 3.9731e-05 - acc: 1.0000
    epochs : 1846
    Epoch 1/1
    0s - loss: 3.8668e-05 - acc: 1.0000
    epochs : 1847
    Epoch 1/1
    0s - loss: 3.7662e-05 - acc: 1.0000
    epochs : 1848
    Epoch 1/1
    0s - loss: 3.6677e-05 - acc: 1.0000
    epochs : 1849
    Epoch 1/1
    0s - loss: 3.5804e-05 - acc: 1.0000
    epochs : 1850
    Epoch 1/1
    0s - loss: 3.4834e-05 - acc: 1.0000
    epochs : 1851
    Epoch 1/1
    0s - loss: 3.3936e-05 - acc: 1.0000
    epochs : 1852
    Epoch 1/1
    0s - loss: 3.3043e-05 - acc: 1.0000
    epochs : 1853
    Epoch 1/1
    0s - loss: 3.2293e-05 - acc: 1.0000
    epochs : 1854
    Epoch 1/1
    0s - loss: 3.1441e-05 - acc: 1.0000
    epochs : 1855
    Epoch 1/1
    0s - loss: 3.0595e-05 - acc: 1.0000
    epochs : 1856
    Epoch 1/1
    0s - loss: 2.9857e-05 - acc: 1.0000
    epochs : 1857
    Epoch 1/1
    0s - loss: 2.9118e-05 - acc: 1.0000
    epochs : 1858
    Epoch 1/1
    0s - loss: 2.8406e-05 - acc: 1.0000
    epochs : 1859
    Epoch 1/1
    0s - loss: 2.7629e-05 - acc: 1.0000
    epochs : 1860
    Epoch 1/1
    0s - loss: 2.6963e-05 - acc: 1.0000
    epochs : 1861
    Epoch 1/1
    0s - loss: 2.6283e-05 - acc: 1.0000
    epochs : 1862
    Epoch 1/1
    0s - loss: 2.5635e-05 - acc: 1.0000
    epochs : 1863
    Epoch 1/1
    0s - loss: 2.4982e-05 - acc: 1.0000
    epochs : 1864
    Epoch 1/1
    0s - loss: 2.4370e-05 - acc: 1.0000
    epochs : 1865
    Epoch 1/1
    0s - loss: 2.3720e-05 - acc: 1.0000
    epochs : 1866
    Epoch 1/1
    0s - loss: 2.3168e-05 - acc: 1.0000
    epochs : 1867
    Epoch 1/1
    0s - loss: 2.2564e-05 - acc: 1.0000
    epochs : 1868
    Epoch 1/1
    0s - loss: 2.2062e-05 - acc: 1.0000
    epochs : 1869
    Epoch 1/1
    0s - loss: 2.1476e-05 - acc: 1.0000
    epochs : 1870
    Epoch 1/1
    0s - loss: 2.0987e-05 - acc: 1.0000
    epochs : 1871
    Epoch 1/1
    0s - loss: 2.0514e-05 - acc: 1.0000
    epochs : 1872
    Epoch 1/1
    0s - loss: 2.0074e-05 - acc: 1.0000
    epochs : 1873
    Epoch 1/1
    0s - loss: 1.9605e-05 - acc: 1.0000
    epochs : 1874
    Epoch 1/1
    0s - loss: 1.9174e-05 - acc: 1.0000
    epochs : 1875
    Epoch 1/1
    0s - loss: 1.8719e-05 - acc: 1.0000
    epochs : 1876
    Epoch 1/1
    0s - loss: 1.8350e-05 - acc: 1.0000
    epochs : 1877
    Epoch 1/1
    0s - loss: 1.7913e-05 - acc: 1.0000
    epochs : 1878
    Epoch 1/1
    0s - loss: 1.7499e-05 - acc: 1.0000
    epochs : 1879
    Epoch 1/1
    0s - loss: 1.7191e-05 - acc: 1.0000
    epochs : 1880
    Epoch 1/1
    0s - loss: 1.6788e-05 - acc: 1.0000
    epochs : 1881
    Epoch 1/1
    0s - loss: 1.6524e-05 - acc: 1.0000
    epochs : 1882
    Epoch 1/1
    0s - loss: 1.6183e-05 - acc: 1.0000
    epochs : 1883
    Epoch 1/1
    0s - loss: 1.5955e-05 - acc: 1.0000
    epochs : 1884
    Epoch 1/1
    0s - loss: 1.5609e-05 - acc: 1.0000
    epochs : 1885
    Epoch 1/1
    0s - loss: 1.5353e-05 - acc: 1.0000
    epochs : 1886
    Epoch 1/1
    0s - loss: 1.5104e-05 - acc: 1.0000
    epochs : 1887
    Epoch 1/1
    0s - loss: 1.4779e-05 - acc: 1.0000
    epochs : 1888
    Epoch 1/1
    0s - loss: 1.4558e-05 - acc: 1.0000
    epochs : 1889
    Epoch 1/1
    0s - loss: 1.4256e-05 - acc: 1.0000
    epochs : 1890
    Epoch 1/1
    0s - loss: 1.3994e-05 - acc: 1.0000
    epochs : 1891
    Epoch 1/1
    0s - loss: 1.3726e-05 - acc: 1.0000
    epochs : 1892
    Epoch 1/1
    0s - loss: 1.3427e-05 - acc: 1.0000
    epochs : 1893
    Epoch 1/1
    0s - loss: 1.3278e-05 - acc: 1.0000
    epochs : 1894
    Epoch 1/1
    0s - loss: 1.2918e-05 - acc: 1.0000
    epochs : 1895
    Epoch 1/1
    0s - loss: 1.2778e-05 - acc: 1.0000
    epochs : 1896
    Epoch 1/1
    0s - loss: 1.2427e-05 - acc: 1.0000
    epochs : 1897
    Epoch 1/1
    0s - loss: 1.2233e-05 - acc: 1.0000
    epochs : 1898
    Epoch 1/1
    0s - loss: 1.1909e-05 - acc: 1.0000
    epochs : 1899
    Epoch 1/1
    0s - loss: 1.1727e-05 - acc: 1.0000
    epochs : 1900
    Epoch 1/1
    0s - loss: 1.1412e-05 - acc: 1.0000
    epochs : 1901
    Epoch 1/1
    0s - loss: 1.1215e-05 - acc: 1.0000
    epochs : 1902
    Epoch 1/1
    0s - loss: 1.0769e-05 - acc: 1.0000
    epochs : 1903
    Epoch 1/1
    0s - loss: 1.0735e-05 - acc: 1.0000
    epochs : 1904
    Epoch 1/1
    0s - loss: 1.0264e-05 - acc: 1.0000
    epochs : 1905
    Epoch 1/1
    0s - loss: 1.0126e-05 - acc: 1.0000
    epochs : 1906
    Epoch 1/1
    0s - loss: 9.8086e-06 - acc: 1.0000
    epochs : 1907
    Epoch 1/1
    0s - loss: 9.6238e-06 - acc: 1.0000
    epochs : 1908
    Epoch 1/1
    0s - loss: 9.3628e-06 - acc: 1.0000
    epochs : 1909
    Epoch 1/1
    0s - loss: 9.1017e-06 - acc: 1.0000
    epochs : 1910
    Epoch 1/1
    0s - loss: 9.1220e-06 - acc: 1.0000
    epochs : 1911
    Epoch 1/1
    0s - loss: 8.6630e-06 - acc: 1.0000
    epochs : 1912
    Epoch 1/1
    0s - loss: 8.5629e-06 - acc: 1.0000
    epochs : 1913
    Epoch 1/1
    0s - loss: 8.2481e-06 - acc: 1.0000
    epochs : 1914
    Epoch 1/1
    0s - loss: 8.1349e-06 - acc: 1.0000
    epochs : 1915
    Epoch 1/1
    0s - loss: 7.8440e-06 - acc: 1.0000
    epochs : 1916
    Epoch 1/1
    0s - loss: 7.7343e-06 - acc: 1.0000
    epochs : 1917
    Epoch 1/1
    0s - loss: 7.4387e-06 - acc: 1.0000
    epochs : 1918
    Epoch 1/1
    0s - loss: 7.2682e-06 - acc: 1.0000
    epochs : 1919
    Epoch 1/1
    0s - loss: 7.0358e-06 - acc: 1.0000
    epochs : 1920
    Epoch 1/1
    0s - loss: 6.9070e-06 - acc: 1.0000
    epochs : 1921
    Epoch 1/1
    0s - loss: 6.6555e-06 - acc: 1.0000
    epochs : 1922
    Epoch 1/1
    0s - loss: 6.5744e-06 - acc: 1.0000
    epochs : 1923
    Epoch 1/1
    0s - loss: 6.2991e-06 - acc: 1.0000
    epochs : 1924
    Epoch 1/1
    0s - loss: 6.2633e-06 - acc: 1.0000
    epochs : 1925
    Epoch 1/1
    0s - loss: 6.0380e-06 - acc: 1.0000
    epochs : 1926
    Epoch 1/1
    0s - loss: 5.8759e-06 - acc: 1.0000
    epochs : 1927
    Epoch 1/1
    0s - loss: 5.7018e-06 - acc: 1.0000
    epochs : 1928
    Epoch 1/1
    0s - loss: 5.6327e-06 - acc: 1.0000
    epochs : 1929
    Epoch 1/1
    0s - loss: 5.4288e-06 - acc: 1.0000
    epochs : 1930
    Epoch 1/1
    0s - loss: 5.3633e-06 - acc: 1.0000
    epochs : 1931
    Epoch 1/1
    0s - loss: 5.1606e-06 - acc: 1.0000
    epochs : 1932
    Epoch 1/1
    0s - loss: 5.0736e-06 - acc: 1.0000
    epochs : 1933
    Epoch 1/1
    0s - loss: 4.9412e-06 - acc: 1.0000
    epochs : 1934
    Epoch 1/1
    0s - loss: 4.8328e-06 - acc: 1.0000
    epochs : 1935
    Epoch 1/1
    0s - loss: 4.7171e-06 - acc: 1.0000
    epochs : 1936
    Epoch 1/1
    0s - loss: 4.6218e-06 - acc: 1.0000
    epochs : 1937
    Epoch 1/1
    0s - loss: 4.5169e-06 - acc: 1.0000
    epochs : 1938
    Epoch 1/1
    0s - loss: 4.4155e-06 - acc: 1.0000
    epochs : 1939
    Epoch 1/1
    0s - loss: 4.3178e-06 - acc: 1.0000
    epochs : 1940
    Epoch 1/1
    0s - loss: 4.2129e-06 - acc: 1.0000
    epochs : 1941
    Epoch 1/1
    0s - loss: 4.0960e-06 - acc: 1.0000
    epochs : 1942
    Epoch 1/1
    0s - loss: 3.9935e-06 - acc: 1.0000
    epochs : 1943
    Epoch 1/1
    0s - loss: 3.8982e-06 - acc: 1.0000
    epochs : 1944
    Epoch 1/1
    0s - loss: 3.8159e-06 - acc: 1.0000
    epochs : 1945
    Epoch 1/1
    0s - loss: 3.7170e-06 - acc: 1.0000
    epochs : 1946
    Epoch 1/1
    0s - loss: 3.6514e-06 - acc: 1.0000
    epochs : 1947
    Epoch 1/1
    0s - loss: 3.5441e-06 - acc: 1.0000
    epochs : 1948
    Epoch 1/1
    0s - loss: 3.4607e-06 - acc: 1.0000
    epochs : 1949
    Epoch 1/1
    0s - loss: 3.3724e-06 - acc: 1.0000
    epochs : 1950
    Epoch 1/1
    0s - loss: 3.3045e-06 - acc: 1.0000
    epochs : 1951
    Epoch 1/1
    0s - loss: 3.2055e-06 - acc: 1.0000
    epochs : 1952
    Epoch 1/1
    0s - loss: 3.1579e-06 - acc: 1.0000
    epochs : 1953
    Epoch 1/1
    0s - loss: 3.0696e-06 - acc: 1.0000
    epochs : 1954
    Epoch 1/1
    0s - loss: 3.0387e-06 - acc: 1.0000
    epochs : 1955
    Epoch 1/1
    0s - loss: 3.0053e-06 - acc: 1.0000
    epochs : 1956
    Epoch 1/1
    0s - loss: 2.9063e-06 - acc: 1.0000
    epochs : 1957
    Epoch 1/1
    0s - loss: 2.7967e-06 - acc: 1.0000
    epochs : 1958
    Epoch 1/1
    0s - loss: 2.8551e-06 - acc: 1.0000
    epochs : 1959
    Epoch 1/1
    0s - loss: 2.7549e-06 - acc: 1.0000
    epochs : 1960
    Epoch 1/1
    0s - loss: 3.7873e-06 - acc: 1.0000
    epochs : 1961
    Epoch 1/1
    0s - loss: 5.4599e-06 - acc: 1.0000
    epochs : 1962
    Epoch 1/1
    0s - loss: 1.6998 - acc: 0.7000
    epochs : 1963
    Epoch 1/1
    0s - loss: 2.6442 - acc: 0.4600
    epochs : 1964
    Epoch 1/1
    0s - loss: 1.5166 - acc: 0.5400
    epochs : 1965
    Epoch 1/1
    0s - loss: 1.3807 - acc: 0.5400
    epochs : 1966
    Epoch 1/1
    0s - loss: 0.8919 - acc: 0.6600
    epochs : 1967
    Epoch 1/1
    0s - loss: 0.5343 - acc: 0.8000
    epochs : 1968
    Epoch 1/1
    0s - loss: 0.4643 - acc: 0.7800
    epochs : 1969
    Epoch 1/1
    0s - loss: 0.3719 - acc: 0.8400
    epochs : 1970
    Epoch 1/1
    0s - loss: 0.2659 - acc: 0.9200
    epochs : 1971
    Epoch 1/1
    0s - loss: 0.2212 - acc: 0.9600
    epochs : 1972
    Epoch 1/1
    0s - loss: 0.4189 - acc: 0.8000
    epochs : 1973
    Epoch 1/1
    0s - loss: 0.5699 - acc: 0.7400
    epochs : 1974
    Epoch 1/1
    0s - loss: 0.2828 - acc: 0.8600
    epochs : 1975
    Epoch 1/1
    0s - loss: 0.1556 - acc: 0.9600
    epochs : 1976
    Epoch 1/1
    0s - loss: 0.0782 - acc: 1.0000
    epochs : 1977
    Epoch 1/1
    0s - loss: 0.0471 - acc: 1.0000
    epochs : 1978
    Epoch 1/1
    0s - loss: 0.0405 - acc: 1.0000
    epochs : 1979
    Epoch 1/1
    0s - loss: 0.1956 - acc: 0.9000
    epochs : 1980
    Epoch 1/1
    0s - loss: 2.1174 - acc: 0.5400
    epochs : 1981
    Epoch 1/1
    0s - loss: 1.1892 - acc: 0.6000
    epochs : 1982
    Epoch 1/1
    0s - loss: 0.4202 - acc: 0.8400
    epochs : 1983
    Epoch 1/1
    0s - loss: 0.4801 - acc: 0.7800
    epochs : 1984
    Epoch 1/1
    0s - loss: 0.2830 - acc: 0.9200
    epochs : 1985
    Epoch 1/1
    0s - loss: 0.2330 - acc: 0.8800
    epochs : 1986
    Epoch 1/1
    0s - loss: 0.1476 - acc: 0.9600
    epochs : 1987
    Epoch 1/1
    0s - loss: 0.0623 - acc: 0.9800
    epochs : 1988
    Epoch 1/1
    0s - loss: 0.0308 - acc: 1.0000
    epochs : 1989
    Epoch 1/1
    0s - loss: 0.0259 - acc: 1.0000
    epochs : 1990
    Epoch 1/1
    0s - loss: 0.0217 - acc: 1.0000
    epochs : 1991
    Epoch 1/1
    0s - loss: 0.0172 - acc: 1.0000
    epochs : 1992
    Epoch 1/1
    0s - loss: 0.0130 - acc: 1.0000
    epochs : 1993
    Epoch 1/1
    0s - loss: 0.0166 - acc: 1.0000
    epochs : 1994
    Epoch 1/1
    0s - loss: 0.0131 - acc: 1.0000
    epochs : 1995
    Epoch 1/1
    0s - loss: 0.0099 - acc: 1.0000
    epochs : 1996
    Epoch 1/1
    0s - loss: 0.0086 - acc: 1.0000
    epochs : 1997
    Epoch 1/1
    0s - loss: 0.0077 - acc: 1.0000
    epochs : 1998
    Epoch 1/1
    0s - loss: 0.0071 - acc: 1.0000
    epochs : 1999
    Epoch 1/1
    0s - loss: 0.0065 - acc: 1.0000
    38/50 [=====================>........] - ETA: 0sacc: 100.00%
    ('one step prediction : ', ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4', 'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4', 'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4', 'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4'])
    ('full song prediction : ', ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4', 'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4', 'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4', 'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4'])


한 스텝 예측 결과과 곡 전체 예측 결과를 악보로 그려보았습니다. Stateful LSTM은 음표를 모두 맞추어서, 전체 곡 예측도 정확하게 했습니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_Stateful_LSTM_song.png)

---

### 결론

익숙한 노래인 "나비야"를 가지고 순한 신경망 모델에 학습시켜봤습니다. 순항 신경망 모델 중 가장 많이 사용되는 LSTM 모델에 대해서 알아보고, 주요 인자들이 어떤 특성을 가지고 있는 지도 살펴보았습니다. 다음 강좌에서는 아래 항목들에 대해서 좀 더 살펴보겠습니다. 
* 입력 속성이 여러 개인 모델 구성
* 시퀀스 출력을 가지는 모델 구성
* LSTM 레이어를 여러개로 쌓아보기
* 상태 유지 모드 여부에 따른 배치사이즈(batch_size)에 대한 이해

---

### 같이 보기

* [강좌 목차](https://tykimos.github.io/Keras/lecture/)
