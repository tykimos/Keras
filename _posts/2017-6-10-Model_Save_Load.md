---
layout: post
title:  "학습 모델 저장하기/불러오기"
author: 김태영
date:   2017-06-10 23:10:00
categories: Lecture
comments: true
image: http://tykimos.github.com/Keras/warehouse/2017-6-10-Model_Load_Save_1.png
---
몇시간 동안 (또는 며칠 동안) 딥러닝 모델을 학습 시킨 후 만족할만한 결과를 얻었다면, 실무에 바로 적용시키고 싶으실 겁니다. 이 때 떠오르는 의문 중 하나가 "딥러닝 모델을 사용하려면 매번 이렇게 몇시간 동안 학습시켜야 되는 거야?"입니다. 대답은 "아니오" 입니다. 딥러닝 모델을 학습시킨다는 의미는 딥러닝 모델이 가지고 있는 뉴런들의 가중치(weight)을 조정한다는 의미이고, 우리는 이러한 가중치만 저장만 해놓으면, 필요할 때 저장한 가중치를 이용하여 사용하면 됩니다. 간단한 딥러닝 모델로 가중치를 저장 및 불러오는 방법에 대해서 알아보겠습니다.

1. 간단한 모델 보기
1. 실무에서의 딥러닝 시스템
1. 학습된 모델 저장하기
1. 학습된 모델 불러오기

---

### 간단한 모델 보기

아래 코드는 MNIST 데이터셋(손글씨)을 이용하여 숫자를 분류하는 문제를 간단한 다층퍼셉트론 모델을 구성한 후 학습 시킨 후 판정하는 코드입니다.


```python
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# 1. 데이터셋 준비하기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit(X_train, Y_train, epochs=5, batch_size=32)

# 5. 모델 사용하기
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

print('')
print('loss_and_metrics : ' + str(loss_and_metrics))
```

    Epoch 1/5
    60000/60000 [==============================] - 1s - loss: 0.6658 - acc: 0.8305     
    Epoch 2/5
    60000/60000 [==============================] - 1s - loss: 0.3494 - acc: 0.9020     
    Epoch 3/5
    60000/60000 [==============================] - 1s - loss: 0.2996 - acc: 0.9154     
    Epoch 4/5
    60000/60000 [==============================] - 1s - loss: 0.2690 - acc: 0.9246     
    Epoch 5/5
    60000/60000 [==============================] - 1s - loss: 0.2458 - acc: 0.9304     
     9920/10000 [============================>.] - ETA: 0s
    loss_and_metrics : [0.22901511510163547, 0.93500000000000005]


이 코드에서 '4. 모델 학습시키기'까지가 학습을 하기 위한 과정이고, '5. 모델 사용하기'이후 코드가 학습된 모델을 사용하는 부분입니다. 이 사이를 분리하여 별도의 모듈로 만들면 우리가 원하는 결과를 얻을 수 있습니다.

---

### 실무에서의 딥러닝 시스템

모듈을 분리하기 전에 실무에서의 딥러닝 시스템을 살펴보겠습니다. 도메인, 문제에 마다 다양한 구성이 있겠지만, 제가 생각하는 딥러닝 시스템 구성은 다음과 같습니다.

![data](http://tykimos.github.com/Keras/warehouse/2017-6-10-Model_Load_Save_1.png)

우리가 만들고자 하는 전체 시스템을 목표 시스템이라고 했을 때, 크게 '학습 segment'와 '판정 segment'로 나누어집니다. '학습 segment'는 학습을 위해, 학습 데이터를 얻기 위한 '학습용 센싱 element', 센싱 데이터에서 학습에 적합한 형태로 전처리를 수행하는 '데이터셋 생성 element', 그리고 데이터셋으로 딥러닝 모델을 학습시키는 '딥러닝 모델 학습 element'으로 나누어집니다. '판정 segment'는 실무 환경에서 수집되는 센서인 '판정용 센싱 element'과 학습된 딥러닝 모델을 이용해서 센싱 데이터를 판정하는 '딥러닝 모델 판정 element'으로 나누어집니다. 앞서 본 코드에는 `딥러닝 모델 학습 element`와 `딥러닝 모델 판정 element`가 모두 포함되어 있습니다. 이 두가지 element를 분리해보겠습니다. 

    딥러닝 시스템은 크게 학습 부분과 판정 부분으로 나누어진다.

---

### 학습된 모델 저장하기

아래 코드는 훈련데이터셋으로 모델을 학습시킨 후, 학습된 모델을 파일로 저장하는 코드입니다. 바뀐 부분은 다음과 같습니다.
- X_test, Y_test가 필요 없습니다.
- 모델을 저장하기 위한 코드를 추가합니다. (아래 코드에서 5번 주석 확인)


```python
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# 1. 데이터셋 준비하기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
# X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
Y_train = np_utils.to_categorical(Y_train)
# Y_test = np_utils.to_categorical(Y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit(X_train, Y_train, epochs=5, batch_size=32)

# 5. 모델 저장하기
from keras.models import load_model
model.save('mnist_mlp_model.h5')
```

    Epoch 1/5
    60000/60000 [==============================] - 1s - loss: 0.6558 - acc: 0.8323     
    Epoch 2/5
    60000/60000 [==============================] - 1s - loss: 0.3383 - acc: 0.9070     
    Epoch 3/5
    60000/60000 [==============================] - 1s - loss: 0.2911 - acc: 0.9193     
    Epoch 4/5
    60000/60000 [==============================] - 1s - loss: 0.2632 - acc: 0.9256     
    Epoch 5/5
    60000/60000 [==============================] - 1s - loss: 0.2427 - acc: 0.9316     


'mnist_mlp_model.h5'라는 파일이 작업 디렉토리에 생성되었는 지 확인해봅니다. 예제에서는 424KB로 생성되었습니다. 저장된 파일에는 다음의 정보가 담겨 있습니다.

- 나중에 모델을 재구성하기 위한 모델의 구성 정보
- 모델를 구성하는 각 뉴런들의 가중치
- 손실함수, 최적하기 등의 학습 설정
- 재학습을 할 수 있도록 마지막 학습 상태

---

### 학습된 모델 불러오기

'mnist_mlp_model.h5'에 학습된 결과가 저장되어 있으니, 이를 불러와서 사용해봅니다. 코드 흐름은 다음과 같습니다.
- X_test, Y_test 데이터셋 준비합니다. 실무에서는 실제로 들어오는 데이터를 사용하시면 됩니다.
- 모델 불러오는 함수를 이용하여 앞서 저장한 모델 파일로부터 모델을 재형성합니다.
- 실제 데이터로 모델을 사용합니다. 예제에서는 정상적으로 모델을 불러왔는 지 확인하기 위해 evaluate() 함수를 사용했지만, 실무에서 입력 데이터에 대한 모델 출력 결과를 얻어야 하므로 predict() 함수를 사용합니다.


```python
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# 1. 데이터셋 준비하기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
# Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# 2. 모델 불러오기
from keras.models import load_model

model = load_model('mnist_mlp_model.h5')

# 3. 모델 사용하기
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

print('')
print('loss_and_metrics : ' + str(loss_and_metrics))
```

     9728/10000 [============================>.] - ETA: 0s
    loss_and_metrics : [0.2422757856875658, 0.93230000000000002]


정상적으로 학습된 모델을 불러와서, 분리하기 전 코드에서의 모델과 유사한 결과를 얻었음을 알 수 있습니다.

---

### 결론

본 강좌에서는 학습한 모델을 저장하고 불러오는 방법에 대해서 알아보았습니다. 저장된 파일에는 모델 구성 및 가중치 정보외에도 학습 설정 및 상태가 저장되므로 모델을 불러온 후 재 학습을 시킬 수 있습니다. 일반적인 딥러닝 시스템에서는 학습 처리 시간을 단축시키기 위해 GPU나 클러스터 장비에서 학습 과정이 이루어지나, 판정 과정은 학습된 모델 결과 파일을 이용하여 일반 PC 및 모바일, 임베디드 등에서 이루어집니다. 도메인, 사용 목적 등에 따라 이러한 환경이 다양하기 때문에, 딥러닝 모델에 대한 연구도 중요하지만, 실무에 적용하기 위해서는 목표 시스템에 대한 설계도 중요합니다.

---

### 같이 보기

* [강좌 목차](https://tykimos.github.io/Keras/lecture/)
