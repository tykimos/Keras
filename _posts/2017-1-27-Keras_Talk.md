---
layout: post
title:  "케라스 이야기"
author: Taeyoung, Kim
date:   2017-01-27 22:07:00
categories: Keras
comments: true
---
케라스(Keras)는 오래전부터 사용된 티아노(Theano)와 요즘 많이 쓰고 있는 텐서플로우(TensorFlow)를 위한 딥러닝 라이브러리입니다. 케라스는 아이디어를 빨리 구현하고 실험하기 위한 목적에 포커스가 맞춰진 만큼 굉장히 간격하고 쉽게 사용할 수 있도록 파이썬으로 구현된 상위 레벨의 라이브러리입니다. 즉 내부적으론 티아노와 텐서플로우가 구동되지만 연구자는 복잡한 티아노와 텐서플로우을 알 필요는 없습니다. 케라스는 쉽게 컨볼루션 신경망, 순환 신경망 또는 이를 조합한 신경망은 물론 다중 입력 또는 다중 출력 등 다양한 연결 구성을 할 수 있습니다. 

---
케라스 주요 특징
-------------

케라스는 아래 4가지의 주요 특징을 가지고 있습니다.

* 모듈화 (Modularity)
    * 케라스에서 제공하는 모듈은 독립적이고 설정 가능하며, 가능한 최소한의 제약사항으로 서로 연결될 수 있습니다. 모델은 시퀀스 또는 그래프로 이러한 모듈들을 구성한 것입니다.
    * 특히 신경망 층, 비용함수, 최적화기, 초기화기법, 활성화함수, 정규화기법은 모두 독립적인 모듈이며, 새로운 모델을 만들기 위해 이러한 모듈을 조합할 수 있습니다.
* 최소주의 (Minimalism)
    * 각 모듈은 짥고 간결합니다.
    * 모든 코드는 한 번 훏어보는 것으로도 이해가능해야 합니다.
    * 단 반복 속도와 혁신성에는 다소 떨어질 수가 있습니다. 
* 쉬운 확장성
    * 새로운 클래스나 함수로 모듈을 아주 쉽게 추가할 수 있습니다. 
    * 따라서 고급 연구에 필요한 다양한 표현을 할 수 있습니다. 
* 파이썬 기반
    * Caffe 처럼 별도의 모델 설정 파일이 필요없으며 파이썬 코드로 모델들이 정의됩니다.
    
케라스를 개발하고 유지보수하고 있는 사람은 구글 엔진니어인 프랑소와 쏠레(François Chollet)입니다.

---
케라스 기본 개념
-------------

케라스의 가장 핵심적인 데이터 구조는 바로 **모델**입니다. 케라스에서 제공하는 시퀀스 모델로 원하는 레이어를 쉽게 순차적으로 쌓을 수 있습니다. 다중 출력이 필요하는 등 좀 더 복잡한 모델을 구성하려면 케라스 함수 API를 사용하면 됩니다.


```python
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# prepare dataset
# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# flatten and normalize
X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# create model
model = Sequential()
model.add(Dense(output_dim=64, input_dim=28*28))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# fit model
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
 
# evaluate
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
print(loss_and_metrics)
```

    Epoch 1/5
    60000/60000 [==============================] - 4s - loss: 0.6870 - acc: 0.8219     
    Epoch 2/5
    60000/60000 [==============================] - 3s - loss: 0.3532 - acc: 0.9000     
    Epoch 3/5
    60000/60000 [==============================] - 3s - loss: 0.3068 - acc: 0.9126     
    Epoch 4/5
    60000/60000 [==============================] - 3s - loss: 0.2781 - acc: 0.9211     
    Epoch 5/5
    60000/60000 [==============================] - 3s - loss: 0.2554 - acc: 0.9285     
    10000/10000 [==============================] - 0s     
    [0.23559541808068751, 0.93400000000000005]


---
### 케라스 기본 개념

케라스의 가장 핵심적인 데이터 구조는 바로 __모델__입니다. 케라스에서 제공하는 시퀀스 모델로 원하는 레이어를 쉽게 순차적으로 쌓을 수 있습니다. 다중 출력이 필요하는 등 좀 더 복잡한 모델을 구성하려면 케라스 함수 API를 사용하면 됩니다.


Why this name, Keras?

Keras (κέρας) means horn in Greek. It is a reference to a literary image from ancient Greek and Latin literature, first found in the Odyssey, where dream spirits (Oneiroi, singular Oneiros) are divided between those who deceive men with false visions, who arrive to Earth through a gate of ivory, and those who announce a future that will come to pass, who arrive through a gate of horn. It's a play on the words κέρας (horn) / κραίνω (fulfill), and ἐλέφας (ivory) / ἐλεφαίρομαι (deceive).

Keras was initially developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System).

"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them." Homer, Odyssey 19. 562 ff (Shewring translation).

왜이 이름 이냐, 케사스?

Keras (κέρας)는 그리스어로 뿔을 의미합니다. 오디세이에서 처음 발견 된 고대 그리스 및 라틴 문학의 문학 이미지에 대한 참고서로, 꿈의 영혼 (Oneiroi, 단수 Oneiros)은 그릇된 시각을 가진 사람들을 속이고 누가 상아의 문을 통해 지구에 도착하는지 , 그리고 앞으로 나올 미래를 선포하고 경적 문을 통해 도착하는 사람들. 그것은 κέρας (경적) / κραίνω (이행), ἐλέφας (상아색) / ἐλεφαίρομαι (속임수)에 관한 연극입니다.

Keras는 초기에 ONEIROS 프로젝트 (Open-ended Neuro-Electronic 지능형 로봇 운영 시스템)의 연구 노력의 일부로 개발되었습니다.

"Oneiroi는 우리가 풀어 놓을 수없는 것 - 사람들이 말하는 이야기가 모두 전달되는 것을 확신 할 수있는 사람이 누구인지는 알지 못한다. 두 개의 게이트는 잠깐 동안의 Oneiroi에게 패스를주고, 하나는 뿔과 상아로 만든다.


```python

```
