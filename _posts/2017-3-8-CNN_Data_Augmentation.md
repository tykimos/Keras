---
layout: post
title:  "컨볼루션 신경망 모델 만들어보기"
author: Taeyoung, Kim
date:   2017-03-08 23:00:00
categories: Lecture
comments: true
image: http://tykimos.github.com/Keras/warehouse/2017-3-8_CNN_Getting_Started_4.png
---
본 강좌에서는 간단한 컨볼루션 신경망 모델을 만들어봅니다. 늘 그렇듯이 다음과 같은 순서로 진행하겠습니다.

1. 문제 정의하기
1. 데이터셋 준비하기
1. 모델 구성하기
1. 모델 엮기
1. 모델 학습시키기
1. 모델 사용하기

---

### 문제 정의하기

좋은 예제와 그와 관련된 데이터셋도 공개된 것이 많이 있지만, 직접 문제를 정의하고 데이터를 만들어보는 것도 처럼 딥러닝을 접하시는 분들에게는 크게 도움이 될 것 같습니다. 컨볼루션 신경망 모델에 적합한 문제는 이미지 기반의 분류입니다. 따라서 우리는 직접 손으로 삼각형, 사각형, 원을 그려 이미지로 저장한 다음 이를 분류해보는 모델을 만들어보겠습니다. 문제 형태와 입출력을 다음과 같이 정의해봅니다.
* 문제 형태 : 다중 클래스 분류
* 입력 : 손으로 그린 삼각형, 사각형, 원 이미지
* 출력 : 삼각형, 사각형, 원일 확률을 나타내는 벡터

매번 실행 시마다 결과가 달라지지 않도록 랜덤 시드를 명시적으로 지정합니다.


```python
import numpy as np

# 랜덤시드 고정시키기
np.random.seed(5)
```

---

### 데이터셋 준비하기

손으로 그린 삼각형, 사각형, 원 이미지를 만들기 위해서는 여러가지 방법이 있을 수 있겠네요. 테블릿을 이용할 수도 있고, 종이에 그려서 사진으로 찍을 수도 있습니다. 저는 그림 그리는 툴을 이용해서 만들어봤습니다. 이미지 사이즈는 24 x 24 정도로 해봤습니다. 

![data](http://tykimos.github.com/Keras/warehouse/2017-3-8_CNN_Getting_Started_1.png)

모양별로 20개 정도를 만들어서 15개를 훈련에 사용하고, 5개를 테스트에 사용해보겠습니다. 이미지는 png나 jpg로 저장합니다. 실제로 데이터셋이 어떻게 구성되어 있는 지 모른 체 튜토리얼을 따라하거나 예제 코드를 실행시키다보면 결과는 잘 나오지만 막상 실제 문제에 적용할 때 막막해질 때가 있습니다. 간단한 예제로 직접 데이터셋을 만들어봄으로써 실제 문제에 접근할 때 시행착오를 줄이는 것이 중요합니다.

데이터셋 폴더는 다음과 같이 구성했습니다.

- train
 - circle
 - rectangle
 - triangle
- validation
 - circle
 - rectangle
 - triangle
 
![data](http://tykimos.github.com/Keras/warehouse/2017-3-8_CNN_Getting_Started_2.png)


---

### 데이터셋 불러오기

케라스에서는 이미지 파일을 쉽게 학습시킬 수 있도록 ImageDataGenerator 클래스를 제공합니다. ImageDataGenerator 클래스는 데이터 증강 (data augmentation)을 위해 막강한 기능을 제공하는 데, 이 기능들은 다른 강좌에세 다루기로 하고, 본 강좌에서는 특정 폴더에 이미지를 분류 해놓았을 때 이를 학습시키기 위한 데이터셋으로 만들어주는 기능을 사용해보겠습니다.

먼저 ImageDataGenerator 클래스를 이용하여 객체를 생성한 뒤 flow_from_directory() 함수를 호출하여 제네레이터(generator)를 생성합니다. flow_from_directory() 함수의 주요인자는 다음과 같습니다.

- 첫번재 인자 : 이미지 경로를 지정합니다.
- target_size : 패치 이미지 크기를 지정합니다. 폴더에 있는 원본 이미지 크기가 다르더라도 target_size에 지정된 크기로 자동 조절됩니다.
- batch_size : 배치 크기를 지정합니다.
- class_mode : 분류 방식에 대해서 지정합니다.
    - categorical : 2D one-hot 부호화된 라벨이 반환됩니다.
    - binary : 1D 이진 라벨이 반환됩니다.
    - sparse : 1D 정수 라벨이 반환됩니다.
    - None : 라벨이 반환되지 않습니다.

본 예제에서는 패치 이미지 크기를 24 x 24로 하였으니 target_size도 (24, 24)로 셋팅하였습니다. 훈련 데이터 수가 클래스당 15개이니 배치 크기를 3으로 지정하여 총 5번 배치를 수행하면 하나의 epoch가 수행될 수 있도록 하였습니다. 다중 클래스 문제이므로 class_mode는 'categorical'로 지정하였습니다. 그리고 제네레이터는 훈련용과 검증용으로 두 개를 만들었습니다. 


```python
from keras.preprocessing.image import ImageDataGenerator

# 데이터셋 불러오기
train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        'warehouse/handwriting_shape/train',
        target_size=(24, 24),
        batch_size=3,
        class_mode='categorical')

validation_datagen = ImageDataGenerator()

validation_generator = validation_datagen.flow_from_directory(
        'warehouse/handwriting_shape/validation',
        target_size=(24, 24),    
        batch_size=3,
        class_mode='categorical')
```

    Found 45 images belonging to 3 classes.
    Found 15 images belonging to 3 classes.


    Using Theano backend.


---

### 모델 구성하기

영상 분류에 높은 성능을 보이고 있는 컨볼루션 신경망 모델을 구성해보겠습니다. 이전 강좌에서 만들어봤던 모델 위에 입력을 24 x 24, 3개 채널 이미지를 받을 수 있고 필터를 12개를 가진 컨볼루션 레이어와 맥스풀링 레이어를 추가해봤습니다.

* 컨볼루션 레이어 : 입력 이미지 크기 24 x 24, 입력 이미지 채널 3개, 필터 크기 3 x 3, 필터 수 12개, 경계 타입 'same', 활성화 함수 'relu'
* 맥스풀링 레이어 : 풀 크기 2 x 2
* 컨볼루션 레이어 : 입력 이미지 크기 8 x 8, 입력 이미지 채널 1개, 필터 크기 3 x 3, 필터 수 2개, 경계 타입 'same', 활성화 함수 'relu'
* 맥스풀링 레이어 : 풀 크기 2 x 2
* 컨볼루션 레이어 : 입력 이미지 크기 4 x 4, 입력 이미지 채널 2개, 필터 크기 2 x 2, 필터 수 3개, 경계 타입 'same', 활성화 함수 'relu'
* 맥스풀링 레이어 : 풀 크기 2 x 2
* 플래튼 레이어
* 댄스 레이어 : 입력 뉴런 수 12개, 출력 뉴런 수 8개, 활성화 함수 'relu'
* 댄스 레이어 : 입력 뉴런 수 8개, 출력 뉴런 수 3개, 활성화 함수 'softmax'


```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

# 모델 구성하기
model = Sequential()
model.add(Convolution2D(12, 3, 3, border_mode='same', input_shape=(3, 24, 24), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(2, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(3, 2, 2, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
```

    Using Theano backend.



```python
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
```




![svg](output_10_0.svg)



![model](http://tykimos.github.com/Keras/warehouse/2017-3-8_CNN_Getting_Started_3.png)

---

### 모델 엮기

모델을 정의했다면 모델을 손실함수와 최적화 알고리즘으로 엮어봅니다. 
- loss : 현재 가중치 세트를 평가하는 데 사용한 손실 함수 입니다. 다중 클래스 문제이므로 'categorical_crossentropy'으로 지정합니다.
- optimizer : 최적의 가중치를 검색하는 데 사용되는 최적화 알고리즘으로 효율적인 경사 하강법 알고리즘 중 하나인 'adam'을 사용합니다.
- metrics : 평가 척도를 나타내며 분류 문제에서는 일반적으로 'accuracy'으로 지정합니다.


```python
# 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### 모델 학습시키기

케라스에서는 모델을 학습시킬 때 주로 fit 함수를 사용하지만 제네레이터로 생성된 배치로 학습시킬 경우에는 fit_generator 함수를 사용합니다. 본 예제에서는 ImageDataGenerator라는 제네레이터로 이미지를 담고 있는 배치로 학습시키기 때문에 fit_generator 함수를 사용하겠습니다.

- 첫번째 인자 : 훈련데이터셋을 제공할 제네레이터를 지정합니다. 본 예제에서는 앞서 생성한 train_generator으로 지정합니다.
- samples_per_epoch : 한 epoch에 사용한 샘플 수를 지정합니다. 총 45개의 훈련 샘플이 있으므로 45로 지정합니다.
- nb_epoch : 전체 훈련 데이터셋에 대해 학습 반복 횟수를 지정합니다. 100번을 반복적으로 학습시켜 보겠습니다.
- validation_data : 검증데이터셋을 제공할 제네레이터를 지정합니다. 본 예제에서는 앞서 생성한 validation_generator으로 지정합니다.
- nb_val_samples : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 샘플 수를 지정합니다. 홍 15개의 검증 샘플이 있으므로 15로 지정합니다.


```python
# 모델 학습시키기
model.fit_generator(
        train_generator,
        samples_per_epoch=45,
        nb_epoch=100,
        validation_data=validation_generator,
        nb_val_samples=15)
```

    Epoch 1/100
    45/45 [==============================] - 0s - loss: 6.3339 - acc: 0.4667 - val_loss: 6.0346 - val_acc: 0.6000
    Epoch 2/100
    45/45 [==============================] - 0s - loss: 5.6510 - acc: 0.6222 - val_loss: 5.6219 - val_acc: 0.6000
    Epoch 3/100
    45/45 [==============================] - 0s - loss: 5.5095 - acc: 0.6222 - val_loss: 5.9724 - val_acc: 0.6000
    Epoch 4/100
    45/45 [==============================] - 0s - loss: 5.5965 - acc: 0.6222 - val_loss: 6.3116 - val_acc: 0.6000
    Epoch 5/100
    45/45 [==============================] - 0s - loss: 5.4784 - acc: 0.6222 - val_loss: 6.4343 - val_acc: 0.6000
    Epoch 6/100
    45/45 [==============================] - 0s - loss: 5.3974 - acc: 0.6444 - val_loss: 6.3503 - val_acc: 0.6000
    Epoch 7/100
    45/45 [==============================] - 0s - loss: 5.3832 - acc: 0.6667 - val_loss: 6.4474 - val_acc: 0.6000
    Epoch 8/100
    45/45 [==============================] - 0s - loss: 5.3757 - acc: 0.6667 - val_loss: 6.4473 - val_acc: 0.6000
    Epoch 9/100
    45/45 [==============================] - 0s - loss: 5.3739 - acc: 0.6667 - val_loss: 6.4344 - val_acc: 0.6000
    Epoch 10/100
    45/45 [==============================] - 0s - loss: 5.3735 - acc: 0.6667 - val_loss: 6.4193 - val_acc: 0.6000
    Epoch 11/100
    45/45 [==============================] - 0s - loss: 5.3734 - acc: 0.6667 - val_loss: 6.4174 - val_acc: 0.6000
    Epoch 12/100
    45/45 [==============================] - 0s - loss: 5.3734 - acc: 0.6667 - val_loss: 6.4177 - val_acc: 0.6000
    Epoch 13/100
    45/45 [==============================] - 0s - loss: 5.3733 - acc: 0.6667 - val_loss: 6.4179 - val_acc: 0.6000
    Epoch 14/100
    45/45 [==============================] - 0s - loss: 5.3732 - acc: 0.6667 - val_loss: 6.4179 - val_acc: 0.6000
    Epoch 15/100
    45/45 [==============================] - 0s - loss: 5.3732 - acc: 0.6667 - val_loss: 6.4175 - val_acc: 0.6000
    Epoch 16/100
    45/45 [==============================] - 0s - loss: 5.3732 - acc: 0.6667 - val_loss: 6.4169 - val_acc: 0.6000
    Epoch 17/100
    45/45 [==============================] - 0s - loss: 5.3731 - acc: 0.6667 - val_loss: 6.4162 - val_acc: 0.6000
    Epoch 18/100
    45/45 [==============================] - 0s - loss: 5.3731 - acc: 0.6667 - val_loss: 6.4154 - val_acc: 0.6000
    Epoch 19/100
    45/45 [==============================] - 0s - loss: 5.3731 - acc: 0.6667 - val_loss: 6.4147 - val_acc: 0.6000
    Epoch 20/100
    45/45 [==============================] - 0s - loss: 5.3731 - acc: 0.6667 - val_loss: 6.4138 - val_acc: 0.6000
    Epoch 21/100
    45/45 [==============================] - 0s - loss: 5.3730 - acc: 0.6667 - val_loss: 6.4130 - val_acc: 0.6000
    Epoch 22/100
    45/45 [==============================] - 0s - loss: 5.3730 - acc: 0.6667 - val_loss: 6.4123 - val_acc: 0.6000
    Epoch 23/100
    45/45 [==============================] - 0s - loss: 5.3730 - acc: 0.6667 - val_loss: 6.4116 - val_acc: 0.6000
    Epoch 24/100
    45/45 [==============================] - 0s - loss: 5.3730 - acc: 0.6667 - val_loss: 6.4108 - val_acc: 0.6000
    Epoch 25/100
    45/45 [==============================] - 0s - loss: 5.3730 - acc: 0.6667 - val_loss: 6.4100 - val_acc: 0.6000
    Epoch 26/100
    45/45 [==============================] - 0s - loss: 5.3730 - acc: 0.6667 - val_loss: 6.4092 - val_acc: 0.6000
    Epoch 27/100
    45/45 [==============================] - 0s - loss: 5.3730 - acc: 0.6667 - val_loss: 6.4084 - val_acc: 0.6000
    Epoch 28/100
    45/45 [==============================] - 0s - loss: 5.3730 - acc: 0.6667 - val_loss: 6.4077 - val_acc: 0.6000
    Epoch 29/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.4070 - val_acc: 0.6000
    Epoch 30/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.4062 - val_acc: 0.6000
    Epoch 31/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.4045 - val_acc: 0.6000
    Epoch 32/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.4022 - val_acc: 0.6000
    Epoch 33/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.4000 - val_acc: 0.6000
    Epoch 34/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3978 - val_acc: 0.6000
    Epoch 35/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3956 - val_acc: 0.6000
    Epoch 36/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3935 - val_acc: 0.6000
    Epoch 37/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3914 - val_acc: 0.6000
    Epoch 38/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3894 - val_acc: 0.6000
    Epoch 39/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3874 - val_acc: 0.6000
    Epoch 40/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3855 - val_acc: 0.6000
    Epoch 41/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3837 - val_acc: 0.6000
    Epoch 42/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3818 - val_acc: 0.6000
    Epoch 43/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3800 - val_acc: 0.6000
    Epoch 44/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3782 - val_acc: 0.6000
    Epoch 45/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3765 - val_acc: 0.6000
    Epoch 46/100
    45/45 [==============================] - 0s - loss: 5.3729 - acc: 0.6667 - val_loss: 6.3747 - val_acc: 0.6000
    Epoch 47/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3730 - val_acc: 0.6000
    Epoch 48/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3711 - val_acc: 0.6000
    Epoch 49/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3693 - val_acc: 0.6000
    Epoch 50/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3675 - val_acc: 0.6000
    Epoch 51/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3658 - val_acc: 0.6000
    Epoch 52/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3641 - val_acc: 0.6000
    Epoch 53/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3625 - val_acc: 0.6000
    Epoch 54/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3609 - val_acc: 0.6000
    Epoch 55/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3595 - val_acc: 0.6000
    Epoch 56/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3580 - val_acc: 0.6000
    Epoch 57/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3566 - val_acc: 0.6000
    Epoch 58/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3553 - val_acc: 0.6000
    Epoch 59/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3540 - val_acc: 0.6000
    Epoch 60/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3529 - val_acc: 0.6000
    Epoch 61/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3522 - val_acc: 0.6000
    Epoch 62/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3514 - val_acc: 0.6000
    Epoch 63/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3505 - val_acc: 0.6000
    Epoch 64/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3498 - val_acc: 0.6000
    Epoch 65/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3491 - val_acc: 0.6000
    Epoch 66/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3485 - val_acc: 0.6000
    Epoch 67/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3479 - val_acc: 0.6000
    Epoch 68/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3474 - val_acc: 0.6000
    Epoch 69/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3467 - val_acc: 0.6000
    Epoch 70/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3463 - val_acc: 0.6000
    Epoch 71/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3458 - val_acc: 0.6000
    Epoch 72/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3451 - val_acc: 0.6000
    Epoch 73/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3445 - val_acc: 0.6000
    Epoch 74/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3441 - val_acc: 0.6000
    Epoch 75/100
    45/45 [==============================] - 0s - loss: 5.3728 - acc: 0.6667 - val_loss: 6.3435 - val_acc: 0.6000
    Epoch 76/100
    45/45 [==============================] - 0s - loss: 4.9170 - acc: 0.6889 - val_loss: 4.1193 - val_acc: 0.6000
    Epoch 77/100
    45/45 [==============================] - 0s - loss: 5.9025 - acc: 0.4889 - val_loss: 5.8147 - val_acc: 0.5333
    Epoch 78/100
    45/45 [==============================] - 0s - loss: 5.3822 - acc: 0.6667 - val_loss: 5.3762 - val_acc: 0.6667
    Epoch 79/100
    45/45 [==============================] - 0s - loss: 5.4691 - acc: 0.6222 - val_loss: 4.9733 - val_acc: 0.6000
    Epoch 80/100
    45/45 [==============================] - 0s - loss: 2.1073 - acc: 0.8000 - val_loss: 4.3309 - val_acc: 0.6667
    Epoch 81/100
    45/45 [==============================] - 0s - loss: 0.7885 - acc: 0.8444 - val_loss: 2.9247 - val_acc: 0.7333
    Epoch 82/100
    45/45 [==============================] - 0s - loss: 0.4819 - acc: 0.9556 - val_loss: 3.1835e-04 - val_acc: 1.0000
    Epoch 83/100
    45/45 [==============================] - 0s - loss: 1.7922 - acc: 0.8667 - val_loss: 1.4778 - val_acc: 0.8000
    Epoch 84/100
    45/45 [==============================] - 0s - loss: 0.4847 - acc: 0.9333 - val_loss: 0.5281 - val_acc: 0.9333
    Epoch 85/100
    45/45 [==============================] - 0s - loss: 0.3686 - acc: 0.9778 - val_loss: 0.7367 - val_acc: 0.9333
    Epoch 86/100
    45/45 [==============================] - 0s - loss: 0.4948 - acc: 0.9556 - val_loss: 0.6812 - val_acc: 0.9333
    Epoch 87/100
    45/45 [==============================] - 0s - loss: 0.3084 - acc: 0.9778 - val_loss: 0.4760 - val_acc: 0.9333
    Epoch 88/100
    45/45 [==============================] - 0s - loss: 0.0136 - acc: 1.0000 - val_loss: 0.1705 - val_acc: 0.9333
    Epoch 89/100
    45/45 [==============================] - 0s - loss: 3.1651e-04 - acc: 1.0000 - val_loss: 0.2161 - val_acc: 0.9333
    Epoch 90/100
    45/45 [==============================] - 0s - loss: 2.8808e-04 - acc: 1.0000 - val_loss: 0.3305 - val_acc: 0.9333
    Epoch 91/100
    45/45 [==============================] - 0s - loss: 1.8425e-04 - acc: 1.0000 - val_loss: 0.3816 - val_acc: 0.9333
    Epoch 92/100
    45/45 [==============================] - 0s - loss: 1.2954e-04 - acc: 1.0000 - val_loss: 0.4005 - val_acc: 0.9333
    Epoch 93/100
    45/45 [==============================] - 0s - loss: 1.0324e-04 - acc: 1.0000 - val_loss: 0.4085 - val_acc: 0.9333
    Epoch 94/100
    45/45 [==============================] - 0s - loss: 8.8324e-05 - acc: 1.0000 - val_loss: 0.4130 - val_acc: 0.9333
    Epoch 95/100
    45/45 [==============================] - 0s - loss: 7.8486e-05 - acc: 1.0000 - val_loss: 0.4163 - val_acc: 0.9333
    Epoch 96/100
    45/45 [==============================] - 0s - loss: 7.1322e-05 - acc: 1.0000 - val_loss: 0.4188 - val_acc: 0.9333
    Epoch 97/100
    45/45 [==============================] - 0s - loss: 6.5756e-05 - acc: 1.0000 - val_loss: 0.4209 - val_acc: 0.9333
    Epoch 98/100
    45/45 [==============================] - 0s - loss: 6.1240e-05 - acc: 1.0000 - val_loss: 0.4227 - val_acc: 0.9333
    Epoch 99/100
    45/45 [==============================] - 0s - loss: 5.7496e-05 - acc: 1.0000 - val_loss: 0.4243 - val_acc: 0.9333
    Epoch 100/100
    45/45 [==============================] - 0s - loss: 5.4306e-05 - acc: 1.0000 - val_loss: 0.4257 - val_acc: 0.9333





    <keras.callbacks.History at 0x10bb0c110>



---

### 모델 사용하기

학습한 모델을 평가해봅니다. 제네레이터에서 제공되는 샘플로 평가할 때는 evaluate_generator 함수를 사용하고, 예측할 때는 predict_generator 함수를 사용합니다. 


```python
# 모델 평가하기
print("-- Evaluate --")

scores = model.evaluate_generator(
            validation_generator, 
            val_samples = 15)

print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 모델 예측하기
print("-- Predict --")

output = model.predict_generator(
            validation_generator, 
            val_samples = 15)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

print(output)
```

    -- Evaluate --
    acc: 93.33%
    -- Predict --
    [[0.000 0.000 1.000]
     [1.000 0.000 0.000]
     [0.000 1.000 0.000]
     [0.000 0.000 1.000]
     [0.000 1.000 0.000]
     [1.000 0.000 0.000]
     [0.000 0.000 1.000]
     [0.000 0.000 1.000]
     [1.000 0.000 0.000]
     [1.000 0.000 0.000]
     [1.000 0.000 0.000]
     [0.000 0.000 1.000]
     [0.000 1.000 0.000]
     [0.000 0.002 0.998]
     [0.000 1.000 0.000]]


간단한 모델이고 데이터셋이 적은 데도 불구하고 93.33%라는 높은 정확도를 얻었습니다. 개수로 따지면 검증 샘플 15개 중 1개가 잘 못 분류가 되었네요. Predict 함수는 입력된 이미지에 대해서 모델의 결과를 알려주는 역할을 하는 데, 각 샘플별로 클래스별 확률을 확인할 수 있습니다. 각 열은 다음을 뜻합니다.
- 첫번째 열 : 원일 확률
- 두번째 열 : 사각형일 확률
- 세번째 열 : 삼각형일 확률

확인을 해보니 rectangle020.png 파일이 사격형이 아니라 삼각형으로 판정이 되었습니다. 사각형을 그릴 때는 몰랐었는데, 막상 모델에서 삼각형이라고 얘기하니 다른 사각형이랑 조금 다르게 그린 것 같습니다.

![predict](http://tykimos.github.com/Keras/warehouse/2017-3-8_CNN_Getting_Started_4.png)


---

### 전체 소스


```python
import numpy as np

# 랜덤시드 고정시키기
np.random.seed(5)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 데이터셋 불러오기
train_datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest')

img = load_img('/Users/tykimos/Projects/Keras/_writing/warehouse/handwriting_shape/validation/triangle/triangle018.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in train_datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='tri', save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
```

---

### 결론

본 강좌에서는 이미지 분류 문제를 직접 정의해보고 데이터셋도 직접 만들어봤습니다. 이미지 분류 문제에 높은 성능을 보이고 있는 컨볼루션 신경망 모델을 이용하여 직접 만든 데이터셋으로 학습 및 평가를 해보았습니다. 학습 결과는 좋게 나왔지만 이 모델은 한 사람이 그린 것에 대해서만 학습이 되어 있어서 다른 사람에 그린 모양은 잘 분류를 못할 것 같습니다. 이후 강좌에서는 다른 사람이 그린 모양으로 평가해보고 어떻게 모델 성능을 높일 수 있을 지 알아보겠습니다.

그리고 실제 문제에 적용하기 전에 데이터셋을 직접 만들어보거나 좀 더 쉬운 문제로 추상화해서 프로토타이핑 하시는 것을 권장드립니다. 객담도말된 결핵 이미지 판별하는 모델을 만들 때, 결핵 이미지를 바로 사용하지 않고, MNIST의 손글씨 중 '1'과 '7'을 결핵이라고 보고, 나머지는 결핵이 아닌 것으로 학습시켜봤었습니다. 결핵균이 간균 (막대모양)이라 적절한 프로토타이핑이었습니다. 

---

### 같이 보기

* [강좌 목차](https://tykimos.github.io/Keras/2017/01/27/Keras_Lecture_Plan/)
* 이전 : [딥러닝 모델 이야기/컨볼루션 신경망 레이어 이야기](https://tykimos.github.io/Keras/2017/01/27/CNN_Layer_Talk/)
* 다음 : [딥러닝 모델 이야기/순환 신경망 레이어 이야기]