---
layout: post
title:  "컨볼루션 신경망 레이어 이야기"
author: Taeyoung, Kim
date:   2017-01-27 04:00:00
categories: Keras
comments: true
image: http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_10.png
---
이번 강좌에서는 컨볼루션 신경망 모델에서 주로 사용되는 컨볼루션(Convolution) 레이어, 맥스풀링(Max Pooling) 레이어, Flatten 레이어에 대해서 알아보겠습니다. 각 레이어별로 레이어 구성 및 역할에 대해서 알아보겠습니다.

---

### 필터로 특징을 뽑아주는 컨볼루션 레이어

케라스에서 제공되는 컨볼루션 레이어 종류에도 여러가지가 있으나 영상 처리에 주로 사용되는 Convolution2D 레이어를 살펴보겠습니다. 레이어는 영상 인식에 주로 사용되며, 필터가 탑재되어 있습니다. 아래는 Convolution2D 클래스 사용 예제입니다.

    Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu')
    
주요 인자는 다음과 같습니다.
* 첫번째 인자 : 컨볼루션 필터의 수 입니다.
* 두번째 인자 : 컨볼루션 커널의 행 수 입니다.
* 세번째 인자 : 컨볼루션 커널의 열 수 입니다.
* border_mode : 경계 처리 방법을 정의합니다.
    * 'valid' : 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.
    * 'same' : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.
* input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다. 
    * (채널 수, 행, 열)로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다. 
* dim_ordering: 차원의 순서를 정의합니다. 
    * 'th' : input_shape을 정의할 때 채널 수를 나타내는 차원이 가장 먼저 나옵니다.
    * 'tf' : input_shape을 정의할 때 채널 수를 나타내는 차원이 가장 마지막에 나옵니다.
* activation : 활성화 함수 설정합니다.
    * 'linear' : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
    * 'relu' : rectifier 함수, 은익층에 주로 쓰입니다.
    * 'sigmoid' : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
    * 'softmax' : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.
    
입력 형태는 다음과 같습니다. 
* dim_ordering이 'th'인 경우 (샘플 수, 채널 수, 행, 열)로 이루어진 4D 텐서입니다.
* dim_ordering이 'tf'인 경우 (샘플 수, 행, 열, 채널 수)로 이루어진 4D 텐서입니다.

출력 형태는 다음과 같습니다.
* dim_ordering이 'th'인 경우 (샘플 수, 필터 수, 행, 열)로 이루어진 4D 텐서입니다.
* dim_ordering이 'tf'인 경우 (샘플 수, 행, 열, 필터 수)로 이루어진 4D 텐서입니다.
* 행과 열의 크기는 border_mode가 'same'인 경우에는 입력 형태의 행과 열의 크기가 동일합니다.

간단한 예제로 컨볼루션 레이어와 필터에 대해서 알아보겠습니다. 입력 이미지는 채널 수가 1, 너비가 3 픽셀, 높이가 3 픽셀이고, 크기가 2 x 2인 필터가 하나인 경우를 레이어로 표시하면 다음과 같습니다.

    Convolution2D(1, 2, 2, border_mode='valid', input_shape=(1, 3, 3))
    
이를 도식화하면 다음과 같습니다.

![lego_1](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_1.png)

필터는 가중치를 의미합니다. 하나의 필터가 입력 이미지를 순회하면서 적용된 결과값을 모으면 출력 이미지가 생성됩니다. 여기에는 두 가지 특성이 있습니다. 
* 하나의 필터로 입력 이미지를 순회하기 때문에 순회할 때 적용되는 가중치는 모두 동일합니다. 이를 파라미터 공유라고 부릅니다. 이는 학습해야할 가중치 수를 현저하게 줄여줍니다.
* 출력에 영향을 미치는 영역이 지역적으로 제한되어 있습니다. 즉 그림에서 y~0~에 영향을 미치는 입력은 x~0~, x~1~, x~3~, x~4~으로 한정되어 있습니다. 이는 지역적인 특징을 잘 뽑아내게 되어 영상 인식에 적합합니다. 예를 들어 코를 볼 때는 코 주변만 보고, 눈을 볼 때는 눈 주변만 보면서 학습 및 인식하는 것입니다.

#### 가중치의 수

이를 Dense 레이어와 컨볼루션 레이어와 비교를 해보면서 차이점을 알아보겠습니다. 영상도 결국에는 픽셀의 집합이므로 입력 뉴런이 9개 (3 x 3)이고, 출력 뉴런이 4개 (2 x 2)인 Dense 레이어로 표현할 수 있습니다. 

    Dense(4, input_dim=9))

이를 도식화하면 다음과 같습니다.

![lego_2](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_2.png)

가중치 즉 시냅스 강도는 녹색 블럭으로 표시되어 있습니다. 컨볼루션 레이어에서는 가중치 4개로 구성된 크기가 2 x 2인 필터를 적용하였을 때의 뉴런 상세 구조는 다음과 같습니다.

![lego_3](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_3.png)

필터가 지역적으로만 적용되어 출력 뉴런에 영향을 미치는 입력 뉴런이 제한적이므로 Dense 레이어와 비교했을 때, 가중치가 많이 줄어든 것을 보실 수 있습니다. 게다가 녹색 블럭 상단에 표시된 빨간색, 파란색, 분홍색, 노란색끼리는 모두 동일한 가중치(파라미터 공유)이므로 결국 사용되는 가중치는 4개입니다. 즉 Dense 레이어에서는 36개의 가중치가 사용되었지만, 컨볼루션 레이어에서는 필터의 크기인 4개의 가중치만을 사용합니다.

#### 경계 처리 방법

이번에는 경계 처리 방법에 대해서 알아봅니다. 컨볼루션 레이어 설정 옵션에는 `border_mode`가 있는데, 'valid'와 'same'으로 설정할 수 있습니다. 이 둘의 차이는 아래 그림에서 확인할 수 있습니다.

![lego_4](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_4.png)

'valid'인 경우에는 입력 이미지 영역에 맞게 필터를 적용하기 때문에 출력 이미지 크기가 입력 이미지 크기보다 작아집니다. 반면에 'same'은 출력 이미지와 입력 이미지 사이즈가 동일하도럭 입력 이미지 경계에 빈 영역을 추가하여 필터를 적용합니다. 'same'으로 설정 시, 입력 이미지에 경계를 학습시키는 효과가 있습니다.

#### 필터 수

다음은 필터의 개수에 대해서 알아봅니다. 입력 이미지가 단채널의 3 x 3이고, 2 x 2인 필터가 하나 있다면 다음과 같이 컨볼루션 레이어를 정의할 수 있습니다.

    Convolution2D(1, 2, 2, border_mode='same', input_shape=(1, 3, 3))
    
이를 도식화하면 다음과 같습니다.

![lego_5](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_5.png)

만약 여기서 사이즈가 2 x 2 필터를 3개 사용한다면 다음과 같이 정의할 수 있습니다.

    Convolution2D(3, 2, 2, border_mode='same', input_shape=(1, 3, 3))
    
이를 도식화하면 다음과 같습니다.

![lego_6](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_6.png)
    
여기서 살펴봐야할 것은 필터가 3개라서 출력 이미지도 필터 수에 따라 3개로 늘어났습니다. 총 가중치의 수는 3 x 2 x 2으로 12개입니다. 필터마다 고유한 특징을 뽑아 고유한 출력 이미지로 만들기 때문에 필터의 출력값을 더해서 하나의 이미지로 만들거나 그렇게 하지 않습니다. 필터에 대해 생소하신 분은 카메라 필터라고 생각하시면 됩니다. 스마트폰 카메라로 사진을 찍을 때 필터를 적용해볼 수 있는 데, 적용되는 필터 수에 따라 다른 사진이 나옴을 알 수 있습니다.

![filter](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_filter.png)

뒤에서 각 레이어를 레고처럼 쌓아올리기 위해서 약식으로 표현하면 다음과 같습니다.

![lego_7](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_7.png)

이 표현은 다음을 의미합니다.
* 입력 이미지 사이즈가 3 x 3 입니다.
* 2 x 2 커널을 가진 필터가 3개입니다. 가중치는 총 12개 입니다.
* 출력 이미지 사이즈가 3 x 3이고 총 3개입니다. 이는 채널이 3개다라고도 표현합니다.

다음은 입력 이미지의 채널이 여러 개인 경우를 살펴보겠습니다. 만약 입력 이미지의 채널이 3개이고 사이즈가 3 x 3이고, 사이즈가 2 x 2 필터를 1개 사용한다면 다음과 같이 컨볼루션 레이어를 정의할 수 있습니다.

    Convolution2D(1, 2, 2, border_mode='same', input_shape=(3, 3, 3))

이를 도식화하면 다음과 같습니다.

![lego_8](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_8.png)

필터 개수가 3개인 것처럼 보이지만 이는 입력 이미지에 따라 할당되는 커널이고, 각 커널의 계산 값이 결국 더해져서 출력 이미지 한 장을 만들어내므로 필터 개수는 1개입니다. 이는 Dense 레이어에서 입력 뉴런이 늘어나면 거기에 상응하는 시냅스에 늘어나서 가중치의 수가 늘어나는 것과 같은 원리입니다. 가중치는 2 x 2 x 3으로 총 12개 이지만 필터 수는 1개입니다. 이를 약식으로 표현하면 다음과 같습니다.

![lego_9](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_9.png)

이 표현은 다음을 의미합니다.
* 입력 이미지 사이즈가 3 x 3 이고 채널이 3개입니다.
* 2 x 2 커널을 가진 필터가 1개입니다. 채널마다 커널이 할당되어 총 가중치는 12개 입니다.
* 출력 이미지는 사이즈가 3 x 3 이고 채널이 1개입니다.

마지막으로 입력 이미지의 사이즈가 3 x 3이고 채널이 3개이고, 사이즈가 2 x 2인 필터가 2개인 경우를 살펴보겠습니다. 

    Convolution2D(2, 2, 2, border_mode='same', input_shape=(3, 3, 3))
    
이를 도식화하면 다음과 같습니다. 

![lego_10](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_10.png)

필터가 2개이므로 출력 이미지도 2개입니다. 약식 표현은 다음과 같습니다.

![lego_11](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_lego_11.png)

이 표현은 다음을 의미합니다.
* 입력 이미지 사이즈가 3 x 3 이고 채널이 3개입니다.
* 2 x 2 커널을 가진 필터가 2개입니다. 채널마다 커널이 할당되어 총 가중치는 3 x 2 x 2 x 2로 24개 입니다.
* 출력 이미지는 사이즈가 3 x 3 이고 채널이 2개입니다.

---

### 사소한 변화따위 무시해주는 MaxPooling Layer

컨볼루션 레이어의 출력 이미지에서 주요값만 뽑아 크기가 작은 출력 영상을 만듭니다. 이것은 지역적인 사소한 변화가 영향을 미치지 않도록 합니다. 

    MaxPooling2D(pool_size=(2, 2))

주요 인자는 다음과 같습니다.
* pool_size : 풀 크기를 지정합니다.

---

### 다차원을 일차원으로 바꿔주는 Flatten Layer

다차원의 텐서를 Dense 레이어에 넘기기 위해서 1차원으로 바꿔주는 역할을 수행합니다. 사용법은 간단합니다.

    Flatten()
    
이전 레이어의 출력 정보를 이용하여 입력 정보를 자동으로 설정되며, 출력 형태는 입력 형태에 따라 자동으로 계산되기 때문에 별도로 사용자가 파라미터를 지정해주지 않아도 됩니다.


```python
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

import numpy

from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
```

    Using Theano backend.



```python
model = Sequential()

model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
```




![svg](output_6_0.svg)



![model](http://tykimos.github.com/Keras/warehouse/2017-1-27_CNN_Layer_Talk_model.png)

---

### 결론

본 강좌를 통해 컨볼루션 모델에서 사용되는 주요 레이어에 대해서 알아보았습니다. 다음 강좌에는 레이어를 조합하여 실제로 컨볼루션 모델을 만들어봅니다.

---

### 같이 보기

* [강좌 목차](https://tykimos.github.io/Keras/2017/01/27/Keras_Lecture_Contents/)
* 이전 : [딥러닝 이야기/다층 퍼셉트론 모델 만들어보기](https://tykimos.github.io/Keras/2017/02/04/MLP_Layer_Getting_Started/)
* 다음 : [딥러닝 이야기/컨볼루션 신경망 모델 만들어보기]


```python

```
