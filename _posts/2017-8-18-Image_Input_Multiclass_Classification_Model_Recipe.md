---
layout: post
title:  "영상입력 다중클래스분류 모델 레시피"
author: 김태영
date:   2017-08-21 02:00:00
categories: Lecture
comments: true
image: http://tykimos.github.com/Keras/warehouse/2017-8-18-Image_Input_Binary_Classification_Model_Recipe_4m.png
---
#### 다중퍼셉트론 모델


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

print('loss_and_metrics : ' + str(loss_and_metrics))
```

    Using Theano backend.
    /Users/tykimos/Projects/Keras/venv/lib/python2.7/site-packages/ipykernel/__main__.py:15: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(units=64, activation="relu", input_dim=784)`
    /Users/tykimos/Projects/Keras/venv/lib/python2.7/site-packages/ipykernel/__main__.py:16: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(units=10, activation="softmax")`
    /Users/tykimos/Projects/Keras/venv/lib/python2.7/site-packages/keras/models.py:837: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      warnings.warn('The `nb_epoch` argument in `fit` '


    Epoch 1/5
    60000/60000 [==============================] - 2s - loss: 0.6839 - acc: 0.8234     
    Epoch 2/5
    60000/60000 [==============================] - 2s - loss: 0.3516 - acc: 0.9016     
    Epoch 3/5
    60000/60000 [==============================] - 2s - loss: 0.3055 - acc: 0.9128     
    Epoch 4/5
    60000/60000 [==============================] - 2s - loss: 0.2774 - acc: 0.9208     
    Epoch 5/5
    60000/60000 [==============================] - 2s - loss: 0.2554 - acc: 0.9268     
     9152/10000 [==========================>...] - ETA: 0sloss_and_metrics : [0.23934107850492001, 0.93169999999999997]


#### 컨볼루션 신경망 모델


```python
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# 1. 데이터셋 준비하기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
```

    Epoch 1/10
    60000/60000 [==============================] - 1266s - loss: 0.2610  
    Epoch 2/10
    27680/60000 [============>.................] - ETA: 174s - loss: 0.0946

---

### 같이 보기

* [강좌 목차](https://tykimos.github.io/Keras/lecture/)
