---
layout: post
title:  "딥브릭 스튜디오 (DeepBrick Studio)"
comments: true
---
딥브릭에서 지원하는 블록들을 소개합니다.

|블록|이름|설명|
|:-:|:-:|:-|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Dataset_Vector_s.png)|Input data, Labels|1차원의 입력 데이터 및 라벨입니다.
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Dataset2D_s.png)|2D Input data|2차원의 입력 데이터입니다. 주로 영상 데이터를 의미하며, 너비, 높이, 채널수로 구성됩니다.

|블록|이름|설명|
|:-:|:-:|:-|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Dense_s.png)|Dense|모든 입력 뉴런과 출력 뉴런을 연결하는 전결합층입니다.
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Embedding_s.png)|Embedding|단어를 의미론적 기하공간에 매핑할 수 있도록 벡터화시킵니다.|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Conv1D_s.png)|Conv1D|필터를 이용하여 지역적인 특징을 추출합니다.|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_GlobalMaxPooling1D_s.png)|GlobalMaxPooling1D|여러 개의 벡터 정보 중 가장 큰 벡터를 골라서 반환합니다.|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_MaxPooling1D_s.png)|MaxPooling1D|입력벡터에서 특정 구간마다 값을 골라 벡터를 구성한 후 반환합니다.|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Conv2D_s.png)|Conv2D|필터를 이용하여 영상 특징을 추출하는 컨볼루션 레이어입니다.|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_MaxPooling2D_s.png)|MaxPooling2D|영상에서 사소한 변화가 특징 추출에 크게 영향을 미치지 않도록 해주는 맥스풀링 레이어입니다.|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Flatten_s.png)|Flatten|2차원의 특징맵을 전결합층으로 전달하기 위해서 1차원 형식으로 바꿔줍니다.
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_LSTM_s.png)|LSTM|Long-Short Term Memory unit의 약자로 순환 신경망 레이어 중 하나입니다.|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Dropout_1D_s.png)|Dropout|과적합을 방지하기 위해서 학습 시에 지정된 비율만큼 임의의 입력 뉴런(1차원)을 제외시킵니다.|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Dropout_2D_s.png)|Dropout|과적합을 방지하기 위해서 학습 시에 지정된 비율만큼 임의의 입력 뉴런(2차원)을 제외시킵니다.|

|블록|이름|설명|
|:-:|:-:|:-|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Activation_sigmoid_s.png)|sigmoid|활성화 함수로 입력되는 값을 0과 1사이의 값으로 출력시킵니다. 출력값이 특정 임계값(예를 들어 0.5) 이상이면 양성, 이하이면 음성이라고 판별할 수 있기 때문에 이진분류 모델의 출력층에 주로 사용됩니다.|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Activation_softmax_s.png)|softmax|활성화 함수로 입력되는 값을 클래스별로 확률 값이 나오도록 출력시킵니다. 이 확률값을 모두 더하면 1이 됩니다. 다중클래스 모델의 출력층에 주로 사용되며, 확률값이 가장 높은 클래스가 모델이 분류한 클래스입니다. 
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Activation_tanh_s.png)|tanh|LSTM의 출력 활성화 함수로 사용됩니다.|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Activation_Relu_s.png)|relu|활성화 함수로 주로 은닉층에 사용됩니다.|
|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Activation_relu_2D_s.png)|relu|활성화 함수로 주로 Conv2D 은닉층에 사용됩니다.|