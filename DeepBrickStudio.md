---
layout: post
title:  "딥브릭 스튜디오 (DeepBrick Studio)"
comments: true
---
딥브릭에서 지원하는 데이터셋과 레이어를 소개합니다.

|종류|구분|상세구분|브릭|비고|
|:-:|:-:|:-:|:-:|
|데이터셋|1D|-|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Dataset_Vector_s.png)|수치|
|데이터셋|2D|-|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Dataset2D_s.png)|영상|
|레이어|Dense|-|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Dense_s.png)|MLP나 FC에 사용|
|레이어|Conv2D|-|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Conv2D_s.png)|CNN에 사용|
|레이어|MaxPooling2D|-|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_MaxPooling2D_s.png)|CNN에 사용|
|레이어|LSTM|-|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_LSTM_s.png)|RNN에 사용, 내부활성화함수로 tanh 사용|
|레이어|Flatten|-|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Flatten_s.png)|FC 연결용|
|레이어|Dropout|1D|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Dropout_1D_s.png)|과적합 방지용|
|레이어|Dropout|2D|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Dropout_2D_s.png)|과적합 방지용, CNN모델에 사용|
|레이어|Activation|relu|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Activation_Relu_s.png)|Dense 은닉층에 주로 사용|
|레이어|Activation|relu|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Activation_relu_2D_s.png)|Conv2D 은닉층에 주로 사용|
|레이어|Activation|sigmoid|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Activation_sigmoid_s.png)|이진분류 모델의 출력층에 주로 사용|
|레이어|Activation|hard_sigmoid|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Activation_hard_sigmoid_s.png)|LSTM 은닉층에 주로 사용|
|레이어|Activation|softmax|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Activation_softmax_s.png)|다중클래스분류 모델의 출력층에 주로 사용|
|레이어|Activation|tanh|![img](http://tykimos.github.com/Keras/warehouse/DeepBrick/Model_Recipe_Part_Activation_tanh_s.png)|-|