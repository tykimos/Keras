---
layout: post
title:  "딥러닝 공부"
author: 김태영
date:   2017-01-27 00:00:00
categories: Study
comments: true
---
딥러닝 관련 논문이나 오픈된 소스를 보면서 공부한 것을 공유하고자 합니다.

<div class="well">
{% capture categories %}{% for category in site.categories %}{% if categories != 'Study' %}{% continue %}{% unless forloop.last %},{% endunless %}{% endfor %}{% endcapture %}
{% assign category = categories | split:',' | sort %}
{% for item in (0..site.categories.size) %}{% unless forloop.last %}
{% capture word %}{{ category[item] | strip_newlines }}{% endcapture %}
<h2 class="category" id="{{ word }}">{{ word }}</h2>
<ul>
{% for post in site.categories[word] %}{% if post.title != null %}
<li><span>{{ post.date | date: "%b %d" }}</span>» <a href="{{ site.baseurl}}{{ post.url }}">{{ post.title }}</a></li>
{% endif %}{% endfor %}
</ul>
{% endunless %}{% endfor %}
<br/><br/>
</div>


1. 딥러닝 이야기
    * [케라스 이야기](https://tykimos.github.io/Keras/2017/01/27/Keras_Talk/)
    * [학습과정과 데이터셋 이야기](https://tykimos.github.io/Keras/2017/03/25/Dataset_and_Fit_Talk/)
    * [평가 이야기](https://tykimos.github.io/Keras/2017/05/22/Evaluation_Talk/)
    * [오프라인 설치](https://tykimos.github.io/Keras/2017/03/15/Keras_Offline_Install/)
1. 딥러닝 모델 이야기
    * [다층 퍼셉트론 레이어 이야기](https://tykimos.github.io/Keras/2017/01/27/MLP_Layer_Talk/)
    * [다층 퍼셉트론 모델 만들어보기](https://tykimos.github.io/Keras/2017/02/04/MLP_Getting_Started/)
    * [컨볼루션 신경망 레이어 이야기](https://tykimos.github.io/Keras/2017/01/27/CNN_Layer_Talk/)
    * [컨볼루션 신경망 모델 만들어보기](https://tykimos.github.io/Keras/2017/03/08/CNN_Getting_Started/)
    * [순환 신경망 레이어 이야기]    
    * [순환 신경망 모델 만들어보기]
1. 딥러닝 성능 높히기
    * [TensorBoard 사용해보기]
    * [컨볼루션 신경망 모델을 위한 데이터 부풀리기](https://tykimos.github.io/Keras/2017/03/08/CNN_Data_Augmentation/) 
    * [드롭아웃 레이어 사용해보기]
    * [최적 모델 선택하는 법]
    * [경사하강법과 미니배치]
    * [과적합 줄이는 법]
1. 딥러닝 분류 해보기
    * [이진 분류 해보기-MLP]
    * [다중 분류 해보기-MLP]
    * [영상 인식 해보기-CNN]
    * [감성 분석 해보기-MLP]
    * [감성 분석 해보기-CNN]    
1. 딥러닝 회귀 해보기
    * [회귀 해보기-MLP]
    * [시계열 예측 해보기-MLP]
    * [시계열 예측 해보기-RNN]
1. 딥러닝 영상처리 해보기
    * [딥러닝으로 객체 검출 해보기]
    * [딥러닝으로 객체 분할 해보기]    
1. 딥러닝 생성 모델 해보기
    * [오토인코더 모델 만들어보기]    
    * [Variational Autoencoder (VAE) 해보기]
    * [Generative Adversarial Network (GAN) 해보기]   
1. 딥러닝 아트 해보기
    * [딥러닝으로 소설쓰기]
    * [딥러닝으로 꿈꾸기]
    * [딥러닝으로 그림그리기]
    * [딥러닝으로 화풍바꾸기]
1. 딥러닝 실무 적용하기
    * [학습 모델 저장하기/불러오기](https://tykimos.github.io/Keras/2017/06/10/Model_Save_Load/)

---

### 같이 보기

* 다음 : [딥러닝 이야기/케라스 이야기](https://tykimos.github.io/Keras/2017/01/27/Keras_Talk/)
