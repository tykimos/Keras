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
{% capture categories %}{% for category in site.categories %}{% unless forloop.last %},{% endunless %}{% endfor %}{% endcapture %}
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
