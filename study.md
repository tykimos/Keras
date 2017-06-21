---
layout: page
comments: true
---

딥러닝 관련 논문이나 오픈된 소스를 보면서 공부한 것을 공유하고자 합니다.

<div class="well">
    {% for post in site.categories['Study'] %}
        {% if post.title != null %}
            <li>
            <span>{{ post.date | date: "%b %d" }}</span>» <a href="{{ site.baseurl}}{{ post.url }}">
            {{ post.title }}</a>
            </li>
        {% endif %}
    {% endfor %}
    <br/>
    <br/>
</div>  
