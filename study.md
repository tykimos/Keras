---
layout: page
comments: true
---

딥러닝 관련 논문이나 오픈된 소스를 보면서 공부한 것을 공유하고자 합니다.

<div class="home">
  <ul class="post-list">

  asdf1
    {% for post in site.posts %}

        <li>
            <span>{{ post.date | date: "%b %d" }}</span>» <a href="{{ site.baseurl}}{{ post.url }}">
            {{ post.title }}{{post.catalogies}}</a>
        </li>

        
    {% endfor %}
  </ul>
</div>
