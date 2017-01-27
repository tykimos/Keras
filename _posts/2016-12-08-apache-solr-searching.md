---
layout: post
comments: true
title:  "Apache Solr 검색기능"
date:   2016-12-08 15:09:58 +0900
categories: Apache
---

http://projectblacklight.org/#examples

이번에는 `Apache Solr`의 다양한 검색 기능에 대해서 알아보겠습니다. 쿼리는 잘 알려진 프로그래밍 언어는 물론 REST 클라이언트, cURL, wget, Chrome POSTMAN 등을 통해서도 쿼리를 수행할 수 있습니다.

또한 관리자용 웹페이지를 아래와 같이 제공하고 있습니다. `http://localhost:8983/solr/#/gettingstarted/query` 이 웹페이지를 통해 검색 기능을 테스트할 예정입니다.

검색 기능을 실행하기 전에 Solr실행과 샘플 데이터를 넣습니다.

```
bin/solr start -e cloud -noprompt
bin/post -c gettingstarted docs/
bin/post -c gettingstarted example/exampledocs/
```

아무 설정없이 쿼리 버튼을 클릭하면 10개의 결과가 나오는 것을 확인할 수 있습니다. 사용자가 설정한 쿼리에 대응하는 URL은 아래와 같이 우측 상단에 표시돕니다. 이 URL은 cURL를 통해 바로 사용될 수 있습니다. 

```
http://localhost:8983/solr/gettingstarted/select?indent=on&q=*:*&wt=json
```

## 기본 검색

`q` 칸에 'foundation'을 입력하여 쿼리를 수행하면, '4153'건의 항목이 검색됩니다. 쿼리는 아래와 같이 생성됩니다.

```
http://localhost:8983/solr/gettingstarted/select?indent=on&q=foundation&wt=json
```
화면에 10개만 출력되는 것은 rows의 기본 값이 10이기 때문이고, 사용자가 rows값을 지정하면, 해당하는 수만큼 결과가 나옵니다. 아래는 rows값을 100으로 지정했을 때의 URL입니다.
```
http://localhost:8983/solr/gettingstarted/select?indent=on&q=*:*&rows=100&wt=json
```
결과 중 원하는 필드만 보고싶다면, `fl`칸에 필드명을 입력합니다. 아래는 `id`필드만 결과를 보고싶을 때의 URL 입니다.

```
http://localhost:8983/solr/gettingstarted/select?fl=id&indent=on&q=*:*&wt=json
```
`q`칸에 'foundation'이라고 입력했다면, 인덱싱된 모든 필드에서 해당 문자열을 검색하여 결과를 보여줍니다. 만약 특정 필드에서 검색하길 원한다면, `q=field:value`으로 지정하시면 됩니다. 만약 문구 검색을 하고 싶다면 `q="multiple terms here"` 따옴표을 이용합니다.

## Faceting 검색

Apache Solr의 좋은 기능 중 하나는 `Faceting`입니다. Faceting은 검색결과를 범주(subsets, buckets, categories)로 분류해주고, 각 범주에 해당하는 결과를 카운팅 해줍니다.

Faceting에는 여러가지 타입이 있습니다. 
- 필드 이름
- 숫자
- 날짜 기간
- pivots (decision tree)
- 임의 쿼리 faceting

`facet` 체크박스을 설정하면 facet와 관련된 옵션을 나타납니다.

```
http://localhost:8983/solr/gettingstarted/select?facet.field=manu_id_s&facet=on&indent=on&q=*:*&wt=json
```

```
http://localhost:8983/solr/gettingstarted/select?q=*:*&wt=json&indent=on&rows=0&facet=true&facet.field=price
```

```
http://localhost:8983/solr/gettingstarted/select?q=*:*&wt=json&indent=on&rows=0&facet=true&facet.range=price&f.price.facet.range.start=0&f.price.facet.range.end=3000&f.price.facet.range.gap=50&facet.range.other=after
```

```
http://localhost:8983/solr/gettingstarted/select?q=*:*&rows=0&wt=json&indent=on'\
'&facet=on&facet.pivot=cat,inStock
```

```
http://localhost:8983/solr/exampledoc/select?hl.fl=series_t&hl=on&indent=on&q=of&wt=json
```

```
http://localhost:8983/solr/gettingstarted/browse
```


