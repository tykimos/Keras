---
layout: post
comments: true
title:  "Apache Solr 간단히 따라해보기"
date:   2016-11-02 15:09:58 +0900
categories: Apache
---
`Apache Solr`는 자료를 빠르게 검색을 하기 위한 인덱싱 서버입니다. 예를 들어, FTP 서버에 파일을 업로드할 때, id, name, type, owner, path 등의 메타정보를 Solr에 삽입하면, Solr는 제공한 메타정보를 기반으로 업로드한 파일을 인덱싱할 겁니다.

# Solr 설치 및 시작하기

Solr 설치 파일은 [Quick Start][quickstart]에서 다운로드 받아 압축 해제한 후 해당 폴더로 이동합니다.

    $ cd solr-6.2.1/

start 명령으로 Solr를 실행시킵니다.

    $ bin/solr start -e cloud -noprompt

Solr가 시작되면 아래 URL로 접속합니다.

    http://localhost:8983/solr

# Solr에 데이터 삽입하기

간단한 예제를 위해 `my_meta'라는 콜렉션을 만듭니다. 
    
    $ bin/solr create -c my_meta

> Solr에서는 `콜렉션(Collection)`이란 단위로 데이터를 관리합니다. 

    curl http://localhost:8983/solr/my_meta/update -d '
    [
      {
        "id" : "book_001",
        "title_t" : "미움받을 용기",
        "author_s" : "기시미 이치로"
      }
    ]'

> 여기서 `id` 속성은 반드시 정의되어야 하는 속성입니다. 참고로 `Document`는 Solr 용어로 속성들의 집합체를 말합니다. 

> `Curl`는 command line tool and library으로 command line이나 script에서 데이터를 전달할 때 사용됩니다.

위에서 삽입한 `book_001`을 `get`으로 확인합니다. 확인은 가능하나 실제로 삽입된 상태는 아닙니다. 

    http://localhost:8983/solr/my_meta/get?id=book_001

삽입 후에는 커밋을 해야 데이터가 실제로 삽입됩니다.

    curl http://localhost:8983/solr/my_meta/update/json?commit=true

쿼리를 통해 삽입된 데이터를 확인합니다.

    http://localhost:8983/solr/my_meta/query?q=*
    
기존에 삽입된 'book_001' 데이터에 필드를 추가할 수 있습니다.

    curl http://localhost:8983/solr/my_meta/update -d '
    [
      {
        "id"         : "book_001",
        "pubyear_i"  : { "add" : 2014 },
        "ISBN_s"     : { "add" : "978-8-9969-9134-2" }
        }
    ]'    

필드 삽입 후에는 커밋을 해야 실제로 반영됩니다.

    curl http://localhost:8983/solr/my_meta/update/json?commit=true

쿼리를 통해 필드가 삽입되었는 지 확인합니다.

    http://localhost:8983/solr/my_meta/query?q=*

수행결과는 다음과 같습니다.

```    
{
  "responseHeader":{
    "zkConnected":true,
    "status":0,
    "QTime":0,
    "params":{
      "q":"*"}},
  "response":{"numFound":1,"start":0,"docs":[
      {
        "id":"book_001",
        "title_t":["미움받을 용기"],
        "author_s":"기시미 이치로",
        "pubyear_i":2014,
        "ISBN_s":"978-8-9969-9134-2",
        "_version_":1549973564154707968}]
  }}
```

`book_002`를 추가해봅니다. 이 때 `book_001`에서는 정의하지 않은 `sales_i` 필드를 삽입합니다.

    curl http://localhost:8983/solr/my_meta/update -d '
    [
      {
        "id"        : "book_002",
        "title_t"   : "코스모스",
        "author_s"  : "칼 세이건",
        "pubyear_i" : 2006,
        "sales_i"     : 100000
      }
    ]'

    curl http://localhost:8983/solr/my_meta/update/json?commit=true

속성에 접미사는 다음의 타입을 정의합니다.

Field Suffix | Multivalued Suffix | Type | Description
---- | ---- | ---- | ----  
_t | _txt | text_general | Indexed for full-text search so individual words or phrases may be matched.
_s | _ss | string | A string value is indexed as a single unit. This is good for sorting, faceting, and analytics. It’s not good for full-text search.
_i | _is | int | a 32 bit signed integer
_l | _ls | long | a 64 bit signed long
_f | _fs | float | IEEE 32 bit floating point number (single precision)
_d | _ds | double | IEEE 64 bit floating point number (double precision)
_b | _bs | boolean | true or false
_dt | _dts | date | A date in Solr’s date format
_p | | location | A lattitude and longitude pair for geo-spatial search



# Solr에 데이터 쿼리하기

전체 데이터 쿼리는 다음과 같이 수행합니다.

    http://localhost:8983/solr/my_meta/query?q=*

`title_t`에 '용기'라는 단어가 포함된 데이터 쿼리합니다.

    http://localhost:8983/solr/my_meta/query?q=title_t:용기

`title_t`에 '용기'라는 단어가 포함된 데이터 쿼리해서 `author_s`와 `title_t` 속성만 표출합니다. 

    http://localhost:8983/solr/my_meta/query?q=title_t:용기&fl=author_s,title_t

`pubyear_i` 속성의 정보로 내림차순으로 정렬합니다.

    http://localhost:8983/solr/my_meta/query?q=*&sort=pubyear_i desc

정렬 쿼리 수행결과는 다음과 같습니다.

```
{
  "responseHeader":{
    "zkConnected":true,
    "status":0,
    "QTime":0,
    "params":{
      "q":"*",
      "sort":"pubyear_i desc"}},
  "response":{"numFound":3,"start":0,"docs":[
      {
        "id":"book_001",
        "title_t":["미움받을 용기"],
        "author_s":"기시미 이치로",
        "pubyear_i":2014,
        "ISBN_s":"978-8-9969-9134-2",
        "_version_":1549973564154707968},
      {
        "id":"book_002",
        "title_t":["코스모스"],
        "author_s":"칼 세이건",
        "pubyear_i":2006,
        "sales_i":100000,
        "_version_":1549974286054195200}]
  }}
```
# Solr 정지하기 

다음 명령으로 실행 중인 Solr를 정지시킬 수 있다.

    bin/solr stop -all

[quickstart]: http://lucene.apache.org/solr/quickstart.html

