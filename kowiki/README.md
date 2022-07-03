# 한국어 위키 말뭉치 생성

원본 깃 레포지토리 : 
<br>https://github.com/paul-hyun/web-crawler
<br>https://github.com/google/sentencepiece

참고 블로그 : 
<br>https://paul-hyun.github.io/vocab-with-sentencepiece/

## 환경
* Python(>=3.6)

```sh
$ pip install tqdm
$ pip install pandas
$ pip install bs4
$ pip install wget
$ pip install pymongo
```


## 한국어 위키 크롤링 (CSV)
* 위키피디아 한국어 버전을 크롤링 하는 기능 입니다.
* 위키파싱은 [wikiextractor](https://github.com/attardi/wikiextractor)의 WikiExtractor.py를 사용 했습니다.

```sh
$ python kowiki.py [--output]
```

#### 주요옵션
* output: 위키를 저장할 폴더 입니다. 기본값은 kowiki 입니다.

#### 결과
* 저장폴더/yyyymmdd.csv 형태로 날짜별로 저정됩니다.
* 컬럼은 [id/url/제목/내용] 순으로 구성 되어 있습니다.
* seperator는 \u241D를 사용 하였습니다.
```
id,url,title,text
5,https://ko.wikipedia.org/wiki?curid=5,"..."
...
```
* pandas를 이용하면 쉽게 사용할 수 있습니다.
```
csv.field_size_limit(sys.maxsize)
SEPARATOR = u"\u241D"
df = pd.read_csv(filename, sep=SEPARATOR, engine="python")
```

#### csv파일 변환
```sh
$ python csvtotxt.py
```

* 위키데이터의 경우 본문(text)에 제목(title)정보를 포함하고 있어서 제목과, 본문을 둘다 저장할 경우 내용이 중복되어 본문만 저장.
* 위키 문서별로 구분하기 위해 구분자로 줄바꿈을 4개 입력
```
import pandas as pd

in_file = "<path of input>/kowiki_yyyymmdd.csv"
out_file = "<path of output>/kowiki.txt"
SEPARATOR = u"\u241D"
df = pd.read_csv(in_file, sep=SEPARATOR, engine="python")
with open(out_file, "w") as f:
  for index, row in df.iterrows():
    f.write(row["text"]) # title 과 text를 중복 되므로 text만 저장 함
    f.write("\n\n\n\n") # 구분자
```

#### vocab model 생성
* sentencepiece 라이브러리를 이용하여 서브워드 토크나이저 모델 생성
```sh
$ python makevocab {option}
```
* option
    * --train : 토크나이저 모델 학습
    * --test : 생성된 모델 테스트