## Jupyter_Notebook Self_Practice

---

> 파이썬에서는 class 라는 키워드를 사용하여 클래스를 정의.

클래스 정의에는 __init__라는 특별한 메소드가 있는데, 클래스를 초기화 하는 방법을 정의한다. 초기화용 메소드를 생성자라고도 하며, 클래스의 인스턴스가 만들어질 때 한 번만 불린다. 또 파이썬에서는 메소드의 첫 번째 인수로 자신을 나타내는 self를 명시적으로 쓰는 것이 특징이다. 

---

### numpy

* 외부 라이브러리, 배열 , 행렬 계산시 필요
* x= np.array([1.0,2.0,3.0])
  y= np.array([2.0,3.0,4.0])
* x+y , x -y , x * y , x / y
* 브로드캐스트 - 형상이 다른 배열끼리 계산할 때
  * 2x2 행렬에 스칼라 곱 10 할때, 10을 2x2로 바꿔준다. 이 기능이 브로드 캐스트.
* 원소접근
  * X = np.array([[51,55],[14,19],[0,4]])
  * X = X.flatten() , X를 1차원 배열로 변환시켜준다. (평탄화)

---

## Pandas 정리

Pandas는 데이터 분석을 위해 쓰는 패키지 주로 딕셔너리를 사용할 때 쓴다


### 시리즈 정의

데이터를 리스트나 1차원 배열 형식으로 series 클래스 생성자에 넣어주면 시리즈 클래스 객체를 만들 수 있다.

시리즈 클래스는 numpy에서 제공하는 1차원 배열과 비슷하지만 각 데이터의 의미를 표시하는 인덱스를 붙일 수 있다. 데이터 자체는 값이라고 한다.

시리즈 = 인덱스 + 값

* 시리즈 

```python
obj = pd.Series([4,5,-2,8])
obj.values # 시리즈의 값만 확인한다
obj.index # 시리즈의 인덱스 확인
obj.dtypes # 시리즈의 데이터 타입 확인하기
```

* 시리즈의 인덱스의 길이는 데이터의 길이와 같아야 하며, 인덱스의 값은 인덱스 라벨
* 인덱스 라벨은 문자열 뿐 아니라 날짜, 시간, 정수 등도 가능하다
* 파이썬의 딕셔너리 자료형을 시리즈 데이터로 만들 수 있다. 
* 딕셔너리의 키가 시리즈의 인덱스가 된다.

```python
obj = pd.Series([4,5,-2,8], index= ["a","b","c","d"])
data = {"Kim": 35000, "Park": 67000, "Joon": 12000, "Choi": 4000}
obj = pd.Series(data)

```

* 시리즈의 이름 지정 및 인덱스의 이름도 지정할 수 있다.

```python
obj2.name = "Salary" 
obj2.index.name = "Names" 

```

### 시리즈 연산

* numpy 배열처럼 시리즈도 벡터화 연산 가능
* 시리즈의 값에만 적용되며 인j덱스 값은 변하지 않는다

시리즈 인덱싱

* 시리즈는 numpy 배열의 인덱스 방법처럼 사용 외에 인덱스 라벨을 이용한 인덱싱
* 배열 인덱싱은 자료의 순서를 바꾸거나 특정한 자료만 선택 가능
* 라벨 값이 영문 문자열인 경우에는 마치 속성인것처럼 점(.)을 이용하여 접근

```python
Obj * 10 	 # 스칼라 배
obj * obj1 # 인덱싱 끼리 연산
obj1.values + obj1.values
a = pd.Series([1024, 2048, 3096, 6192],
              index=["서울", "부산", "인천", "대구"])

a[1] = a["부산"] # 같다!
obj1.A # 처럼 접근
```

시리즈 슬라이싱

* 배열 인덱싱이나 인덱스 라벨을 이용한 슬라이싱도 가능하다
* 문자열 라벨을 이용한 슬라이싱은 콜론 기호 뒤에 오는 인덱스에 해당하는 값이 결과에 포함

```python
a[1:3] != a ["부산" : "대구"] # 다른 결과가 나온다
```

시리즈의 데이터 갱신, 추가, 삭제

* 인덱싱을 이용하여 딕셔너리처럼 데이터를 갱신하거나 추가한다
* 데이터 삭제시 딕셔너리처럼 del 명령을 사용한다

```python
a["부산"] = 1234
del a["서울"]

```

**데이터프레임(DataFrame)**

* 시리즈가 1차원 벡터 데이터에 행 방향 인덱스라면, 데이터 프레임 클래스는 2차원 행렬 데이터에 합친 것으로 행 인덱스와 열 인덱스를 지정.
* 즉, 데이터프레임 = 시리즈 { index + value } +시리즈 + 시리즈의 연속체
* 데이터프레임은 공통 인덱스를 가지는 열 시리즈를 딕셔너리로 묶어놓은 것이다.
* 데이터프레임은 numpy의 모든 2차원 배열 속성이나 메소드를 지원한다

> 생성

* 하나의 열이 되는 데이터를 리스트나 일차원 배열을 준비
* 각 열에 대한 이름의 키를 갖는 딕셔너리를 생성
* pandas의 DataFrame 클래스로 생성
* 열 방향 인덱스는 columns 인수로 , 행 방향 인덱스는 index 인수로 지정

```python
# Data Frame은 python의 dictionary 또는 numpy의 array로 정의
data = {
'name': ["Choi", "Choi", "Choi", "Kim", "Park"], 
'year': [2013, 2014, 2015, 2016, 2017], 
'points': [1.5, 1.7, 3.6, 2.4, 2.9]
} 
df = pd.DataFrame(data) 

df.index # 행 방향의 index
df.columns # 열 방향의 index
df.values # 값들을 출력

```

데이터프레임 열 갱신 추가

* 데이터프레임은 열 시리즈의 딕셔너리로 볼 수 있으므로 열 단위로 데이터를 갱신하거나 추가, 삭제
* data에 포함되어 있지 않은 값은 nan으로 나타내는 null 과 같은 개념이다
* 딕셔너리, numpy의 배열, 시리즈의 다양한 방법으로 추가 가능하다

```python
# DataFrame을 만들면서 columns와 index를 설정
df = pd.DataFrame(data, columns=["year", "name", "points", "penalty"],
                                  index=(["one", "two", "three", "four", "five"]))

df[["year","points"]] # 특정 열만 선택한다
df["penalty"] = 0.5 # 특정 열을 선택하고 값(0.5)을 대입한다
df["penalty"] = [0.1,0.2,0.3,0.4,0.5] #특정 열에 대한 값을 리스트로 대입할 수 있다.

import numpy as np
df['zeros'] = np.arrange(5) # numpy 의 np.arrange로 새로운 열을 추가할 수 있다.

# 인덱스 인자로 특정행을 지정하여 시리즈로 추가 가능하다
val = pd.Series([-1.2,-1.5,-1.7], index=['two','four','five'])
df['debt'] = val

df['net_points'] = df['points'] - df['penalty'] # 연산후 새로운 열 추가하기
df['high_points'] = df['net_points'] > 2.0 # 조건 연산으로 열 추가

del df['high_points'] # 열 삭제하기

df.columns # 컬럼명 확인하기
# 인덱스와 컬럼 이름 지정
df.index.name = "Order"
df.columns.name = "Info"

```

데이터프레임 인덱싱

열 인덱싱

* 데이터프레임을 인덱싱 할 때도 열 라벨을 키 값으로 생각하여 인덱싱한다
* 인덱스로 라벨 값 하나만 넣으면 시리즈 객체가 반환되고 라벨의 배열 또는 리스트를 넣으면 부분적인 데이터프레임이 반환
* 하나의 열만 빼내면서 데이터프레임 자료형을 유지하고 싶다면 원소가 하나인 리스트를 써서 인덱싱

행 인덱싱

* 행 단위로 인덱싱을 하고자 하면 항상 슬라이싱을 해야 한다.
* 인덱스의 값이 문자 라벨이면 라벨 슬라이싱

```python
# 열 인덱싱
df["year"] 
# 다른 방법의 열 인덱싱
df.year
# 행 인덱싱은 슬라이싱으로 0번째부터 1번째로 지정하면 1행을 반환
df[0:1]
# 행 인덱싱 슬라이싱으로 0번째 부터 2(3-1) 번째까지 반환
df[0:3]
```

loc 인덱싱

* 인덱스의 라벨값 기반의 2차원 (행,렬) 인덱싱

```python
# .loc 함수를 사용하여 시리즈로 인덱싱
df.loc["two"]
# .loc 또는 .iloc 함수를 사용하여 데이터프레임으로 인덱싱
df.loc["two":"four"]
df.loc["two":"four", "points"] 
df.loc["three":"five","year":"penalty"] 

```

iloc 인덱싱

* 인덱스의 숫자 기반의 2차원 (행,열) 인덱싱

```python
# 새로운 행 삽입하기 
df.loc['six',:] = [2013,'Jun',4.0,0.1,2.1] 
# 4번째 행을 가져오기 위해 .iloc 사용:: index 번호를 사용
df.iloc[3]
# 슬라이싱으로 지정하여 반환
df.iloc[3:5, 0:2]
# 행을 전체, 열은 두번째열부터 마지막까지 슬라이싱으로 지정하여 반환
df.iloc[:,1:4] 
```

Boolean 인덱싱

```python
df["year"] > 2014 # year > 2014보다 큰 불린 값
df.loc[df['name'] == "Choi", ['name','points']]
```



데이터프레임 다루기

Numpy randn 데이터프레임 생성

```python
# DataFrame을 만들때 index, column을 설정하지 않으면 기본값으로 0부터 시작하는 정수형 숫자로 입력된다. 
df = pd.DataFrame(np.random.randn(6,4)) 

df.columns = ["A","B","C","D"]
#pandas에서 제공하는 date range함수는 datetime 자료형으로 구성된, 날짜/시간 함수 
df.index = pd.date_range('20160701', periods=6)

```

numpy로 데이터프레임 결측치 다루기

```python
# np.nan은 NaN값을 의미
df["F"] = [1.0, np.nan, 3.5, 6.1, np.nan, 7.0] 
# 행의 값중 하나라도 nan인 경우 그 행을 없앤다. 
df.dropna(how="any") 
# 행의 값의 모든 값이 nan인 경우 그 행을 없앤다.
df.dropna(how='all')
# NaN에 특정 value 값 넣기
df.fillna(value=0.5)
```

drop 명령어

```python
# 특정 행 drop하기 
df.drop(pd.to_datetime('20160701'))
# 2개 이상도 가능하다
df.drop([pd.to_datetime('20160702'),pd.to_datetime('20160704')])
# 특정 열 삭제하기 
df.drop('F', axis = 1)
# 2개 이상의 열도 가능 
df.drop(['B','D'], axis = 1) 
```

### 파일 읽어오기

```python
df = pd.read_csv('filename.csv')
# c1을 인덱스로 불러오기
pd.read_csv('data/sample1.csv', index_col="c1")
# index를 False로 주면 index는 빼고 저장한다
df.to_csv('filename',index = False)
# 인터넷 링크의 데이터 불러오기
titanic = pd.read_excel("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls")
titanic.head()
```

### 데이터 처리하기

정렬

* 데이터를 정렬로 sort_index는 인덱스 값을 기준으로, sort_values는 데이터 값을 기준으로 정렬한다

```python
# np.random으로 시리즈 생성
s = pd.Series(np.random.randint(6, size=100))
# value_counts 메서드로 값을 카운트
s.value_counts()
# sort_index 메서드로 정렬하기
s.value_counts().sort_index()
# ascending=False 인자로 내림차순 정리
s.sort_values(ascending=False)
```

Apply 함수

* 행이나 열 단위로 더 복잡한 처리를 하고 싶을 때는 apply 메소드를 사용
* 인수로 행 또는 열을 받는 함수를 apply 메소드의 인수로 넣으면 각 열(행) 을 반복하여 수행
* 람다함수 - 익명함수

```python
df = pd.DataFrame({
    'A': [1, 3, 4, 3, 4],
    'B': [2, 3, 1, 2, 3],
    'C': [1, 5, 2, 4, 4]
})
# 람다 함수 사용
df.apply(lambda x: x.max() - x.min())
# 만약 행에 대해 적용하고 싶으면 axis=1 인수 사용
df.apply(lambda x: x.max() - x.min(), axis=1)
```

Describe 메소드

* DataFrame의 계산 가능한 값들의 통계값을 보여준다

Pandas 시계열 분석

Pd.to_datetime 함수

* 날짜/시간을 나타내는 문자열을 자동으로 datetime 자료형으로 바꾼 후 datetimeindex 자료형 인덱스를 생성한다

```python
date_str = ["2018, 1, 1", "2018, 1, 4", "2018, 1, 5", "2018, 1, 6"]
idx = pd.to_datetime(date_str
# 인덱스를 사용하여 시리즈나 데이터프레임을 생성
np.random.seed(0)
s = pd.Series(np.random.randn(4), index=idx)
pd.date_range("2018-4-1", "2018-4-5")
pd.date_range(start="2018-4-1", periods=5)

```


