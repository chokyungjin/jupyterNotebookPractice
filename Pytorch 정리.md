# Pytorch

* 함수의 의미 뿐만 아니라 어떤 식으로 굴러가는지 쓰는 이유를 알아야 하고 메소드를 뜯어보면 결국 for , while 문일텐데 공부하면서 코드를 구현해볼것.
* 이 식이 왜 나왔는지 수식을 설명하고 이해할 것.

---
> ### 기본적인 import 라이브러리

```python

import torch
import torch.nn as nn
# 신경망 모델들이 포함되어있다.
# torch.nn은 미니 배치만을 지원한다
import torch.optim as optim
# 경사하강법 알고리즘이 포함되어있다.
import torch.nn.init as init
# 텐서에 초기값을 주기위해 필요한 함수들이있다.
import numpy as np
# 수학 계산을 python에서 쉽게 하기위한 라이브러리
import unidecode
# 파이썬에 unicode 사용
import string
import random
import re
# 파이썬에서 정규식을 지원하는 모듈
import time, math
import os
# OS 모듈은 환경 변수나 디렉터리, 파일 등의 OS 자원을 제어할 수 있게 해주는 모듈이다.
import matplotlib.pyplot as plt
# 그래프를 그리는 툴
import torch.utils as utils
import torch.utils.data as data
# SGD를 중심으로 한 파라미터 최적화 알고리즘이 구현돼 있다.
import torchvision.models as models
# 사전에 정의된 모델들의 서브패키지
import torchvision.utils as v_utils
import torchvision.datasets as dset
# MNIST같은 사전에 모델링된 데이터셋이 있다.
import torchvision.transforms as transforms
# torchvision은 유명한 영상처리용 데이터 셋, 모델, 이미지 변환기가 들어있는 패키지
# transform 에는 이미지 데이터를 자르거나 확대 및 다양하게 변형시키는 함수들이 구현되어 있다.
# transform.ToTensor()는 PIL 이미지나 Numpy 배열을 토치 텐서로 바꿔준다.
# dataset 모듈은 데이터를 읽어오는 역할, transforms는 불러온 이미지를 필요에 따라 변환해주는 역할
from torch.utils.data import DataLoader
# 데이터를 하나씩 전달하지 않고 원하는 배치 사이즈대로 묶어서 전달하거나 더 효율적인 학습을 위해 데이터를
# 어떤 규칙에 따라 정렬하거나 섞어줄때 해주는 역할
# DataLoader 설정
# 사용할 데이터
# 배치 사이즈 (batch_size)
# 섞을지 여부 (shuffle)
# 사용할 프로세스 개수 (num_workers)
# 마지막에 남는 데이터의 처리 여부 (drop_last)


```

> ### Numpy

```python
# 브로드캐스트 - 형상이 다른 배열끼리 계산할 때
# 2x2 행렬에 스칼라 곱 10 할때, 10을 2x2로 바꿔준다. 이 기능이 브로드 캐스트.
# 원소접근
X = np.array([[51,55],[14,19],[0,4]])
X = X.flatten()
# X를 1차원 배열로 변환시켜준다. (평탄화)
x = tf.placeholder(dtype=tf.float32,[2,4])
# 노드를 설계한다. 선언 후 그 다음 값을 전달한다. 다른 텐서를 placeholder에 맵핑 시키는 것

start = np.zeros(shape=len(char_list), dtype = int)
end = np.zeros(shape=len(char_list), dtype = int)
start = np.vstack([start,end])
# vertical
# start 에 [start,zero]의 배열을 행으로 추가한다.
start = np.hstack([start,end])
# horizontal
# start 에 [start,zero]의 배열을 열로 추가한다.
x.detach().numpy() 
# detach를 통해 분리하고, 텐서를 넘파이 배열로 바꾼다.

OrderedDict
# 딕셔너리는 순서가 없지만 OrderedDict은 순서가 있는 딕셔너리다

```

> ### hyperparameters

```python
num_data = 1000
# 사용할 데이터의 수
num_epoch = 500
# 경사하강법 반복 횟수, batch가 데이터 전체를 다 도는 횟수
batch_size = 256
# 배치사이즈 , data loader 로부터 한 iteration에 불러오는 데이터의 수
learning_rate = 0.0002
# 학습률
num_gpus = 1
# 병렬처리할 GPU의 갯수
x = init.uniform_(torch.Tensor(num_data,1), -10 , 10)
#-10 부터 10 까지의 균등하게 초기화. x 에는 -10 부터 10 까지의 숫자들이 무작위로 들어가있다. 

noise = init.normal_(torch.FloatTensor(num_data,1),std=1)
# 현실성을 반영하기위한 표준 정규 분포를 따르는 가우시안 노이즈. std = 표준편차 mean = 0 


```

> ### torch의 기본적인 메소드 , 기본 문법

```python
X = torch.Tensor(2,3)
# 2x3 짜리 난수 배열을 만들어준다. 
x = torch.tensor([[1,2,3],[4,5,6]])
# 2x3을 값을 초기화하여 배열을 만들어준다.

torch.sum
torch.abs
torch.add
torch.mul
torch.sub
torch.div
torch.pow
torch.exp
torch.log
torch.dot
# 두 텐서간의 프로덕트 연산
torch.mm
# 내적 단일 tensor로 계산을 한다.
torch,bmm(x,y)
# tensor 의 곱을 배치 단위로 처리한다. 맨 앞에 batch 차원은 무시하고 뒤에 요소들을 행렬곱
torch.t()
# 전치 tensor를 리턴한다
torch.transpose(x,1,2)
# 특정 dimension을 변경할 수 있다. x 의 1,2번째 차원을 교환
torch.max
torch.rand()
# 0~1 사이의 절대값 숫자
torch.randn()
# 정규분포 N(0,1) 에서 추출된 난수를 생성, 음수 혹은 절대값이 1보다 클수도 있다.
torch.FloatTensor(num_data,1)
torch.LongTensor()
torch.cat() 
# 복수의 텐서를 결합할 떄 사용
torch.stack([x,x,x,x],dim=0)
# stack 함수를 통해 텐서를 붙일수도 있다.
torch.chunk() , torch.split()
# 텐서를 여러개로 나눌 때 사용
torch.arange() 
# 주어진 범위 내의 정수를 순서대로 생성
torch.ones() 
# 주어진 사이즈의 1로 이루어진 텐서 생성
torch.zeros() 
# 주어진 사이즈의 0으로 이루어진 텐서 생성
torch.ones_like() 
# 사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
torch.zeros_like() 
# 사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
torch.clone()
# 자신과 같은 tensor를 복사함

torch.view()
# 텐서의 모양을 바꾸고 싶은 경우(reshape) view함수 사용한다.
# t1 = torch.ones(4,3)
# t2 = t1.view(3,4), t3 = t1.view(t2)
# [4,16] 이었으면 tensor.view(2,-1)를 거치면 [2,32]로 바꿔준다. 여기서 -1 은 알아서 계산하라는 뜻.

torch.squeeze(), torch.unsqueeze()
# squeeze() 함수는 차원의 원소가 1인 차원을 없애주고, unsqueeze() 함수는 인수로 받은 위치에 새로운 차원을 삽입한다. 디폴트는 1차원

x=torch.randn(4,3)
x=[1:3,:] # 행은 0,1,2 번, 열은 전부
x = torch.index_select(x,dim=1,index=torch.LongTensor([0,2]))
# 0,2 번째 열들만 뽑아낸다.
mask = torch.ByteTensor([0,0,1],[0,1,0])
out = torch.masked_select(x,mask)
# masked_select를 통해 뽑고자 하는 값들을 마스킹해서 선택할 수 있다.

torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
# cuda 사용 여부
model = CNN().to(device)
# model을 지정한 장치로 올린다.
model = nn.DataParallel(model)
# 여러개의 GPU를 사용하기 위해 DataParallel을 이용하여 모델을 병렬로 실행

x = init.uniform_(torch.FloatTensor(3,4), a=0,b=9)
# a~b까지의 범위로의 연속 균등 분포를 만들어낸다.
x2 = init.normal_(torch.FloatTensor(3,4), std=0.2)
# 정규 분포를 따라 텐서를 초기화
x3 = init.constant_(torch.FloatTensor(3,4), 3.1415)
# 지정한 값으로 텐서를 초기화

x= torch.tensor([1,2,3])
print(x,x.requires_grad)
# 기울기 계산여부
z = x + y 
z.sum().backward()
# 기울기 계산이 꺼져있으므로 backward를 하면 오류가 난다
x = torch.tensor([1,2,3],requires_grad=True) 
# 기울기 계산을 킨다.

with torch.no_grad():
# 기울기 계산이 켜져 있더라도 torch.no_grad()로 끌 수 있다.
# with를 사용해 해당 부분만 기울기 계산을 끔으로써 모델을 인퍼런스 모드로 사용할 수 있다.

autograd()
tensor([1,2,3,4],requires_grad=True)
# 텐서의 연산에 대해 자동으로 미분값을 구해주는 기능. 텐서 자료를 생성시 .requires_grad_(True)를 실행하면 그 텐서에 행해지는 모든 연산에 대한 미분값을 계산한다. 계산을 멈추고 싶으면 detach() 함수를 이용하면 된다,

transforms.Compose([
  # 데이터의 전처리 방식을 정해준다
transforms.Scale()
  # 한 축의 크기를 변경
transforms.Normalize()
  # 정규화
transforms.Resize()
  # 사이즈 재정의
transforms.CenterCrop()
  # 이미지의 정중앙을 숫자만큼 크롭한다
transforms.ToTensor(),
  # 텐서로 바꿔줌
                   ])

img = img.clamp()
# img 를 렐루를 한 번 거치게한다.

isinstance(1,int)
# 1이 int형인지 알아본다. return True
isinstance(m,nn.Conv2d)
# m이 nn.Conv2d형인지 알아본다.
generator()
# 이터레이터를 생성해주는 함수. 클래스에 __iter__, __next__ 또는 __getitem__ 메소드를 구현해야 하지만 제네레이터는 함수안에서 yield라는 키워드만 사용하면 된다.
*list(resnet.children())[0:-1]
# *은 언패킹으로 여러개의 물건을 하나의 박스에 패키징하듯이 여러개의 변수를 하나의 컨테이너 타입으로 묶어주는걸 패킹이라고 한다.


```

> ### nn.Module

```python
nn.Module
# 뉴럴 네트워크 모듈로서 파라미터를 GPU로 옮기거나 내보내기, 불러오기 등의 보조 작업을 이용하여 파라미터를 캡슐화
# 뉴럴 네트워크에서 layer에 해당. 상태나 학습가능한 weight을 저장한다
# 모든 뉴럴 네트워크의 기본클래스이다. 클래스를 만들 때 전체로 상속받는다

def forward(self,)
# Module를 상속받은 클래스는 뉴럴 네트워크의 정방향을 계산을 수행하는 forward() 메소드를 반드시 구현해야하만 한다. forward() 메소드는 model 오브젝트를 데이터와 함께 호출하면 자동으로 실행된다.

backward(retain_graph=True)
# 기울기 오차역전파 계산
# 기본적으로 기울기 계산은 그래프에 포함된 모든 내부 버퍼를 플러시한다.
# retain_graph = True 는 2번씩 역전파 할때 여부를 확인

model = nn.Linear(1,1)
# x,y는 1개의 특성을 가진 데이터.
# 입력 데이터에 대해서 선형 변환 y = Ax + b
# nn.Bilinear()도 있다.

model.parameters() 
# 위의 모델은 Linear 이다. 파라미터를 가져올때 사용

model = nn.Sequential()
# 레이어들을 순차적으로 담아주는 컨테이너라고 생각하자
# 입력값이 하나일 때, 각 레이어를 데이터가 순차적으로 지나갈 때 사용
model.named_children()
# 모델의 직속 자식 노드들을 불러온다

nn.ReLU()
# ReLU 활성화 함수
nn.LeakyReLU()
# 리키 렐루 활성화 함수

nn.Tanh()
# 하이퍼볼릭 탄젠트 활성화 함수
nn.Sigmoid()
# 시그모디으 활성화 함수

 
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# 필수 요소로는 in_channels, out_channels, kernel_size
# in_channels : input image 의 채널수 
# out_channels : Convolution에 의해서 생성된 채널의 수
# kernel_size : 필터의 크기
# stride : 스트라이드를 얼마나 줄 것인지
# padding : zero padding 을 input의 양쪽에 얼만큼 줄 건지
nn.ConvTranspose2d()
# 전치 합성곱 연산. 디코더와 같다. 
# 하나의 입력값을 받아 여기에 서로 다른 가중치를 곱해 필터의 크기만큼 입력값을 퍼트린다.

nn.MaxPool2d()
# kernal_size, stride , padding이 있다.
# 풀링할 때 최대값을 남겨서 풀링한다. nn.AvgPool2d()는 평균값으로 풀링한다.

nn.Dropout2d()
# 0~1 사이로 훈련시에 요소에 zero input tensor를 넣을 확률

nn.BatchNorm1d()
nn.BatchNorm2d()
# 배치 정규화

nn.init.kaiming_normal_()
# kaiming He 정규화
# Relu 함수를 사용시 kaming He 초기화를 사용하는것이 좋다.

nn.Embedding(총 단어의 개수, 임베딩 시킬 벡터의 차원)
# Embedding 모듈은 index를 표현하는 LongTensor를 인풋으로 기대하고 해당 벡터로 인덱싱
# 알파벳이나 단어같은 기본 단위 요소들을 일정한 길이를 가지는 벡터공간에 투영시키는 것

nn.LSTM
# 기본 RNN 은 timestamp 가 너무 길면 기울기 소실이 생기고 히든 사이즈를 고하지 않은 많은 스텝을 거쳐오면 정보가 희소해진다.
# 기울기 소실을 개선하기 위한 모델
# 기존의 은닉상태 뿐만 아니라 셀 상태라는 이름을 가지는 전달 부분을 추가하여 같이 recurrent

# 기존의 정의된 모델을 불러올 때
resnet = models.resnet50(pretrained = True)

```
<img width="500" alt="스크린샷 2019-11-08 오전 1 29 09" src="https://user-images.githubusercontent.com/46750574/68407632-41a9fb00-01c7-11ea-8f31-1fa5f17d132a.png">


> ### loss function

```python
loss_func = nn.L1Loss()
# L1 손실 함수는 절대값 평균 오차함수
loss_func = nn.CrossEntropyLoss()
# 크로스 엔트로피 손실함수
```

> ### Optimizer

```python
optimizer = optim.SGD(model.parameters(),lr=0.01)
# SGD 와 그냥 경사하강법의 차이?
# 경사하강법은 모든 훈련 데이터에서 대해서 값을 평가하고 매개변수 업데이트를 진행하기 때문에 속도가 느리다.
# Stochastic Gradient Descent 은 확률적으로 선택한 데이터에 대해서 값을 평가하고 매개변수를 업데이트를 하기때문에 경사하강법에 비해서 빠르다
optimizer.zero_grad()
# 학습 직전에 기울기를 0 으로 초기화해야 새로운 가중치와 편차에 대해서 새로운 기울기를 구할수있다. 
optimizer.step()
# 가중치 업데이트
```

