---
title: "[Probabilistic_DL] Probabilistic Curve Fitting"
excerpt: "Bayesian Probabilistic Model에 대해 살펴보기"

categories:
  - statistics
tags:
  - [tag1,tag2]

permalink: /statistics/prob-curve-fitting/

toc: true
toc_sticky: true

date: 2024-01-13
last_modified_at: 2024-01-13
---

*카테고리를 통계에 넣을지 데싸에 넣을지 고민을 했지만 아무래도 모델보다는 수식적인 내용을 많이 담고 있어 Statistics에 담기로 했다.*

ESC의 첫 번째 세션으로 진행되었던 Probabilistic Model.  
그동안 흩어져있던 퍼즐들이 이제야 딱딱 끼워맞춰진 것 같아 몰입해서 공부할 수 있었던 것 같다. 수없이 들어본 그 이름 베이지안...  
대체 베이지안이 뭐길래 frequentist, bayesian 파를 나눠서 논쟁하고 있는지 그 정체에 대해 한 번 알아보도록 하자.  
***

### Contents
*1. Probability on ML*
  - Nonprobabilistic, Probabilistic, Bayesian Probabilistic 모델들을 각각 비교해본다.

*2. Curve Fitting*
  - 회귀 분석을 위와 같은 3가지로 나누어 최적의 가중치/계수(w)값을 찾아본다.

*3. Decision Theory*
  - Decision의 관점에서 w의 최적값을 찾아본다.

*4. Information Theory*
  - 정보이론의 관점에서 w의 최적값을 찾아본다.

그러면 아주 신기한 결론이 나온다:
**어떠한 방법으로 접근해도 모두 같은 w를 도출한다!**
그렇다면 이 w는 모든 성질들을 만족하고 모든 의문들에 대답이 가능한 대단한 아이가 아닐까 싶다.
  
### 1. Probability on ML
Nonprobabilistic, probabilistic, Bayesian Probablistic 3가지 모델을 나누는 것의 기준은 결과를 어떤 형태로 내보내는가에 따라 다른 것 같다.  
  
**Nonprobabilistic의 단점**
네비게이션 예시를 드는 것이 가장 이해가 쉬울 것이다. 특정 목적지에 도달하는 길을 보여줄 때 우리는 보통 네비게이션에서 더 빠르다고 하는 길을 선택해 갈 것이다.  
<사진>  
하지만 정말 19분짜리 길을 선택하는 것이 합리적인 판단일까? 실제로 각 경로를 따라갔을 때 걸리는 시간의 분포가 오른쪽과 같다고 하자.   
운전자가 $\alpha$분 보다 짧게 목적지에 도달해야 한다면 각 분포에서 $P(t\le \alpha)$의 넓이를 비교해봐야 할 것이다.  
따라서 Nonprobabilistic은 다른 말로 Deterministic view라고 할 수 있다. (값을 딱 하나로 제공)  

**Probabilistic vs Bayesian Probabilistic**
여기서 좌측의 probabilistic은 frequentist의 view라고 간주한다.  
두 사람을 분류하는 Binary Classifier가 있다고 가정해보자. 우리는 두 사람의 데이터에 대해서만 학습을 한 이 모델이, 아예 새로운 사람이 들어왔을 때 우리에게 그 사실을 알려주기 바란다.  
<사진> 
Probabilistic Model은 해당 데이터가 특정 클래스에 속할 확률을 나타내준다. 여기서 "0.7, 0.3? 차이가 별로 안 나니까 새로운 인물이라는 뜻인가보다!" 라는 결론을 쉽게 내려서는 안 된다. 모델의 성능에 따라, 혹은 데이터의 미세한 차이 때문에 모델을 돌릴 때마다 전혀 다른 값들이 등장하고는 한다. 실제로 특정 두 인물 A와 B의 차이에 대해서만 feature 구분을 했던 모델이 C에서 A와 비슷한 feature에 주목한다면 A 클래스로, b와 비슷한 feature에 주목한다면 반대로 분류하게 될테니 말이다.  
  
Bayesian Probabilistic은 여기에서 차이를 나타낸다. 새로운 인물이 들어왔을 때 확률값이 요동치는 모습을 보여주어 우리는 이 모델이 제대로 구분을 못하고 있구나!라는 결론을 내리는 것이 가능해진다. 즉, Bayesian은 Uncertainty를 제대로 반영한다고 볼 수 있을 것이다.   
자율주행 자동차, 암 진단 등 오류가 치명적인 분야에서는 uncertainty를 반영하는 이 bayesian이 필수적일 것이다. 모델이 100개의 데이터를 돌려 1개의 오류를 내는 것보다, 모델이 90개의 데이터를 확인하고 10게는 사람이 직접 확인해 오류가 없게 하는 것이 아직은 더 합리적이기 때문이다.   
  
Nonprob, prob, bayesian prob  
값, 분포, 분포의 분포  
이렇게 생각하니 이해가 잘 되었다.


