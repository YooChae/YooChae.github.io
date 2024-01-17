---
title: "[PRML] Decision Theory"
excerpt: "Pattern Recognition and Machine Learning 1.5 내용 정리"

categories:
  - statistics
tags:
  - [tag1,tag2]

permalink: /statistics/decision-theory/

toc: true
toc_sticky: true

date: 2024-01-17
last_modified_at: 2024-01-17
---
## Decision Theory

불확실성 속에서 Optimal Decision을 결정할 수도 있도록 하는 것이 decision theory

### 1. Motivation

Training Data (**x**, **t**)가 주어져 있을 때 

- *p*(**x**, **t**) (→ data의 완전한 분포)는 uncertainty에 대한 완벽한 요약을 제공
    - 적분을 통해 posterior, likelihood 구하기 가능
    - *p*(**x**, **t**)를 찾는 과정(inference)은 매우 어려움
- 환자의 암 여부 판단 decision 내리기
    
    $$
    Bayes'\:Thm. \rightarrow\: p(C_k|\textbf{x})=\dfrac{p(\textbf{x}|C_k)P(C_k)}{p(\textbf{x})}
    $$
    
    - 암이 존재하면 t = 0 → C1 class, 암이 없으면 t = 1 → C2 class 라 하자.
    - 합리적인 판단 결과는 사진(**x**)을 보고 암일 확률과 아닐 확률을 비교해보는 것
        
        → Posterior *p*(Ck | **x** )가 큰 것을 고르면 되지 않을까? **2-1**에서 증명해보자!
        

### 2-1. Minimizing the Misclassification Rate

**목표: 오분류 줄이기**

- **Decision Region** R1, R2를 잡아서 R1에 해당하는 point는 모두 C1으로, R2에 해당하는 point는 C2로 분류하자. Decision Region들의 경계를 **Decision Boundary**라고 한다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/26af9838-41a7-400d-9c93-f993565da096/ed036219-8a40-45fb-9c15-de9c701e700b/Untitled.png)

- 오분류 계산
    
    $$
    \begin{aligned}p(mistake) &= p(\textbf{x} \in R_1,C_2)+ p(\textbf{x} \in R_2,C_1)\\&= \int_{R_1}p(\textbf{x},C_2) d\textbf{x}+\int_{R_2}p(\textbf{x},C_1)d\textbf{x} \end{aligned}
    $$
    
    - 그림으로 확인해보면
        - x(hat)을 decision boundary로 잡았을 때: 오분류 = Red + Green + Blue
        - x_0를 decision boundary로 잡았을 때 오분류 = Green + Blue ← 가장 최소일 때
    - 따라서 어떤 **x**에 대해서 *p*(**x**, C1), *p*(**x**, C2) 중 더 큰 값으로 클래스를 분류하는 것이 위의 오분류를 줄이는 방법이다.
    
    $$
    p(\textbf{x},C_k)=p(C_k|\textbf{x})p(\textbf{x})
    $$
    
    - *p*(**x**)는 공통이기 때문에 *p*(**x**, Ck)가 크다는 것은 posterior인 *p*(Ck | **x**)가 큰 것과 같다.
- Multiclass로 확장
    
    $$
    p(correct) = \sum_{k=1}^Kp(\textbf{x}\in R_k, C_k) = \sum_{k=1}^K\int_{R_k}p(\textbf{x},C_k)d\textbf{x}
    $$
    
    $\therefore$ *p*(correct)를 최대화하는 것은 또 *p*(**x**, C_k)가 큰 class에 할당되도록 decision region을 설정
    

> **요약** 💡
1. 오분류를 줄이는 것은 *p*(**x**, C_k)이 큰 class에 배정하는 것과 같다.
2. 이는 결국 *p*(C_k | **x**)를 비교하는 것과 같다.
> 

### 2-2. Minimizing the Expected Loss

- 오분류를 방지하되 1종 오류보다 2종 오류 고려를 더 우선시하고 싶다면? → Loss Matrix 도입
- 발생 가능한 Loss를 줄인다 = Total Loss를 줄인다 = Average Loss를 줄인다
    
    $$
    E[L] = \sum _k\sum_j\int_{R_j}L_{kj}p(\textbf{x}, C_k)d\textbf{x}
    $$
    
    따라서 각 x에 대해 $\sum_kL_{kj}p(\textbf{x}, C_k)$ → $\sum_kL_{kj}p(C_k|\textbf{x})$를 최소화하는 것과 같다. 
    
    - Loss Matrix 예시 -> L_12 = 1000으로 설정함으로서 2종 오류를 최소화하는 것을 우선시하도록 함
    
    | Loss Matrix | cancer | normal |
    | --- | --- | --- |
    | cancer | 0 | 1000 |
    | normal | 1 | 0 |

### 3. The Reject Option

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/26af9838-41a7-400d-9c93-f993565da096/f0452289-c829-4600-ad2e-2c47935de036/Untitled.png)

- Region의 uncertainty가 높은 경우 데이터의 분류를 유보할 수 있다. threshold $\theta$에 대해 다음을 만족하면 유보한다.

$$
\max_k \{p(C_k|\textbf{x})\} \le \theta
$$

### 4. Inference and Decision

- 분류문제는 크게 두 단계로 나뉠 수 있다.
    - Inference : train data를 이용한 posterior 학습
    - Decision : posterior 이용해서 class 할당

#### Decision Problem(Classification)을 위한 방법

1. Generative Models
    - Inference Stage : Likelihood, Prior를 찾은 후 Bayes’ Thm. 이용해서 Posterior 구하기
    - Decision Stage : 새로운 input에 대한 class 할당
    - $p(\textbf{x}) = \sum_k p(\textbf{x}|c_k)p(C_k)$ 으로 구할 수 있으므로 확률이 낮은 x를 찾을 수 있음
        
        → outlier detection
        
    - 복잡도 높고 대량의 training data 필요
2. Discriminative Models
    - Inference Stage : posterior 학습
    - Decision Stage : 새로운 input에 대한 class 할당
    - 생성모델이 비해 복잡도 낮지만 이상치 탐지 어려움
3. Linear Discriminative Models
    - Inference, Decision의 구분이 없음 : input point를 class로 mapping하는 discriminant function을 바로 학습
    - decision boundary인 초록색 수직선을 찾음
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/26af9838-41a7-400d-9c93-f993565da096/eca44dac-911f-4959-9e5d-77e92d48ee19/Untitled.png)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/26af9838-41a7-400d-9c93-f993565da096/36d0a739-a689-42cc-bcbe-dfc5945269d7/Untitled.png)
    

### 5. Loss Functions for Regression

Decision Theory를 Regression에 적용해보자.

- Decision Stage: input x에 대하여 t에 대한 estimate y(x)를 선택하는 것: 2가지 방법
- Total Loss ↔ Average Loss를 줄이는 y(x)를 선택한다.
    
    $$
    E[L] = \int\int L(t, y(\textbf{x}))p(\textbf{x},t)d\textbf{x}dt\\E[L] = \int\int (y(\textbf{x})-t)^2p(\textbf{x},t)d\textbf{x}dt\\\dfrac{\delta E[L]}{\delta y(\textbf{x})}= 2\int \{y(x)-t\}p(\textbf{x},t)dt = 0\\\therefore y(\textbf{x})= \dfrac{\int tp(\textbf{x},t)dt}{p(\textbf{x})}=\int tp(t|\textbf{x})dt = E_t[t|\textbf{x}]
    $$
    
    → x에 대한 t의 값의 평균으로 예측을 하는 것이 좋다 (회귀의 결과와 동일)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/26af9838-41a7-400d-9c93-f993565da096/3681a4b0-8f86-4ac8-b06b-1bbffe0b5e76/Untitled.png)
    
    - 만약 t 또한 벡터로 나온다면 $y(\textbf{x})=E_t[\textbf{t}|\textbf{x}]$ 로 사용할 수 있다.

다른 방면으로 본다면

$$
\{y(\textbf{x})-t\}^2 = \{y(\textbf{x})-E_t[t|\textbf{x}]+E_t[t|\textbf{x}]-t\}^2 \\=\{y(\textbf{x})-E_t[t|\textbf{x}]\}^2+2\{y(\textbf{x})-E_t[t|\textbf{x}]\}\{E_t[t|\textbf{x}]-t\}+\{E_t[t|\textbf{x}]-t\}^2
$$

$$
\therefore E[L] = \int\int (y(\textbf{x})-t)^2p(\textbf{x},t)d\textbf{x}dt \\=\int \{y(\textbf{x})-E_t[t|\textbf{x}]\}^2p(\textbf{x})d\textbf{x}+\int \{E_t[t|\textbf{x}]-t\}^2p(\textbf{x})d\textbf{x}\\=\int \{y(\textbf{x})-E_t[t|\textbf{x}]\}^2p(\textbf{x})d\textbf{x}+\int var [t|\textbf{x}]p(\textbf{x})d\textbf{x}
$$

- 따라서 *E*[**L**]을 최소화하기 위한 y(x) = *E*[t | **x**]를 선택해야 하며 우측항은 지워지지 않기에 target data 내에 내재된 noise로, E[L]의 최솟값이 된다.
- (확장)Decision Problem(Regression)을 위한 방법

$$
E[L_q] = \int\int (y(\textbf{x})-t)^qp(\textbf{x},t)d\textbf{x}dt
$$

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/26af9838-41a7-400d-9c93-f993565da096/807c02fb-42c9-44b3-a057-3e5e03fd5c39/Untitled.png)


#### Decision Theory(Regression)을 위한 방법

1. Inference stage: *p*(**x**, t)를 구함 → 이후 *p*(t | **x**)를 찾아 *E*[t | **x**] 계산
2. Inference stage: *p*(t | **x**)를 구함 → 이후 *E*[t | **x**] 계산
3. Training Data로부터 y(x) function을 구함