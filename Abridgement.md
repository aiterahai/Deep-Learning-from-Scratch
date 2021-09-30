# Deep Learning from Scratch

## Perceptron

​	**퍼셉트론**은 **신경망**의 기원이 되는 알고리즘입니다. 

* **퍼셉트론이란?**

  퍼셉트론은 **다수**의 **신호**를 입력으로 받아 하나의 **신호**를 출력하는 것 입니다. 

  퍼셉트론 **신호**는 **흐름**을 만들고 정보를 앞으로 전달합니다. 

  퍼셉트론은 **1과 0** 즉 **신호가 흐른다/ 신호가 흐르지 않는다**로 **두 가지 값**을 가질 수 있습니다.

![01. 퍼셉트론 - Perceptron](https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99BDCE4D5B98A1022C)

​	위의 그림은 **입력**으로 2개의 신호를 받은 퍼셉트론의 예입니다.

​	x_1과 x_2는 **입력 신호**, y는 **출력 신호**, w_1과 w_2는 **가중치**를 뜻합니다.

​	그림의 원을 **뉴런** 혹은 **노드**라고 부릅니다.

​	입력 신호가 뉴런에 보내질 때는 각각 고유한 **가중치**가 곱해집니다. (x_1w_1, x_2w_2)

​	뉴런에서 보내온 신호의 총합이 **정해진 한계**를 넘어설 때만 1을 출력합니다.

​	그 한계를 **임계값**이라 하며 θ 기호로 나타냅니다.

​	이 것을 수식으로 나타내면 아래의 수식이 됩니다.
$$
y =
\begin{cases}
0\quad  (w_1x_1\ +\ w_2x_2 \leθ)\\
1\quad  (w_1x_1\ +\ w_2x_2 >θ)
\end{cases}
$$
​	퍼셉트론은 복수의 입력 신호 각각에 고유한 **가중치**를 부여합니다.

​	**가중치**는 각 신호가 결과에 주는 **영향력을 조절하는 요소**로 작용합니다.

​	**가중치**가 클수록 해당 신호가 그만큼 **더 중요함**을 뜻합니다.

* **논리 회로**

  * **AND 게이트**

    AND 게이트는 입력이 둘이고 출력은 하나입니다.

    아래 그림 처럼 입력 신호와 출력 신호의 대응 표를 **진리표**라고 합니다.

    이 그림은 AND 게이트의 진리표로, 두 입력이 모두 1일 때만 1을 출력하고, 그 외에는 0을 출력합니다.

    | x_1  | x_2  | y    |
    | ---- | ---- | ---- |
    | 0    | 0    | 0    |
    | 0    | 1    | 0    |
    | 1    | 0    | 0    |
    | 1    | 1    | 1    |

    이 AND 게이트를 퍼셉트론으로 표현을 할 것 입니다.

    우선 진리표대로 작동하도록 하는 **w_1, w_2, θ**의 값을 정해야 합니다.

    위의 진리표를 만족하는 **매개변수** 조합은 **무한**히 많습니다.

    (w_1, w_2, θ)가 (0.5, 0.5, 0.7) 일 때, 또 (0.5, 0.5, 0.8)이나 (1.0, 1.0, 1.0) 일때 모두 **AND 게이트의 조건을 만족합니다.**

    매개변수를 이렇게 설정하면 x_1과 x_2가 모두 1일 때만 가중 신호의 총합이 주어진 **임계 값**을 웃돌게 됩니다.

  * **NAND 게이트와 OR 게이트**

    NAND 는 Not AND를 의미합니다.

    그 동작은 AND 게이트의 출력을 **뒤집은 것**이 됩니다.

    | x_1  | x_2  | y    |
    | ---- | ---- | ---- |
    | 0    | 0    | 1    |
    | 0    | 1    | 1    |
    | 1    | 0    | 1    |
    | 1    | 1    | 0    |

    진리표로 나타내면 x_1와 x_2가 모두 1일 때만 0을 출력하고 그 이외의 경우에는 1을 출력합니다.

    NAND 게이트를 표현하려면 다음과 같은 조합이 있습니다.
    $$
    (w_1, w_2, θ) = (-0.5, -0.5, -0.7)
    $$
    AND 게이트를 구현하는 **매개변수의 부호를 모두 반전**하기만 하면 NAND의 게이트가 됩니다.
    
    OR 게이트는 입력 신호 중 하나 이상이 1이면 출력이 1이 되는 논리 회로입니다.
    
    | x_1  | x_2  | y    |
    | ---- | ---- | ---- |
    | 0    | 0    | 0    |
    | 0    | 1    | 1    |
    | 1    | 0    | 1    |
    | 1    | 1    | 1    |
    
    OR 게이트를 표현하라면 다음과 같은 조합이 있습니다.
    $$
    (w_1, w_2, θ) = (0.5, 0.5, 0.3)
    $$

  이것들 처럼 퍼셉트론으로 AND, NAND, OR 과 같은 논리 회로를 표현할 수 있습니다.

  여기서 중요한 점은 퍼셉트론의 구조는 AND, NAND, OR 게이트 모두에서 똑같다는 것 입니다.

  세 가지 게이트에서 다른 것은 매개변수의 값뿐입니다.

  즉 똑같은 구조의 퍼셉트론이 매개변수의 값만 적절히 조정하여 AND, NAND, OR로 변신 할 수 있습니다.

  

* **퍼셉트론 구현하기**

  * **AND 게이트**

    ![image-20210914210504078](Image\AND.png)

    다음은 x1과 x2를 매개변수로 받는 AND라는 함수입니다.

    매개변수 w1, w2, theta는 함수 안에서 초기화하고, 가중치를 곱한 입력의 총합이 임계값을 넘으면 1을 반환하고 그 외에는 0을

    반환합니다.

    위 그림 처럼 AND 가 잘 작동하는 것을 볼 수 있습니다.

  * **가중치와 편향**

    전에 봤던 아래의 식에서
    $$
    y =
    \begin{cases}
    0\quad  (w_1x_1\ +\ w_2x_2 \leθ)\\
    1\quad  (w_1x_1\ +\ w_2x_2 >θ)
    \end{cases}
    $$
    θ를 -b로 치환하면 퍼셉트론의 동작이 아래 식처럼 됩니다.
    $$
    y =
    \begin{cases}
    0\quad  (b\ +\ w_1x_1\ +\ w_2x_2 \le0)\\
    1\quad  (b\ +\ w_1x_1\ +\ w_2x_2 >0)
    \end{cases}
    $$
    

    두개의 식은 기호만 바꿧을 뿐, 의미는 같습니다.

    여기서 b를 **편향 Bias**이라고 하고 w_1, w_2는 그대로 가중치 입니다.

    ![image-20210914212906289](Image\AND_Bias.png)

    여기에서 -θ가 편향 b로 치환이 되었습니다.

    그리고 편향은 가중치와 기능이 다르다는 사실에 주의해야 합니다.

    가중치는 각 입력 신호가 결과에 주는 영향력(중요도)을 조절하는 매개변수이고, 편향은 뉴런이 얼마나 쉽게 활성화 하느냐를

    조정하는 매개변수입니다.

  * NAND

    ![image-20210914215034074](Image\NAND.png)

    NAND는 AND의 가중치와 편향의 부호를 바꾸어 주면 됩니다.

  * OR

    ![image-20210914215814518](Image\OR.png)

    OR 게이트는 AND 게이트의 가중치와 편향만 바꾸어주면 됩니다.

  이처럼 AND, NAND, OR의 게이트는 모두 가중치와 편향만 바꾸어주면 다른 게이트를 만들어 줄 수 있습니다.

* **퍼셉트론의 한계**

  지금까지 퍼셉트론을 이용해서 AND, NAND, OR의 3가지 논리 회로를 구현 해보았습니다.

  * **XOR 게이트**

    XOR 게이트는 **베타적 논리합**이라는 논리 회로 입니다. 

    x_1과 x_2중 한쪽이 1일 때만 1을 출력합니다.

    | x_1  | x_2  | y    |
    | ---- | ---- | ---- |
    | 0    | 0    | 0    |
    | 0    | 1    | 1    |
    | 1    | 0    | 1    |
    | 1    | 1    | 0    |

    지금 까지 배운 퍼셉트론으로는 XOR 게이트를 구현할 수 없습니다.

    ![img](https://t1.daumcdn.net/cfile/tistory/992152485B98A1C705)

    위 그림과 같이 퍼셉트론은 **직선**으로 나뉜 두 영역을 만듭니다.

  * **선형과 비선형**

    만약 퍼셉트론이 직선 즉 선형이라는 제약을 없엔 **비선형**구조 라면, 아래 그림과 같이 영역을 나눌 수 있습니다.

    ![img](https://t1.daumcdn.net/cfile/tistory/991D844D5B98A1DE08)

* **다층 퍼셉트론**

  * **게이트 조합**

    ![img](https://t1.daumcdn.net/cfile/tistory/995729435B98A1FE3D)

    NAND게이트와 OR게이트와 AND게이트로 XOR게이트를 만들 수 있습니다.

  * **XOR 게이트 구현**

    ![image-20210914221845851](Image\XOR.png)

    이 것을 퍼셉트론으로 나타내면 아래 그림 처럼 됩니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/754c7c20-3a21-11ea-8734-d1dac55eae87/2-13XOR%EC%9D%98-%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0.png)

    AND, OR이 단층 퍼셉트론 인데 반해 XOR은 2층 퍼셉트론입니다.

    따라서 이처럼 층이 여러 개인 퍼셉트론을 다층 퍼셉트론이라고 합니다.

* **정리**

  퍼셉트론은 입출력을 갖춘 알고리즘입니다. 입력을 주면 정해진 규칙에 따른 값을 출력합니다.

  퍼셉트론은 가중치와 편향을 매개변수로 설정합니다.

  퍼셉트론으로 AND, OR 게이트 등의 논리 회로를 표현할 수 있습니다.

  XOR 게이트는 단층 퍼셉트론으로는 표현할 수 없습니다.

  2층 퍼셉트론을 이용하면 XOR 게이트를 표현할 수 있습니다.

  단층 퍼셉트론은 직선형 영역만 표현할 수 있고, 다층 퍼셉트론은 비선형 영역도 표현할 수 있습니다.

  다층 퍼셉트론은 이론상 컴퓨터를 표현할 수 있습니다.
  
  

# 신경망

* **신경망이란?**

  ![img](https://media.vlpt.us/post-images/dscwinterstudy/82dcf020-38f2-11ea-98c4-63cdb84222e2/%EC%8B%A0%EA%B2%BD%EB%A7%9D.png)

  입력층이나 출력층과 달리 은닉층의 뉴런은 사람의 눈에 보이지 않습니다.

  0층을 입력층, 1층을 은닉층, 2층을 출력층이라고 합니다.

  위의 그림 속 신경망은 3층으로 구성되어있지만 가중치를 갖는 층이 2개이기 때문에 **2층 신경망**이라고 합니다.

  활성화 함수 Activation function :

  * 입력 신호의 총합을 출력 신호로 변환하는 함수입니다.
  * 변환된 신호를 다음 뉴런에 전달합니다.
  * 입력 신호의 총합이 활성화를 일으키는지를 정하는 역활을 합니다.

  ![img](https://media.vlpt.us/post-images/dscwinterstudy/d1593000-38e9-11ea-b942-cb9b82d31200/%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98%EC%9D%98-%EC%B2%98%EB%A6%AC-%EA%B3%BC%EC%A0%95.PNG)

  > a = b+w1x1+w2x2 
  > y=h(a) A를 함수 h()에 넣어서 y를 출력합니다.

  단순 퍼셉트론 : 단층 네트워크에서 계단 함수를 활성화 함수로 사용한 모델입니다.

  다층 퍼셉트론 : 신경망 즉 여러 층으로 구성되어있고, Sigmoid 함수 등의 매끈한 활성화 함수를 사용하는 네트워크 입니다.

* **활성화 함수의 종류**

  * **Sigmoid function**

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/a905ddd0-3829-11ea-8217-2f54dec14d43/%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C%ED%95%A8%EC%88%98.PNG)

    시그모이드 함수는 출력 값의 범위가 0~1 사이이며, 매우 매끄러운 곡선을 가집니다.

    분류는 0과 1로 나뉘며, 출력 값이 어느 값에 가까운지를 통해 어느 분류에 속하는지 쉽게 알 수 있습니다.

    

    ![image-20210915223034974](Image\Sigmoid.png)

  * **Step function**

    ![image-20210915223337368](Image\Step.png)

    계단 함수는 0과 1중 하나의 값만 돌려줍니다.

    하지만 시그모이드 함수는 0과 1사이의 연속적인 실수를 돌려줍니다.

    * **계단 함수와 시그모이드 함수의 공통점**

      입력이 중요하면 큰 값을 출력하고 입력이 중요하지 않으면 작은 값을 출력합니다.

      출력은 0과 1사이 입니다.

      비선형 함수입니다.

      > **선형함수의 문제점**
      >
      > * 층을 아무리 깊게 해도 은닉층이 없는 네트워크로도 똑같은 기능을 할 수 있습니다.
      >
      >   선형 함수로 레이어를 쌓는다면 여러 층으로 구성하는 이점을 살릴 수 없습니다.

    * **계단 함수와 시그모이드 함수의 차이점**

      계단 함수는 0과 1중 하나의 값만을 돌려주지만

      시그모이드는 연속적인 실수를 돌려주어서 확률, 통계적으로 사용할 수 있습니다.

  * **ReLU function**

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/c627e7a0-3829-11ea-8bae-8b3dd278412c/ReLU%ED%95%A8%EC%88%98.PNG)

    ReLU 함수는 입력이 0이 넘으면 그 입력을 그대로 출력하고 , 0 이하이면 0 을 출력하는 함수 입니다.

    ![image-20210915224017264](Image\ReLU.png)

    학습이 매우 빠르고, 연산 비용이 적고, 구현이 매우 간단하다는 장점이 있습니다.

    하지만 0보다 작은 값들에서 뉴런이 죽을 수 있다는 단점이 있습니다.

* **다차원 배열의 계산**

  * **행렬의 곱**

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/11890d50-38fc-11ea-8a14-7ddb1d5b4b35/%ED%96%89%EB%A0%AC%EA%B3%B1.png)

    ![image-20210915225732083](Image\np_2.png)

    np.dot() 행렬 곱을 계산합니다.

    np.dot(A, B)와 np.dot(B, A)는 다른 값이 될 수 있습니다.
    
    ![img](https://media.vlpt.us/post-images/dscwinterstudy/bd94fa40-38fd-11ea-8a14-7ddb1d5b4b35/%ED%96%89%EB%A0%AC%ED%98%95%EC%83%81%EA%B3%B1.png)
    
    행렬A의 1번째 차원의 원소 수 와 행렬B의 0번째 차원의 원소 수가 같아야합니다.
    
    ![image-20210916230725737](Image\np_3.png)
    
  * **신경망에서의 행렬의 곱**
  
    ![img](https://media.vlpt.us/post-images/dscwinterstudy/54edf130-38fe-11ea-8a14-7ddb1d5b4b35/%EC%8B%A0%EA%B2%BD%EB%A7%9D%ED%96%89%EB%A0%AC%EA%B3%B1.png)
  
    ![image-20210916230938515](Image\np_4.png)
  
* **3층 신경망 구현**

  * **표기법**

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/f88732c0-38fe-11ea-b428-5dc446614305/%EA%B0%80%EC%A4%91%EC%B9%98%ED%91%9C%EA%B8%B0.png)

  * **각 층의 신호 전달 구현**

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/4f6a4300-39fb-11ea-b6b9-5148bcc9f2c4/%EC%9E%85%EB%A0%A5%EC%B8%B5%EC%97%90%EC%84%9C1%EC%B8%B5%EB%B3%B4%EC%B6%A9.png)

    은닉층에서의 가중치의 합을 a로 표기하고 활성화 함수 h()로 변환된 신호를 z로 표현합니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/b18ace60-39fb-11ea-b6b9-5148bcc9f2c4/a1.png)

    1층의 가중치 부분을 행렬식으로 나타낸다면 아래 그림처럼 됩니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/70bc13d0-39fb-11ea-b6b9-5148bcc9f2c4/1%EC%B8%B5%EA%B0%80%EC%A4%91%EC%B9%98%ED%96%89%EB%A0%AC%EC%8B%9D.png)

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/7bede9c0-39fd-11ea-9694-9dbcffa449db/%EA%B0%80%EC%A4%91%EC%B9%98%ED%96%89%EB%A0%AC.PNG)

    ![image-20210916231337228](Image\NN_1.png)

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/f77f36c0-39fd-11ea-9304-8d735a4c8d2b/1%EC%B8%B5%EC%97%90%EC%84%9C-2%EC%B8%B5%EC%9C%BC%EB%A1%9C.png)

    ![image-20210916231432331](Image\NN_2.png)

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/e3844950-3a00-11ea-85c7-af1be234c277/%EC%B6%9C%EB%A0%A5%EC%B8%B5%EC%9C%BC%EB%A1%9C-%EC%8B%A0%ED%98%B8-%EC%A0%84%EB%8B%AC.png)

    ![image-20210916231530378](Image\NN_3.png)

    일반적으로 회귀에서는 출력층의 활성화 함수를 항등함수로, 바이너리 클래스 분류에서는 시그모이드 함수를, 다중 클래스 분류에서는 소프트맥스 함수를

    사용한다고 합니다.

    

  * **3층 신경망 구현**

    ![image-20210916231741342](Image\NN_4.png)

* **출력층 설계**

  * **항등 함수와 소프트맥스 함수**

    - 항등함수(identity function): 입력을 그대로 출력

      ![img](https://media.vlpt.us/post-images/dscwinterstudy/98bd1470-3a04-11ea-a976-bbc34e4880b0/%ED%95%AD%EB%93%B1%ED%95%A8%EC%88%98.png)

    - 소프트맥스 함수(softmax function):

      ![img](https://media.vlpt.us/post-images/dscwinterstudy/a70d9ad0-382a-11ea-83b1-3b44e26c4216/%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4.PNG)

      소프트맥스 함수의 분자는 입력 신호의 지수 함수, 분모는 모든 입력 신호의 지수 함수의 합으로 이루어져 있습니다.\

      ```python
      def softmax(a):
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        
        return y
      ```

      ![img](https://media.vlpt.us/post-images/dscwinterstudy/8c838090-3a04-11ea-a976-bbc34e4880b0/%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4-%ED%95%A8%EC%88%98.png)

      * **소프트맥스 함수 구현의 주의점**

        소프트맥스는 지수함수를 사용하기 때문에, Overflow의 문제가 발생해서 수치가 불안정해질 수 있다는 문제점이 있습니다.

        ![img](https://media.vlpt.us/post-images/dscwinterstudy/7fbb6850-3a04-11ea-a976-bbc34e4880b0/%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4%ED%95%A8%EC%88%98%EC%88%98%EC%A0%95.png)

        소프트맥스의 지수 함수를 계산할 때 어떤 정수를 더 하거나 빼도 결과는 바뀌지 않는다는 특성을 이용해 식을 수정합니다,

        오버플로우를 막기위해 C에 최댓값을 넣는게 일반적입니다.

        ```python
        def softmax(a):
          c = np.max(a)
          exp_a = np.exp(a-c)
          sum_exp_a = np.sum(exp_a)
          y = exp_a / sum_exp_a
          
          return y
        ```

      * **소프트맥스 함수의 특징**

        * 출력값은 0과 1사이의 실수입니다.

        * 출력의 총합이 1입니다. (확률로 해석할 수 있습니다.)

        * 지수함수가 단조 증가 함수 이기 때문에 소프트맥스 함수를 적용해도 각 원소의 대소 관계는 변하지 않습니다.

          따라서 신경망으로 분류할 때에 출력층의 소프트맥스 함수를 생략해도 됩니다.

      * **출력층의 뉴런 수**

        보통 분류 문제 에서는 분류하고 싶은 클래스 수로 뉴런 수를 설정하는 것이 일반적입니다.

* **MNIST 숫자 인식**

  * **신경망의 문제 해결 단계**
    1. 학습 : 학습 데이터를 사용해 가중치 매개변수를 학습합니다.
    2. 추론 : 학습한 매개변수를 사용하여 입력 데이터를 분류합니다.

  **신경망의 순전파(forward propagation)**: 이미 학습이 된 매개변수로 입력 데이터를 분류하는 추론 과정.

  * **MNIST 데이터셋**

    * 28*28 크기의 회색조 이미지입니다.

    * 각 픽셀은 0~255까지의 값을 가지고 취합니다.

    * 각 이미지에 실제 의미하는 숫자가 레이블로 붙어 있습니다.

    * load_mnist에는 3가지의 파라미터가 있습니다.

      - normalize :
        - 입력 이미지의 픽셀 값을 0 ~ 1로 정규화합니다.
        - False : 0 ~ 255 사이 값 유지합니다.
      - flatten :
        - 입력 이미지를 784개의 원소를 지닌 1차원 배열로 만듭니다.
        - False : 입력 이미지를 1 * 28 * 28 의 3차원 배열로 설정합니다.
      - one_hot_label
        - 데이터를 원-핫-인코딩 형태로 저장합니다. (정답을 뜻하는 원소만 1. 나머지는 모두 0)
        - False : ‘7’이나 ‘2’와 같이 숫자 형태의 레이블을 저장합니다.

      ![image-20210917213134486](C:\Users\terra\Desktop\Git\Deep-Learning-from-Scratch\Image\MNIST_1.png)

      ![image-20210917213112823](Image\MNIST_2.png)

      데이터 셋을 다운 받은 후 이미지를 출력을 하면 28*28 크기의 회색조 이미지가 출력이 됩니다.

  * **신경망의 추론 처리**

    ![image-20210917214522680](Image\MNIST_3.png)

    이 소스코드를 실행시켜 보면, 0.9352의 정확도를 가집니다.

  * **배치 처리**

    이미지 여러 장을 한꺼번에 입력하는 경우

    **배치(batch) : 하나로 묶은 입력 데이터**

    이미지 100개를 묶어 predict() 함수에 한번에 넘긴 경우 :

    * ![img](https://media.vlpt.us/post-images/dscwinterstudy/ee9dbaa0-3a15-11ea-a586-f780b0702393/%EB%B0%B0%EC%B9%98.png)

    x[0]와 y[0]에는 0번째 이미지와 그 추론 결과가, x[1]과 y[1]에는 1번째의 이미지와 그 결과가 저장됩니다.

    또한 배치로 한꺼번에 처리를 하면 효율성이 올라갑니다.

    * 이미지 한장당 처리 시간을 대폭 줄여줍니다.
    * 디스크 I/O가 줄면서 데이터가 병목되는 지점이 줄어듭니다.

    ```python
    x, t = get_data() 
    network = init_network()
    
    batch_size = 100
    accuracy_cnt = 0
    
    for i in range(0, len(x), batch_size):
    	x_batch = x[i:i+batch_size]
    	y_batch = predict(network, x_batch)
    	p = np.argmax(y_batch, axis=1)
    	accuracy_cnt += np.sum(p == t[t:t+batch_size])
    		
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
    ```



# 신경망 학습

* **데이터에서 학습한다 !**

  * **데이터 주도 학습**

    머신러닝이란, 데이터에서 답을 찾고 데이터에서 패턴을 발견하고 데이터로 이야기를 만드는 것입니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/ba47a280-4199-11ea-bed1-737062fffe57/image.png)

    만약, 5를 인식하고 싶다면 이미지에서 특징을 추출하고 그 특징의 패턴을 기계학습 기술로 학습하는 방법이 있습니다.

    여기서의 특징이란, 입력 데이터에서 본질적인 데이터를 정확하게 추출할 수 있도록 설계된 변환기를 가르킵니다.

    Computer Vision 분야에서는 SIFT, SURF HOG 등의 특징을 사용하고, 이들의 특징을 이용해서 이미지 데이터를 벡터로 변환한 분류기법인

    SVM, KNN 등으로 학습이 가능합니다.

    이 방식에서 더욱 발전하여 이후에는 완전히 데이터를 기계가 학습하는 방식의 딥러닝을 활용하였습니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/651fbfd0-419a-11ea-bfff-032d2a744144/image.png)

  * **훈련 데이터와 시험 데이터**

    머신러닝 실험 시 우선 훈련 데이터만 사용하여 학습하면서 최적의 매개변수를 찾습니다.

    그 이유는 범용 능력, 즉 아직 보지 못한 데이터로도 문제를 올바르게 풀어내지 못한 능력을 제대로 평가하기 위함입니다.

    

    오버피팅(Overfitting) : 모델이 실제 분포보다 학습 샘플들 분포에 더 근접하게 학습되는 현상

* **손실 함수**

  지금 얼마나 행복한지에 대한 답은 아주 행복하다 혹은 그리 행복한 거 같지 않다라고 막연한 답이 돌어오는 것이 보통입니다.

  그러나 누군가 수치로 10.23 만큼 행복하다라고 답하면 질문자는 당황할 것입니다.

  이 사람은 자신의 행복을 행복 지표를 이용해 측정합니다.

  

  이와 같이 신경망 학습에도 하나의 지표가 있습니다.

  그 지표를 가장 좋게 만들어주는 가중치 매개변수의 값을 탐색하는 것이 목적입니다.

  신경망 학습에서는 손실 함수가 바로 그 지표입니다.

  

  손실 함수 : 신경망 학습에서 사용하는 지표 (보통 평균 제곱 오차와 교차 엔트로피 오차를 사용합니다)

  * **평균 제곱 오차**

    회귀문제에서 주로 사용되는 평가 지표입니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/0c2c2490-419a-11ea-9479-ad3ff669a3cd/image.png)

    오차가 더 작은 경우 정답에 더욱 가깝다는 것을 알 수 있습니다.

    ```python
    def sum_squares_error(y, t):
    	return 0.5 * np.sum((y-t)**2)
    ```

    

  * **교차 엔트로피 오차**

    분류문제에서 주로 사용되는 평가 지표입니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/b176eb30-42a1-11ea-a67c-c756fb257db9/e-4.2.png)

    여기서 log는 밑이 e인 자연로그 입니다. 

    y_k는 신경망의 출력, t_k는 정답 레이블입니다.

    신경망 출력이 0.6이라면 교차 엔트로피 오차는 -log0.6으로 결과는 0.51이 됩니다.

    즉 교차 엔트로피 오차는 정답일 때의 출력이 전체 값을 정하게 됩니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/771df610-419b-11ea-a3b8-fd5b14e3378a/image.png)

    ```python
    def cross_entropy_error(y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))
    ```

    

  * **미니배치 학습**

    MNIST 데이터셋은 훈련 데이터가 60,000개 이므로 신경망 학습에서 훈련 데이터로부터 일부만 골라 학습을 수행합니다.

    그 일부가 바로 미니배치이며 가령 60,000장의 훈련 데이터 중에서 100장을 무작위로 뽑은 학습 방법이 미니배치 학습입니다.

    

  * **손실 함수를 설정하는 이유**

    손실 함수를 사용하는 이유는 정확도를 끌어내는 매개변수 값을 찾는 것이 우리의 목표이기 때문입니다.

    정확도라는 지표를 두고 손실 함수의 값을 사용하는 이유는 정확도를 지표로 삼으면 미분 값이 대부분의 장소에서 0이 되어 매개변수를 갱신할 수 없기

    때문입니다.

  

* **수치 미분**

  * **미분**

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/44ad4090-419c-11ea-bed1-737062fffe57/image.png)

    ```python
    def numerical_diff(f, x):
        h = 1e-4
        return (f(x+h) - f(x-h)) / (2*h)
    ```

  * **수치 미분의 예**

    ```python
    def function(x):
    
        return 0.01*x**2 + 0.1*x 
    ```

    계산된 미분 값은 x에 대한 f(x)의 변화량, 즉 함수의 기울기에 해당합니다.

    그리고 x가 5일때와 10일 때의 진정한 미분은 0.2와 0.3입니다.

  * **편미분**

    편미분 : 변수가 여러개인 함수에 대한 미분

    앞에와 달리 변수가 2개라는 점에 주의를 해야 합니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/6c2996f0-419c-11ea-a3b8-fd5b14e3378a/image.png)

    ```python
    def function(x):
    	return x[0]**2 + x[1]**2
    ```

    인수 x는 넘파이 배열이라고 가정하며 넘파이 배열의 각 원소를 제곱하고 그 합을 간단한 형태로 구현할 수 있습니다.

  

* **기울기**

  ```python
  def numerical_gradient(f, x):
      h = 1e-4
      grad = np.zeros_like(x)
      
      for idx in range(x.size):
          tmp_val = x[idx]
          x[idx] = tmp_val+h
          fxh1 = f(x)
          x[idx] = tmp_val - h
          fxh2 = f(x)
          
          grad[idx] = (fxh1 - fxh2) / (2*h)
          x[idx] = tmp_val
          
      return grad
  ```

  이 함수는 복잡해 보이지만, 동작 방식은 변수가 하나일때의 수치 미분과 거의 동일합니다.

  f는 함수, x는 넘파이 배열이므로 넘파이 배열 x의 각 원소에 대하서 수치 미분을 구합니다.

  

  기울기는 각 지점에서 낮아지는 방향을 가르킵니다.

  즉, 기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향입니다.

  * **경사 하강법**

    신경망에서 최적의 매개변수 중 최적이란 손실 함수가 최솟값이 될 때의 매개변수 값 입니다.

    하지만 일반적인 손실함수는 복잡하고 최솟값이 되는 곳을 찾기 어렵습니다.

    이런 상황에서 기울기를 잘 이용해 함수의 최솟값(또는 가능한 작은 값)을 찾으려는 것이 경사 하강법입니다.

    

    각 지점에서 함수의 값을 낮추려고 제시하는 지표가 기울기입니다.

    경사법을 수식으로 나타낼 때 갱신하는 양을 학습률(Learning rate)이라고 표현합니다.

    즉, 한 번의 학습으로 얼마만큼 학습해야 할지를 정하는 것이 학습률입니다.

    ```python
    def gradient_descent(f, init_x, lr=0.01, step_num=100):
        x = init_x
        x_history = []
        for i in range(step_num):
            x_history.append( x.copy() )
            grad = numerical_gradient(f, x)
            x -= lr * grad
        return x, np.array(x_history)
    ```

    학습률이 너무 크면 발산하고, 너무 작으면 수렴하는데 많은 학습을 해야 합니다.

    따라서 적당한 학습률을 정해줘야 합니다.

  * **신경망에서의 기울기**

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/8865d4a0-419c-11ea-af2e-4fe713384e5c/image.png)

    신경망 학습에서의 기울기는 가중치 매개변수에 대한 손실 함수의 기울기입니다.

    가중치 W, 손실 함수가 L인 신경망의 경우 편미분을 합니다.

    그리고 손실 함수 L이 얼마나 변하는지에 대해서 알려주는 것이 w_11입니다.

    ![image-20210917232353772](Image\GD_1.png)

    기울기를 구하는 코드 :

    ```python
    class simpleNet:
        def __init__(self):
            self.W = np.random.randn(2,3)
        def predict(self, x):
            return np.dot(x, self.W)
        def loss(self, x, t):
            z = self.predict(x)
            y = softmax(z)
            loss = cross_entropy_error(y, t)
    return loss
    ```
  
  
  
* **학습 알고리즘 구현**

  신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 학습이라고 합니다.

  1. 미니배치 : 훈련 데이터 중 일부를 무작위로 가져옵니다. 미니배치의 손실 함수 값을 줄이는 것이 목표입니다.

  2. 기울기 산출 : 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구합니다. 기울기는 손실 함수의 값을

     가장 적게 하는 방향을 제시합니다.

  3. 매개변수 갱신 : 가중치 매개변수를 기울기 방향으로 갱신합니다.

  4. 1~3단계를 반복합니다.

  하강법으로 매개변수를 갱신하는 방법이며 이때 데이터를 미니배치로 무작위로 선정하기 때문에

  확률적 경사 하강법 stochastic gradient descent, SGD라고 부릅니다.

  * **2층 신경망 클래스 구현**

    ```python
    import sys
    import os
    import numpy as np
    sys.path.append(os.pardir)
    from common.functions import sigmoid, softmax, cross_entropy_error
    from common.gradient import numerical_gradient
    
    
    class TwoLayerNet:
        def __init__(self, input_size, hidden_size, output_size,
                     weight_init_std=0.01):
            self.params = {}
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
                np.random.randn(input_size, hidden_size)
            self.params['b1'] = np.zeros(hidden_size)
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
                np.random.randn(hidden_size, output_size)
            self.params['b2'] = np.zeros(output_size)
    
        def predict(self, x):
            W1, W2 = self.params['W1'], self.params['W2']
            b1, b2 = self.params['b1'], self.params['b2']
    
            a1 = np.dot(x, W1) + b1
            z1 = sigmoid(a1)
            a2 = np.dot(z1, W2) + b2
            y = softmax(a2)
    
            return y
    
        def loss(self, x, t):
            y = self.predict(x)
    
            return cross_entropy_error(y, t)
    
        def accuracy(self, x, t):
            y = self.predict(x)
            y = np.argmax(y, axis=1)
            t = np.argmax(t, axis=1)
    
            accuracy = np.sum(y == t) / float(x.shape[0])
            return accuracy
    
        def numerical_gradient(self, x, t):
            loss_W = lambda W: self.loss(x, t)
    
            grads = {}
            grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
            grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
            grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
            grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
    
            return grads
    
    
    if __name__ == '__main__':
        net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
        print(net.params['W1'].shape)
        print(net.params['b1'].shape)
        print(net.params['W2'].shape)
        print(net.params['b2'].shape)
    
        x = np.random.rand(100, 784)
        t = np.random.rand(100, 10)
    
        grads = net.numerical_gradient(x, t)
        print(grads['W1'].shape)
        print(grads['b1'].shape)
        print(grads['W2'].shape)
        print(grads['b2'].shape)
    ```

  * **미니배치 학습 구현**

    미니배치 학습이란 훈련 데이터 중 일부를 무작위로 꺼내고, 그 미니배치에 대해서 경사하강법으로 매개변수를 갱신하는 것 

    입니다. TwoLayerNet으로 학습을 수행합니다.

    ```python
    import sys, os
    sys.path.append(os.pardir)
    import numpy as np
    import matplotlib.pyplot as plt
    from dataset.mnist import load_mnist
    from two_layer_net import TwoLayerNet
    
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        grad = network.gradient(x_batch, t_batch)
        
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
    ```

    위 코드를 통하여 신경망의 가중치 매개변수가 학습 횟수가 늘어나면서 손실 함수의 값이 줄어드는 것을 확인할 수 있습니다.

    즉, 신경망의 가중치 매개변수가 서서히 데이터에 적응하고 있음을 의미합니다.

    신경망이 학습을 하고있다는 것을 알 수 있습니다.

    

    데이터를 반복적으로 학습을 하여 최적의 가중치 매개변수로 서서히 다가가고 있음을 알 수 있습니다.

  * **시험 데이터로 평가하기**

    신경망 학습의 목표는 범용적인 능력을 익히는 것 입니다.

    오버피팅을 일으키지 않는지 확인해야합니다.

    아래 코드는 평가를 하기 위한 코드 입니다.

    ```python
    import sys, os
    sys.path.append(os.pardir)
    import numpy as np
    import matplotlib.pyplot as plt
    from dataset.mnist import load_mnist
    from two_layer_net import TwoLayerNet
    
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        grad = network.gradient(x_batch, t_batch)
        
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
    ```



## 오차역전파

신경망의 가중치 매개변수의 기울기는 수치 미분을 이용해 구했습니다.

수치 미분은 단순하고 구현하기도 쉽지만 계산 시간이 오래 걸린다는 것이 단점입니다.

오차역전파법은 가중치 매개변수의 기울기를 효율적으로 계산할 수 있습니다.



오차역전파법을 이해하는 방법은 두 가지가 있습니다.

수식을 통한 방법 또는 계산 그래프를 이용한 방법입니다.

수식을 통한 이해는 본질을 놓치거나, 수많은 수식에 당황하는 일이 벌어질 수도 있어 계산그래프로 '시각적'으로 설명합니다.

* **계산 그래프**

  계산그래프는 계산 과정을 그래프로 나타낸 것입니다.

  여기에서의 그래프는 우리가 아는그래프 자료구조로, 복수의 노드와 엣지로 표현됩니다.

  * **계산 그래프로 풀다**

    간단한 문제를 계산 그래프로 풀어봅시다.

    > 문제 1 : 현빈 군은 슈퍼에서 1개에 100원인 사과를 2개 샀습니다. 이때 지불 금액을 구하세요. 단, 소비세가 10% 부가됩니다

    계산 그래프는 계산 과정을 노드와 화살표로 표현합니다.

    노드는 원으로 표기하고 원 안에 연산 내용을 적습니다.

    또, 계산 결과를 화살표 위에 적어 각 노드의 계산 결과가 왼쪽에서 오른쪽으로 전해지게 합니다.

    문제 1을 계산 그래프로 풀면 다음 그림처럼 됩니다.

    ![image-20210923200756160](\Image\Computational_Graphs_1.png)

    그림과 같이 처음에 사과의 100원이 'x2' 노드로 흐르고, 200원이 되어 다음 노드로 전달됩니다.

    이제 200원이 'x1.1' 노드를 거쳐 220원이 됩니다. 따라서 이 계산 그래프에 따르면 최종 답은 220원이 됩니다.

    

    또한, 그림에서는 'x2'와 'x1.1'을 각각 하나의 연산으로 취급해 원 안에 표기했지만, 곱셈만을 연산으로 생각할 수도 있습니다.

    이렇게 하면 아래 그림 처럼 '2'와 '1.1'은 각각 사과의 개수와 소비세 변수가 되어 원 밖에 표기하게 됩니다.

    ![image-20210923201441822](\Image\Computational_Graphs_2.png)

    그럼 다음 문제입니다.

    > 문제 2 : 현빈 군은 슈퍼에서 사과를 2개, 귤을 3개 샀습니다. 사과는 1개에 100원, 귤은 1개 150원입니다. 소비세가 10%일 떄 지불 금액을 구하세요.

    ![image-20210923202146432](\Image\Computational_Graphs_3.png)

    이 문제에는 덧셈 노드인 +가 새로 등장하여 사과와 귤의 금액을 합산합니다.

    계산 그래프는 왼쪽에서 오른쪽으로 계산을 진행합니다.

    회로에 전류가 흐르듯 계산 결과가 왼쪽에서 오른쪽 끝으로 전달된다고 생각하시면 됩니다.

    계산 결과가 오른쪽 끝에 도착하면 거기서 끝납니다. 그래서 위 그림에서의 답은 715원입니다.

    

    지금까지 살표본 것처럼 계산 그래프를 이용한 문제풀이는 다음 흐름으로 진행합니다.

    1. 계산 그래프를 구성한다.
    2. 그래프에서 계산을 왼쪽에서 오른쪽으로 진행한다.

    여기서 2번째 "계산을 왼쪽에서 오른쪽으로 진행"하는 단계를 순전파라고 합니다.

    순전파는 계산 그래프의 출발점부터 종착점으로의 전파입니다.

    순전파라는 이룸이 있다면 반대 방향인 전파인 역전파도 있습니다.

    역전파는 미분을 계산할때 중요한 역활을 합니다.

  * **국소적 계산**

    계산 그래프의 특징은 "국소적 계산"을 전파함으로써 최종 결과를 얻는다는 점에 있습니다.

    국소적이란 "자신과 직접 관계된 작은 범위"라는 뜻입니다.

    국소적 계산은 결국 전체에서 어떤 일이 벌어지든 상관없이 자신과 관계된 정보만으로 결과를 출력할 수 있다는 것입니다.

    예를 들어 슈퍼마켓에서 사과 2개를 포함한 여러 식품을 구입하는 경우를 생각해봅시다. 이를 아래 그림과 같은 계산그래프로

    나타낼 수 있을 겁니다.

    ![image-20210923203048637](\Image\Computational_Graphs_4.png)

    그림에서는 여러 식품을 구입하여 총 금액이 4,000원이 되었습니다.
    
    여기에서 핵심은 각 노드에서의 계산은 국소적 계산이라는 점입니다.
    
    가령 사과와 그 외의 물품 값을 더하는 계산(4,000 + 200 -> 4,200)은 4,000이라는 숫자가 어떻게 계산되었느냐 와는 상관없이
    
    단지 두 숫자를 더하면 된다는 뜻입니다.
    
    각 노드는 자신과 관련한 계산 외에는 아무것도 신경 쓸 게 없습니다.
    
    
    
    이처럼 계산 그래프는 국소적 계산에 집중합니다. 전체 계산이 제아무리 복잡하더라도 각 단계에서 하는 일은 해당 노드의 
    
    "국소적 계산"입니다. 국소적인 계산은 단순하지만, 그 결과를 전달함으로써 전체를 구성하는 복잡한 계산을 해낼 수 있습니다.
    
  * **왜 계산 그래프로 푸는가?**
  
    지금까지 계산 그래프를 써서 두 문제를 풀어봤습니다.
  
    계산 그래프의 이점중 하나는 국소적 계산입니다.
  
    전체가 아무리 복잡해도 각 노드에서는 단순한 계산에 집중하여 문제를 단순화할 수 있습니다.
  
    또 다른 이점으로, 계산 그래프는 중간 계산 결과를 모두 보관할 수 있습니다.
  
    예를 들어 사과 2개까지 계산했을 때의 금액은 200원, 소비세를 더하기 전의 금액은 650원인 식이죠.
  
    가장 큰 이점은 역전파를 통해 '미분'을 효율적으로 계산할 수 점에 있습니다.
    
    
    
    계산 그래프의 역전파를 설명하기 위해 문제 1을 봅시다.
    
    문제 1은 사과를 2개 사서 소비세를 포함한 최종 금액을 구하는 것이었죠.
    
    여기서 가령 사과 가격이 오르면 최종 금액에 어떤 영향을 끼치는지를 알고싶습니다.
    
    이는 "사과 가격에 대한 지불 금액의 미분"을 구하는 문제에 해당합니다.
    
    
    
    계산 그래프의 역전파를 설명하기 위해 문제 1을 다시 꺼내보겠습니다.
    
    문제 1은 사과 2개를 사서 소비세를 포함한 최종 금액은 구하는 것이였습니다.
    
    여기서 가령 사과 가격이 오르면 최종 금액에 어떤 영향을 끼치는지를 알고 싶습니다.
    
    이는 "사과 가격에 대한 지불 금액의 미분"을 구하는 문제에 해당합니다.
    
    사과 값을 x, 지불 금액을 L이라고 했을 때
    $$
    {\delta L\over \delta x}
    $$
    을 구하는 것입니다.
    
    이 미분 값은 사과 값이 '아주 조금' 올랐을 때 지불 금액이 얼마나 증가하느냐를 표시한 것입니다.
    
    앞에서 말했듯이 '사과 가겨에 대한 지불 금액의 미분' 같은 값은 계산 그래프에서 역전파를 하면 구할 수 있습니다.
    
    먼저 결과만을 나타내면 다음 그림처럼 계산 그래프 상의 역전파에 의해서 미분을 구할 수 있습니다.
    
    ![image-20210924191239449](\Image\Computational_Graphs_5.png)
    
    위 그림과 같이 역전파는 순전파와는 반대 방향의 화살표로 그립니다.
    
    이 전파는 '국소적 미분'을 전달하고 그 미분 값은 화살표의 아래와 적습니다.
    
    이 예에서 역전파는 오른쪽에서 왼쪽으로 '1 -> 1.1 -> 2.2' 순으로 미분 값을 전달합니다.
    
    이 결과로부터 '사과 가겨에 대한 지불 금액의 미분'은 2.2라고 할 수 있습니다.
    
    사과가 1원이 오르면 최종 금액은 2.2원 오른다는 뜻입니다.
    
    
    
    위 예에서는 사과 가격에 대한 미분만 구했지만, 소비세에 대한 지불 금액의 미분이나 사과 개수에 대한 미분도 같은 순서로
    
    구할 수 있습니다.
    
    그리고 그때는 중간까지 구한 미분 결과를 공유할 수 있어서 다수의 미분을 효율적으로 계산할 수 있습니다.
    
    이처럼 계산 그래프의 이점은 순전파와 역전파를 활용해서 각 변수의 미분을 효율적으로 구할 수 있다는 것입니다.
  
* **연쇄법칙**

  계산 그래프의 순전파는 계산 결과를 왼쪽에서 오른쪽으로 전달했습니다.

  하지만 역전파는 국소적인 미분을 순서방향과는 반대인 오른쪽에서 왼쪽으로 전달합니다.

  또한, 이 국소적 미분을 전달하는 원리는 연쇄법칙에 따른 것입니다.

  * **계산 그래프의 역전파**

    계산 그래프를 사용한 역전파의 예를 하나 살펴보겠습니다.

    y = f(x)라는 계산의 역전파를 아래 그림으로 그려봅시다.

    ![image-20210924192138059](\Image\Chain_Rule_1.png)

    그림과 같이 역전파의 계산 절차는 신호 E에 노드의 국소적 미분을 곱한 후 다음 노드로 전달하는 것입니다.

    여기에서 말하는 국소적 미분은 순전파 때의 y=f(x) 계산의 미분을 구한다는 것이며, 이는 x에 대한 y의 미분을 구한다는 뜻입니다.

    가령 y = f(x) = x^2이라면 δy/δx = 2x 가 됩니다.

    그리고 이 국소적인 미분을 상류에서 전달된 값에 곱해 앞쪽 노드로 전달하는 것입니다.

  * **연쇄법칙이란?**

    연쇄법칙을 설명하려면 합성 함수를 알아야 합니다.

    합성 함수란 여러 함수로 구성된 함수입니다.

    예를 들어 z = (x + y)^2라는 식은 아래 식처럼 두개의 식으로 구성됩니다.
    $$
    z = t^2\\
    t = x + y
    $$
    연쇄법칙은 합성 함수의 미분에 대한 성질이며, 다음과 같이 정의됩니다.

    > 합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있습니다.

    이것이 연쇄법칙의 원리입니다.

    위의 그림으로 설명하자면, δz/δx은 δz/δt과 δt/δx의 곱으로 나타낼 수 있다는 것입니다.

    수식으로는 아래의 수식처럼 쓸 수 있습니다.
    $$
    {δz\overδx} = {δz\overδt}{δt\overδx}
    $$
    위 식은 δt를 서로 지울 수 있습니다.

    연쇄법칙을 써서 위 식의 미분 δz/δx를 구합시다.

    가장 먼저 위 식{z = t^2, t = x + y}의 국소적 미분을 구합니다.
    $$
    {δz\overδt} = 2t\\
    {δt\overδx} = 1
    $$
    위 식과 같이 δz/δt는 2t이고, δt/δx는 1입니다.

    이는 미분 공식에서 해석적으로 구한 결과입니다.

    그리고 최종적으로 구하고 싶은 δz/δx는 위 식에서 구한 두 미분을 곱해 계산합니다.
    $$
    {δz\overδx} = {δz\overδt}{δt\overδx} = 2t * 1=2(x+y)
    $$

  * **연쇄법칙과 계산 그래프**

    그럼 위 식처럼 연쇄법칙 계산을 계산 그래프로 나타내보면 아래 그림처럼 그릴 수 있습니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/e47fd900-41a3-11ea-b40d-6705eaadcebd/fig-5-7.png)

    그림과 같이 계산 그래프의 역전파는 오른쪽에서 왼쪽으로 신호를 전파합니다.

    역전파의 계산 절차에서는 노드로 들어온 입력 신호에 그 노드의 국소적 미분을 곱한 후 다음 노드로 전달합니다.

    예를 들어 **2 노드에서의 역전파를 보자면 입력은 δz/δt이며, 이에 국소적 미분인 δz/δt를 곱하고 다음 노드로 넘깁니다.

    

    이 계산은 연쇄법칙에 따르면
    $$
    {δz\overδz}{δz\overδt}{δt\overδx}={δz\overδt}{δt\overδx} = {δz\overδx}
    $$
    수식이 성립하여 x에 대한 z의 미분이 됩니다.

    즉, 역전파가 하는 일은 연쇄법칙의 원리와 같다는 것입니다.

    위 그림에 수식의 결과를 대입하면 아래 그림이 되며, δz/δx는 2(x + y)임을 구할 수 있습니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/e10223f0-41a3-11ea-9e70-43b4cf1f0bf4/fig-5-8.png)

* **역전파**

  * **덧셈 노드의 역전파**

    먼저 덧셈 노드의 역전파입니다.

    z = x + y라는 식을 대상으로 역전파를 살펴보겠습니다.

    z = x + y의 미분은 다음과 같이 해석적으로 계산할 수 있습니다.
    $$
    {δz\overδx}=1\\
    {δz\overδy}=1
    $$
    위 식에서 같이 δz/δx와 δz/δy는 모두 1이 됩니다.

    이를 계산 그래프로는 아래 그림처럼 그릴 수 있습니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/55494a90-41a4-11ea-b40d-6705eaadcebd/fig-5-9.png)

    그림과 같이 역전파 때는 상류에서 전해진 미분에 1을 곱하여 하류로 흘립니다.

    즉, 덧셈 노드의 역전파는 1을 곱하기만 할 뿐이므로 입력된 값을 그대로 다음 노드로 보내게 됩니다.

    

    이 예에서는 상류에서 전해진 미분값을 δL/δz라고 했는데, 이는 아래 그림과 같이 최종적으로 L이라는 값을 출력하는

    큰 계산 그래프를 가정하기 때문입니다.

    z = x + y 계산은 그 큰 계산 그래프의 중간 어딘가에 존재하고, 상류로부터 δL/δz 값이 전해진 것입니다.

    그리고 다시 하류로는 δL/δx과 δL/δy값을 전달합니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/aa4c23a0-41a4-11ea-bed1-737062fffe57/fig-5-10.png)

    가령 10 + 5 = 15라는 계산이 있고, 상류에서 1.3이라는 값이 흘러옵니다.

    이를 계산 그래프로 그리면 아래 그림처럼 됩니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/ad665bf0-41a4-11ea-b40d-6705eaadcebd/fig-5-11.png)

    덧셈 노드 역전파는 입력 신호를 다음 노드로 출력할 뿐이므로 그림처럼 1.3을 그대로 다음 노드로 전달합니다.

  * **곱셈 노드의 역전파**

    z = xy라는 식을 생각해봅시다.

    이식의 미분은 다음과 같습니다.
    $$
    {δz\overδx}=y\\
    {δz\overδy} = x
    $$
    이 식에서 계산 그래프는 다음과 같이 그릴 수 있습니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/1be4a500-41a5-11ea-8248-4760a63b1878/fig-5-12.png)

    곱셈 노드 역전파는 상류의 값에 순전파 떄의 입력 신호들을 서로 바꾼 값을 곱해서 하류로 보냅니다.

    서로 바꾼 값이란 위 그림 처럼 순전파 때 x였다면 역전파에서는 y, 순전파 때 y 였다면 역전파에서는 x로 바꾼다는 의미입니다.

    

    가령 10 * 5 = 50 이라는 계산이 있고, 역전파 떄 상류에서 1.3 값이 흘러나온다고 하면, 이를 계산 그래프로 그리면 아래 그림처럼 됩니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/6d1da4d0-41a5-11ea-8248-4760a63b1878/fig-5-13.png)

    곱셈의 역전파에서는 입력 신호를 바꾼 값을 곱하여 하나는 1.3 * 5 = 6.5, 다른 하나는 1.3 * 10 = 13이 됩니다.

    덧셈의 역전파에서는 상류의 값을 그대로 흘려보내서 순방향 입력 신호의 값은 필요하지 않았지만, 곱셈의 역전파는 순방향 입력

    신호의 값이 필요합니다.

    그래서 곱셈 노드를 구현할 때는 순전파의 입력 신호를 변수에 저장해둡니다.

    

  * **사과 쇼핑의 예**

    이번 장을 시작할 때 본 사과 쇼핑 예를 다시 살펴보겠습니다.

    이 문제에서는 사과의 가격, 사과의 개수, 소비세라는 세 변수 각각이 최종 금액에 어떻게 영향을 주느냐를 풀고자 합니다.

    이는 사과 가격에 대한 지불 금액의 미분, 사과 개수에 대한 지불 금액의 미분, 소비세에 대한 지불 금액의 미분을 구하는 것에

    해당합니다.

    이를 계산 그래프의 역전파를 사용해서 풀면 아래 그림 처럼 됩니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/dbf2e780-41a5-11ea-8248-4760a63b1878/fig-5-14.png)

    지금까지 설명한 바와 같이 곱셈 노드의 역전파에서는 입력 신호를 서로 바꿔서 하류로 흘립니다.

    결과를 보면 사과 가격의 미분은 2.2, 사과 개수의 미분은 110, 소비세의 미분은 200입니다.

    이것은 소비세와 사과 가격이 같은 양만큼 오르면 최종 금액에는 소비세가 200의 크기로, 사과 가격이 2,2크기로 영향을 준다고

    해석할 수 있습니다.
  
* **활성화 함수 계층 구현하기**

  * **ReLU 계층**

    활성화 함수로 사용되는 ReLU의 수식은 다음과 같습니다.
    $$
    y =
    \begin{cases}
    x\;\;\;(x>0)\\
    0\;\;\;(x\le0)
    \end{cases}
    $$
    위 식에서 x에 대한 y의 미분은 아래 처럼 구합니다.
    $$
    {\delta y\over \delta x} = 
    \begin{cases}
    1\;\;\;(x>0)\\
    0\;\;\;(x\le0)
    \end{cases}
    $$
    위에서와 같이 순전파 때의 입력이 x가 0보다 크면 역전파는 상류의 값을 그대로 하류로 흘립니다.

    반면, 순전파 때 x가 0 이하면 역전파 때는 하류로 신호를 보내지 않습니다.

    

    파이썬 소스는 아래와 같이 짤 수 있습니다.

    ```python
    class Relu:
        def __init__(self):
            self.mask = None
    
        def forward(self, x):
            self.mask = (x <= 0)
            out = x.copy()
            out[self.mask] = 0
    
            return out
    
        def backward(self, dout):
            dout[self.mask] = 0
            dx = dout
    
            return dx
    ```

    Relu 클래스는 mask라는 인스턴스 변수를 가집니다.

    mask는 True/False로 구성된 넘파이 배열로, 순전파의 입력인 x의 원소 값이 0 이하인 인덱스는 True, 그 외는 False로

    유지합니다. mask 변수는 True False로 구성된 넘파이 배열을 유지합니다.

  * **Sigmoid 계층**

    시그모이드 함수는 다음 식을 의미하는 함수입니다.
    $$
    y = {1\over1 + exp(-x)}
    $$
    이를 계산 그래프로 그리면 아래 그림처럼 됩니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/12ace2a0-41a9-11ea-8248-4760a63b1878/fig-5-19.png)

    위 그림에는 x와 +노드 말고도 exp와 /노드가 새롭게 등장합니다.

    exp노드는 y = exp(x)계산을 구행하고 / 노드는 y = 1/x를 수행합니다.

    그림과 같이 식의 계산은 국소적 계산의 전파로 이뤄집니다.

    

    **1 단계**

    ​	'/' 노드, 즉 y = 1/x 을 미분하면 다음 식이 됩니다.
    $$
    {\delta y\over \delta x} = -{1\over x^2}\\
    = -y^2
    $$
    ​	위 식에 따르면 역전파 때는 상류에서 흘러온 값에 -y^2를 곱해서 하류로 전달합니다.

    ​	계산 그래프에서는 다음과 같습니다.

    ​	![img](https://media.vlpt.us/post-images/dscwinterstudy/ab2468f0-41a9-11ea-886e-51c0587d4327/fig-5-191.png)

    **2 단계**

    ​	'+' 노드는 상류의 값을 여과 없이 하류로 내보내는 게 다입니다.

    ​	계산 그래프에서는 다음과 같습니다.

    ​	![img](https://media.vlpt.us/post-images/dscwinterstudy/c1235c60-41a9-11ea-bc96-9d2b4aed3aec/fig-5-192.png)

    **3 단계**

    ​	'exp' 노드는 y = exp(x) 연산을 수행하며, 그 미분은 다음과 같습니다.

    ​	
    $$
    {\delta y\over \delta x} = exp(x)
    $$
    ​	계산 그래프에서는 상류의 값에 순전파 때의 출력을 곱해 하류로 전파합니다.

    ​	![img](https://media.vlpt.us/post-images/dscwinterstudy/dc82cea0-41a9-11ea-886e-51c0587d4327/fig-5-193.png)

    **4 단계**

    ​	'x' 노드는 순전파 때의 값을 서로 바꿔 곱합니다.

    ​	![img](https://media.vlpt.us/post-images/dscwinterstudy/1c21b130-41a9-11ea-bc96-9d2b4aed3aec/fig-5-20.png)

    위 그림과 같이 Sigmoid 계층의 역전파를 계산 그래프로 완성했습니다.

    그림에서 보듯이 역전파의 최종 출력인
    $$
    {\delta L\over \delta y}y^2exp(-x)
    $$
    값이 하류 노드로 전파됩니다.

    여기에서 위 식을 순전파의 입력 x와 출력 y만으로 계산할 수 있습니다.

    그래서 위 그림의 계산 그래프의 중간 과정을 모두 묶어 아래 그림처럼 단순한 sigmoid 노드 하나로 대체할 수 있습니다.

    ![img](https://media.vlpt.us/post-images/dscwinterstudy/20781300-41a9-11ea-baa8-418dfae37ccf/fig-5-21.png)

    위의 계산그래프와 아래의 간소화 버전의 결과는 똑같습니다.

    그러나 간소화 버전은 역전파 과정의 중간 계산들을 생략할 수 있어 더 효율적으로 계산을 할 수 있습니다.

* **Affine / Softmax 계층 구현하기**

  * **Affine 계층**

    신경망의 순전파에서는 가중치 신호의 총합을 계산하기 때문에 행렬의 곱을 사용했습니다.

    ```python
    X = np.random.rand(2)
    W = np.random.rand(2, 3)
    B = np.random.rand(3)
    
    X.shape
    W.shape
    B.shape
    
    Y = np.dot(X, W) + B
    ```

    여기에서 X, W, B는 각각 형상이 (2,), (2, 3), (3,)인 다차원 배열입니다.

    그러면 뉴런의 가정치 합은 Y = np.dot(X, W) + B 처럼 계산합니다.

    그리고 이 Y를 다시 활성화 함수로 변환해 다음 층으로 전파하는 것이 신경망 순전파의 흐름이었습니다.

    행렬의 곱 계산은 대응하는 차원의 원소 수를 일치시키는 게 핵심입니다.

    

    그럼 앞에서 수행한 계산을 계산 그래프로 그려봅시다.

    곱을 계산하는 노드를 'dot'이라 하면 np.dot(X, W) + B 계산은 아래 그림처럼 그려집니다.

    ![img](https://blog.kakaocdn.net/dn/bTGn9u/btqAKhfApnO/EIHkMsfrdk74BrLsRvPVC0/img.png)

    이는 비교적 단순한 계산 그래프입니다.

    지금까지의 계산 그래프는 노드 사이에 '스칼라값'이 흘렀는데 반해, 이 예에서는 '행렬'이 흐르고 있습니다.

    

    행렬을 사용한 역전파도 행렬의 원소마다 전개해보면 스칼라값을 사용한 지금까지의 계산 그래프와 같은 순서로

    생각할 수 있습니다. 전개해보면 다음 식이 도출됩니다.
    $$
    {\delta L \over \delta X} = {\delta L \over \delta Y}.W^T\\
    {\delta L \over \delta W} = X^T.{\delta L \over \delta Y}
    $$
    위 식에서 W^T의 T는 전치행렬을 뜻합니다.

    전치행렬은 W의 (i, j) 위치의 원소를 (j, i)위치로 바꾼 것을 말합니다.

    수식으로는 다음과 같이 쓸 수 있습니다.
    $$
    W = \left[
    \begin{matrix}
        W_{11} & W_{12} & W_{13}\\
        W_{21} & W_{22} & W_{23}\\
    \end{matrix}
    \right]\\
    W^T = \left[
    \begin{matrix}
        W_{11} & W_{21}\\
        W_{12} & W_{22}\\
        W_{13} & W_{23}\\
    \end{matrix}
    \right]
    $$
    위와 같이 W의 형상이 (2, 3) 이었다면 전치 행렬 W^T의 형상은 (3, 2)가 됩니다.

    

    계산 그래프의 역전파를 구해봅시다.

    결과는 아래 그림처럼 됩니다.

    ![img](https://blog.kakaocdn.net/dn/bBHb0k/btqALvjUUVe/P4srUaWP5vSp7gAXCnNfh0/img.png)

    ![img](https://blog.kakaocdn.net/dn/YXVSh/btqALPPYnZM/gkSUpcquaead8EfKPbkKQ0/img.png)

    이때, 행렬의 곱의 역전파는 행렬에 대응하는 차원의 원소수가 일치하도록 잘 조립해주어야 합니다.

    ![img](https://blog.kakaocdn.net/dn/oHhBI/btqAJvZGwDQ/ajl59KDzcg5opKlhxU1gt0/img.png)

  * **Softmax-with-Loss 계층**

    소프트맥스 함수는 입력 값을 정규화하여 출력합니다.

    예를 들어 손글씨 숫자 인식에서의 Softmax 계층의 출력은 아래 그림처럼 됩니다.

    ![img](https://blog.kakaocdn.net/dn/zGPE4/btqALQH9zso/bpLtNaGnlk61XWurzui000/img.png)

    그림과 같이 Softmax 계층은 입력 값을 정규화하여 출력합니다.
    
    또한, 손글씨 숫자는 가짓수가 10개이므로 Softmax 계층의 입력은 10개가 됩니다.
    
    먼저 Softmax-with-Loss 계층의 계산 그래프입니다.
    
    ![img](https://blog.kakaocdn.net/dn/bpGksi/btqVcrhYzBd/Wg3IzKLpKdilp6AXkWLER0/img.png)
    
    위의 계산 그래프는 아래 그림처럼 간소화할 수 있습니다.
    
    ![img](https://blog.kakaocdn.net/dn/brXL4X/btqVbXIflGl/Ac0AEvNtk2TQzDY2CV9o6K/img.png)
    
    그림의 계산 그래프에서 소프트맥스 함수는 Softmax 계층으로, 교차 엔트로피 오차는 Cross Entropy Error 계층으로
    
    표기했습니다.
    
    여기에서는 3클래스 분류를 가정하고 이전 계층에서 3개의 입력을 받습니다.
    
    그림과 같이 Softmax 계층은 입력(a_1, a_2, a_3)을 정규화하여 (y_1, y_2, y_3)를 출력합니다.
    
    Cross Entropy Error 계층은 Softmax의 출력 (y_1, y_2, y_3)와 정답 레이블 (t_1, t_2, t_3)를 받고, 이 데이터들로부터 손실
    
    L을 출력합니다.
    
    
    
    Softmax 계층의 역전파는 (y_1 - t_1, y_2 - t_2, y_3 - t_3)오라는 결과를 말끔한 결과를 내놓고 있습니다.
    
    (y_1, y_2, y_3) 는 Softmax 계층의 출력이고 (t_1, t_2, t_3)는 정답 레이블이므로 (y_1 - t_1, y_2 - t_2, y_3 - t_3)는 Softmax
    
    계층의 출력과 정답 레이블의 차분인 것입니다.
    
    신경망의 역전파에서는 이 차이인 오차가 앞 계층에 전해지는 것입니다.
    
    이는 신경망 학습의 중요한 성질입니다.
    
    
    
    그런데 신경망 학습의 목적은 신경망의 출력이 정답 레이블과 가까워지도록 가중치 매개변수의 값을 조정하는 것이었습니다.
    
    그래서 신경망의 출력과 정답 레이블의 오차를 효율적으로 앞 계층에 전달해야 합니다.
    
    앞의 (y_1 - t_1, y_2 - t_2, y_3 - t_3)라는 결과는 바로 Softmax 계층의 출력과 정답 레이블의 차이로, 신경망의 현재 출력과
    
    정답 레이블의 오차를 있는 그대로 드러내는 것입니다.
    
    
    
    가령 정답 레이블이 (0, 1, 0) 일 때 Softmax 계층이 (0.3, 0.2, 0.5)를 출력했다고 해봅시다.
    
    정답 레이블을 보면 정답의 인덱스는 1입니다. 그런데 출력에서는 이때의 확률이 0.2라서 이 시점의 신경망은 제대로
    
    인식하지 못하고 있습니다. 이 경우 Softmax 계층의 역전파는 (0.3, -0.8, 0.5)라는 커다란 오차를 전파합니다.
    
    결과적으로 Softmax 계층의 앞 계층들은 그 큰 오차로부터 큰 깨달음을 얻게 됩니다.
    
    
    
    이번에 살펴볼 예는 정답 레이블이 똑같이 (0, 1, 0) 일 때 Softmax 계층이 (0.01, 0.99, 0)을 출력한 경우입니다.
    
    이 경우 Softmax 계층의 역전파가 보내는 오차는 비교적 작은 (0.01, -0.01, 0) 입니다. 이번에는 앞 계층으로 전달된
    
    오차가 작으므로 학습하는 정도도 작아집니다.
    
    ```python
    class SoftmaxWithLoss:
    	def __init__(self):
    		self.loss = None
    		self.y = None
    		self.t = None
    		
    	def forward(self, x, t):
    		self.t = t
    		self.y = softmax(x)
    		self.loss = cross_entropy_error(self.y, self.t)
    		return self.loss
    		
    	def backward(self, dout=1):
    		batch_size = self.t.shape[0]
    		dx = (self.y - self.t) / batch_size
    		
    		return dx
    ```
  
* **오차역전파법 구현하기**

  * **신경망 학습의 전체 그림**
  
    다음은 신경망 학습의 순서입니다.
  
    * 전제
  
      신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 학습이라
  
      합니다. 신경망 학습은 다음과 같이 4단계로 수행합니다.
  
    * 1단계 - 미니배치
  
      훈련 데이터 중 일부를 무작위로 가져옵니다.
  
      이렇게 선별한 데이터를 미니배치라 하며, 그 미니배치의 손실 함수 값을 줄이는 것이 목표입니다.
  
    * 2단계 - 기울기 산출
  
      미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구합니다. 기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시합니다.
  
    * 3단계 - 매개변수 갱신
  
      가중치 매개변수를 기울기 방향으로 아주 조금 갱신합니다.
  
    * 4단계 - 반복
  
      1~3단계를 반복합니다.
  
    지금까지 설명한 오차역전파법이 등장하는 단계는 두 번쨰인 '기울기 산출'입니다.
  
    앞 장에서는 이 기울기를 구하기 위해서 수치 미분을 사용했습니다. 그런데 수치 미분은 구현하기는 쉽지만 계산이 오래 걸렸습니다.
  
    오차역전파법을이용하면 느린 수치 미분과 달리 기울기를 효율적이고 빠르게 구할 수 있습니다.
  
  * **오차역전파법을 적용한 신경망 구현하기**
  
    여기에서는 2층 신경망을 TwoLayerNet 클래스로 구현합니다.
  
    | 인스턴스 변수 | 설명                                                         |
    | ------------- | ------------------------------------------------------------ |
    | params        | 딕셔너리 변수로, 신경망의 매개변수를 보관                    |
    | layers        | params['W1']은 1번째 층의 가중치, params['b1']은 1번째 층의 편향 |
    |               | params['W2']은 2번째 층의 가중치, params['b2']는 2번째 층의 편향 |
    |               | layers['Affine1'], layers['Relu1'], layers['Affine2']와 같이 각 계층을 순서대로 유지 |
    | lastLayer     | 신경망의 마지막 계층, 이 예에서는 SoftmaxWithLoss 계층       |
  
    
  
    | 메서드                                                       | 설명                                                         |
    | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | _init_(self, input_size, hidden_size, output_size, wight_init_std) | 초기화를 수행한다 인수는 앞에서부터 입력층 뉴런              |
    |                                                              | 은닉층 뉴런 수, 출력층 뉴런 수, 가중치 초기화 시 정규분포의 스케일 |
    | predict(self, x)                                             | 예측을 수행한다. 인수 x는 이미지 데이터                      |
    | loss(self, x)                                                | 손실함수의 값을 구한다. 인수 x는 이미지 데이터, t는 정답 레이블 |
    | accuracy(self, x, t)                                         | 정확도를 구한다.                                             |
    | numerical_gradient(self, x, t)                               | 가중치 매개변수의 기울기를 수치 미분 방식으로 구한다.        |
    | gradient(self, x, t)                                         | 가중치 매개변수의 기울기를 오차역전파법으로 구한다.          |
  
    # coding: utf-8
    import sys, os
    sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
    import numpy as np
    from common.layers import *
    from common.gradient import numerical_gradient
    from collections import OrderedDict
  
  
    class TwoLayerNet:
  
    ```python
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
    
        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
    
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
    
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
    
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
    
        return grads
    ```
  
    OrderedDict은 순서가 있는 딕셔너리입니다. '순서가 있는'이란 딕셔너리에 추가한 순서를 기억하다는 것입니다.
  
    그래서 순전파 때는 추가한 순서대로 각 계층의 forward() 메서드를 호출하기만 하면 처리가 완료됩니다.
  
    마찬가지로 역전파 때는 계층을 반대 순서로 호출하기만 하면 됩니다.
  
    Affine 계층과 ReLU 계층이 각자 내부에서 순전파와 역전파를 처리하고 있으니, 여기에서는 계층을 연결한 다음 순서대로 호출해주면 끝입니다.
  
    이처럼 신경망의 구성 요소를 계층으로 구현한 덕분에 신경망을 쉽게 구축할 수 있었습니다.
  
  * **오차역전파법으로 구한 기울기 검증하기**
  
    수치미분은 느립니다. 그리고 오차역전파법을 제대로 구현해두면 수치 미분은 더 이상 필요없습니다.
  
    수치 미분은 오차역전파법을 정확하게 구현했는지 확인하기 위해 필요합니다.
  
    ```python
    # coding: utf-8
    import sys, os
    sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
    import numpy as np
    from dataset.mnist import load_mnist
    from two_layer_net import TwoLayerNet
    
    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    x_batch = x_train[:3]
    t_batch = t_train[:3]
    
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)
    
    # 각 가중치의 절대 오차의 평균을 구한다.
    for key in grad_numerical.keys():
        diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
        print(key + ":" + str(diff))
    ```
  
    이 코드의 실행 결과는 다음과 같습니다.
  
    ```d
    b1:9.70418809871e-13
    W2:8.41139039497e-13
    b2:1.1945999745e-10
    W2:2.2232446644e-13
    ```
  
    이 결과는 수치 미분과 오차역전파법으로 구한 기울기의 차이가 매우 작다고 말해줍니다.
  
    이로써 오차역전파법으로 구한 기울기도 올바름이 드러나면서 실수 없이 구현했다고 생각할 수 있습니다.
  
  * **오차역전파법을 사용한 학습 구현하기**
  
    ```python
    # coding: utf-8
    import sys, os
    sys.path.append(os.pardir)
    
    import numpy as np
    from dataset.mnist import load_mnist
    from two_layer_net import TwoLayerNet
    
    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 기울기 계산
        #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
        grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
        
        # 갱신
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)
    ```
  
* **정리**

  계산 과정을 시각적으로 보여주는 방법인 계산 그래프를 배웠습니다. 계산 그래프를 이용하여 신경망의 동작과 오차역전파법을 설명하고, 그 처리과정을 계층이라는 단위로 구현했습니다.

  ReLU 계층, Softmax-with-Loss 계층, Affine 계층, Softmax 계층 같은 것을 구현했습니다.

  모든 계층에서 forward와 backward라는 메서드를 구현합니다.

  forward는 데이터를 순방향으로 전파하고, backward는 역방향으로 전파함으로써 가중치 매개변수의 기울기를 효율적으로 구할 수

  있습니다. 이처럼 동작을 계층으로 모듈화한 덕분에, 신경망의 계층을 자유롭게 조합하여 원하는 신경망을 쉽게 만들 수 있습니다.

  **이번 장에서 배운 내용**

  * 계산 그래프를 이용하면 계산 과정을 시각적으로 파악할 수 있습니다.
  * 계산 그래프의 노드는 국소적 계산으로 구성됩니다. 국소적 계산을 조합해 전체 계산을 구성합니다.
  * 계산 그래프의 순전파는 통상의 계산을 수행한다. 한편, 계산 그래프의 역전파로는 각 노드의 미분을 구할 수 있다.
  * 신경망의 구성 요소를 계층으로 구현하여 기울기를 효율적으로 계산할 수 있습니다(오차역전파법).
  * 수치 미분과 오차역전파법의 결과를 비교하면 오차역전파법의 구현에 잘못이 없는지 확인할 수 있습니다(기울기 확인).

## 학습 관련 기술들

* **매개변수 갱신**

  신경망 학습의 목적은 손실 함수의 값을 가능한 한 낮추는 매개변수를 찾는 것이였습니다.

  이는 곧 매개변수의 최적값을 찾는 문제이며, 이러한 문제를 푸는 것을 최적화라 합니다.

  신경망 최적화는 굉장히 어려운 문제입니다. 매개변수 공간은 매우 넓고 복잡해서 최적의 솔루션을 쉽게 찾을 수 없기 때문입니다.

  수식을 풀어 순식간에 최소값을 구하는 방법 같은 것은 없습니다.

  게다가 심층 신경망에서는 매개변수의 수가 엄청나게 많아져서 사태는 더욱 심각해집니다.

  

  우리는 지금까지 최적의 매개변수 값을 찾는 단서로 매개변수의 기울기를 이용했습니다.

  매개변수의 기울기를 구해, 기울어진 방향으로 매개변수 값을 갱신하는 일을 몇번이고 반복해서 점점 최적의 값에 다가갔습니다.

  이것이 확률적 경사 하강법(SGD)이라는 단순한 방법인데, 매개변수 공간을 무작정 찾는 것 보다 똑똑한 방법입니다.

  SGD는 단순하지만, SGD보다 똑똑한 방법도 있습니다.

  * **모험가 이야기**

    최적화를 해야 하는 우리의 상황을 모험가 이야기에 비유해보겠습니다.

    > 색다른 모험가가 있습니다. 광할한 메마른산맥을 여행하면서 날마다 깊은 골짝를 찾아 발걸음을 옮깁니다.
    >
    > 그는 전설에 나오는 세상에서 가장 깊고 낮은 골짜기, 깊은 곳을 찾아가려 합니다.
    >
    > 그것이 그의 여행 목적이죠. 게다가 그는 엄격한 제약 2개로 자신을 옭아맸습니다.
    >
    > 하나는 지도를 보지 않을 것, 또 하나는 눈가리개를 쓰는 것입니다. 지도도 없고 보이지도 않으니 가장 낮은 골짜기가
    >
    > 광대한 땅 어디에 있는지 알 도리가 없습니다. 그런 혹독한 조건에서 이 모험가는 어떻게 깊은 곳을 찾을 수 있을까요?
    >
    > 어떻게 걸음을 옮겨야 효율적으로 깊은 곳을 찾아낼 수 있을까요?

    최적의 매개변수를 탐색하는 우리도 이 모험가와 같은 어둠의 세계를 탐험하게 됩니다.

    광대하고 복잡한 지형을 지도도 없이 눈을 가린 채로 깊은 곳을 찾지 않으면 안됩니다.

    

    이 어려운 상황에서 중요한 단서가 되는 것이 땅의 기울기입니다.

    모험가는 주위 경치는 볼 수 없지만 지금 서 있는 땅의 기울기는 알 수 있습니다. 발바닥으류ㅗ 전해지죠.

    그래서 지금 서 있는 장소에서 가장 크게 기울어진 방향으로 가자는 것이 SGD의 전략입니다.

    이 일을 반복하면 언젠가 깊은 곳에 찾아갈 수 있습니다.

  * **확률적 경사 하강법(SGD)**

    SGD는 수식으로는 다음과 같이 쓸 수 있습니다.
    $$
    W <-\; W - \eta{\delta L\over \delta W}
    $$
    여기에서 W는 갱신할 가중치 매개변수고 δL/δW은 W에 대한 손실 함수의 기울기입니다.

    n는 학습률을 의미하는데, 실제로는 0.01이나 0.001 같은 값을 미리 정해서 사용합니다.

    또 <-는 우변의 값으로 좌변의 값을 갱신한다는 뜻입니다. 식에서 보듯 SGD는 기울어진 방향으로 일정 거리만

    가겠다는 단순한 방법입니다.

    ```python
    class SGD:
    	def __init__(self, lr=0.01)
    		self.lr = lr
    		
    	def update(self, params, grads):
    		for key in params.keys():
    			params[key] -= self.lr * grads[key]
    ```

    초기화 때 받는 인수인 lr은 learning rate(학습률)을 뜻합니다. 이 학습률을 인스턴스 변수로 유지합니다.

    update 메서드는 SGD 과정에서 반복해서 불립니다. 인수인 params와 grads는 딕셔너리 변수입니다.

    params['W1'], grads['W1'] 등과 같이 각각 가중치 매개변수와 기울기를 저장하고 있습니다.

    

    SGD 클래스를 사용하면 신경망 매개변수의 진행을 다음과 같이 수행할 수 있습니다.

    ```python
    network = TwoLayerNet(...)
    optimizer = SGD()
    
    for i in range(10000):
    	...
    	x_batch, t_batch = get_mini_batch(...)
    	grads = network.gradient(x_batch, t_batch)
    	params = network.params
    	optimizer.update(params, grads)
    	...
    ```

    optimizer는 최적화를 행하는 자라는 뜻의 단어입니다.

    이 코드에서는 SGD가 그 역할을 합니다.

    매개변수 갱신은 optimizer가 책임지고 수행하니 우리는 optimizer에 매개변수와 기울기 정보만 넘겨주면 되는 것입니다.

    이처럼 최적화를 담당하는 클래스를 분리해 구현하면 기능을 모듈화하기 좋습니다.

    모멘텀이라는 최적화 기법 역시 update(params, gards)라는 공통의 메서드를 갖도록 구현합니다.

    그때 optimizer = SGD() 문장을 optimizer = Momentum()으로만 변경하면 SGD가 모멘텀으로 바뀌는 것이죠.

  * **SGD의 단점**

    SGD는 단순하고 구현도 쉽지만, 문제에 따라서는 비효율적일 때가 있습니다.

    SGD의 단점을 알아보고자 다음 함수의 최솟값을 구하는 문제를 생각해보겠습니다.
    $$
    f(x,y) = {1\over 20}x^2+y^2
    $$
    이 함수는 그림의 왼쪽과 같이 밥그릇을 x출 방향으로 늘인 듯한 모십이고, 실제로 그 등고선은 오른쪽과 같이 x축 방향으로 늘인 

    타원으로 되어 있습니다.

    ![img](https://blog.kakaocdn.net/dn/buUtdh/btq3USK5kMv/dKBBLfn80fk1LAqvez4K6K/img.png)

    위 식의 함수의 기울기를 그려보면 아래 그림처럼 됩니다.

    이 기울기는 y축 방향은 크고 x축 방향은 작다는 것이 특징입니다.

    말하자면 y축 방향은 가파른데 x축 방향은 완만한 것이죠.

    또, 여기에서 주의할 점으로는 위 수식이 최솟값이 되는 장소는 (x, y) = (0, 0)이지만, 아래 그림이 보여주는 기울기의 대부분은

    (0, 0) 방향을 가리키지 않는다는 것입니다.

    ![img](https://blog.kakaocdn.net/dn/pNmVS/btq3TsGyuzu/q02r2Dxbs9cGYncG0QyZRk/img.png)

    이제 그림에 SGD를 적용하고 초깃값은 (x, y) = (-7.0, 2.0)으로 하겠습니다. 결과는 아래 그림과 같습니다.

    ![img](https://blog.kakaocdn.net/dn/VneJh/btq3U63sLtN/b7G2s8KfeaZskCKTHOkV5K/img.png)

    SGD는 그림과 같은 심하게 굽이진 움직임을 보여줍니다. 상당히 비효율적인 움직임입니다.

    즉, SGD의 단점은 비등방성 함수에서는 탐색 경로가 비효율적이라는 것입니다.

    이럴 때는 SGD 같이 무작정 기울어진 방향으로 진행하는 단순한 방식보다 더 영리한 묘안이 간절해집니다.

    또한, SGD가 지그재그로 탐색하는 근본 원인은 기울어진 방향이 본래의 최솟값과 다른 방향을 가리켜서라는 점도 생각해볼 필요가 있습니다.

  * **모멘텀**

    모멘텀은 운동량을 뜻하는 단어로, 물리와 관계가 있습니다.

    모멘텀 기법은 수식으로는 다음과 같이 쓸 수 있습니다.
    $$
    v <-\; \alpha v - \eta {\delta L\over \delta W}\\
    W <-\; W + v
    $$
    SGD의 수식처럼 여기에서도 W는 갱신할 가중치 매개변수, δL/δW은 W에 대한 손실 함수의 기울기, n은 학습률입니다.

    v라는 새로운 변수가 나오는데, 이는 물리에서 말하는 속도에 해당합니다.

    위 수식은 기울기 방향으로 힘을 받아 물체가 가속된다는 물리 법칙을 나타냅니다.

    모멘텀은 그림과 같이 고잉 그릇의 바닥을 구르는 듯한 움직임을 보여줍니다.

    ![img](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMjY3/MDAxNTAxMTIzMjgxNzM3.w9b6TOLwpgUiewZxUZURx6fqnkNIeEK74p6JJ7JzTxwg.yiFTwTm_cPvGVkcBJlT13igEPmNAoFXAXMeTrLWcE2cg.PNG.cjswo9207/fig_6-4.png?type=w2)

    또 위 수식의 av항은 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할을 합니다. (Alpha는 0.9등의 값으로 설정합니다.)

    물리에서의 지면 마찰이나 공기 저항에 해당합니다.

    다음은 모멘텀의 구현입니다.

    class Momentum:

    ```python
    class Momentum:
        def __init__(self, lr=0.01, momentum=0.9):
            self.lr = lr
            self.momentum = momentum
            self.v = None
    
        def update(self, params, grads):
            if self.v is None:
                self.v = {}
                for key, val in params.items():                                
                    self.v[key] = np.zeros_like(val)
    
            for key in params.keys():
                self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
                params[key] += self.v[key]
    ```

    인스턴스 변수 v가  물체의 속도입니다.

    v는 초기화 때는 아무 값도 담지 않고, 대신 update()가 처음 호출될 떄 매개변수와 같은 구조의 데이터를 딕셔너리 변수로 저장합니다.

    나머지 부분은 수식을 간단히 코드로 옮긴 것입니다.

    모멘텀을 사용해서 최적화 문제를 풀어봅시다. 결과는 아래 그림 처럼 됩니다.

    ![img](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMjQ5/MDAxNTAxMTIzNzcxMjY4.pW8LeX0QqcoQMI1qBQmpYf9skv8k0G2SBAqrmk_uP84g.CKPbYA_icOECAZXhBTB_g1qzmDRdtL3vUbEfbU4Y8yMg.PNG.cjswo9207/fig_6-5.png?type=w2)

    그림에서 보듯 모멘텀의 갱신 경로는 공이 그릇 바닥을 구르듯 움직입니다.

    SGD와 비교하면 지그재그 정도가 덜한 것을 알 수 있습니다.

    이는 x축의 힘은 아주 작지만 방향은 변하지 않아서 한 방향으로 일정하게 가속하기 떄문입니다.

    거꾸로 y축의 힘은 크지만 위아래로 번갈아 받아서 상충하여 y축 방향의 속도는 안정적이지 않습니다.

    전체적으로는 SGD보다 x축 방향으로 빠르게 다가가 지그재그 움직임이 줄어듭니다.
    
  * **AdaGrad**
  
    신경망 학습에서는 학습률 값이 중요합니다.
  
    이 값이 너무 작으면 학습 시간이 너무 길어지고, 반대로 너무 크면 발산하여 학습이 제대로 이뤄지지 않습니다.
  
    이 학습률을 정하는 효과적 기술로 학습률 감소가 있습니다.
  
    이는 학습을 진행하면서 학습률을 점차 줄여가는 방법입니다. 
  
    처음에는 크게 학습하다가 조금씩 작게 학습한다는 얘기로, 실제 신경망 학습에 자주 쓰입니다.
  
    학습률을 서서이 낮추는 가장 간단한 방법은 매개변수 전체의 학습률 값을 일괄적으로 낮추는 것입니다.
  
    이를 더욱 발전시킨 것이 AdaGrad입니다.
  
    AdaGrad는 각각의 매개변수에 맞춤형 값을 만들어줍니다.
  
    AdaGrad는 개별 매개변수에 적응적으로 학습률을 조정하면서 학습을 진행합니다.
  
    AdaGrad의 갱신 방법은 수식으로는 다음과 같습니다.
    $$
    h <-\; {\delta L\over \delta W} \odot {\delta L\over \delta W}\\
    W <-\; W - \eta{1\over\sqrt{h}}{\delta L \over \delta W}
    $$
    마찬가지로 W는 갱신할 가중치 매개변수, δL/δW은 W에 대한 손실 함수의 기울기, n는 학습률을 뜻합니다.
  
    여기에서는 h라는 새로운 변수가 등장합니다.
  
    h는 식에서 보듯 기존 기울기 값을 제곱하여 계속 더해줍니다.
  
    그리고 매개변수를 갱신할 때 1/sqrt(h)을 곱해 학습률을 조정합니다.
  
    매개변수의 원소 중에서 많이 움직인 원소는 학습률이 낮아진다는 뜻인데, 다시 말해 학습률 감소가 매개변수의 원소마다
  
    다르게 적용됨을 뜻합니다.
  
    > AdaGrad는 학습을 진행할수록 갱신 강도가 약해집니다. 실제로 무한히 계속 학습한다면 어느 순간 갱신량이 0이 되어
    >
    > 전혀 갱신하지 않게 됩니다. 이 문제를 개선한 기법으로서 RMSProp라는 방법이 있습니다.
    >
    > PMSProp은 과거의 모든 기울기를 균일하게 더해가는 것이 아니라, 먼 과거의 기울기는 서서히 잊고 새로운 기울기 정보를
    >
    > 크게 반영합니다. 이를 지수이동평균(Exponential Moving Average, EMA)라고 하여, 과거의 기울기 반영 규모를
    >
    > 기하급수적으로 감소시킵니다.
  
    ```python
    class AdaGrad:
        def __init__(self, lr=0.01):
            self.lr = lr
            self.h = None
    
        def update(self, params, grads):
            if self.h is None:
                self.h = {}
                for key, val in params.items():
                    self.h[key] = np.zeros_like(val)
    
            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
    ```
  
    여기서 주의할 것은 마지막 줄에서 1e-7이라는 작은 값을 더하는 부분입니다.
  
    이 작은 값은 self.h[key]에 0이 담겨 있다 해도 0으로 나누는 사태를 막아줍니다.
  
    대부분의 딥러닝 프레임워크에서는 이 값도 인수로 설정할 수 있습니다.
  
    ![img](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfNDQg/MDAxNTAxMTI0NTczMDQz.1HzCY4kj0AqKMon65oOJdYfaIxGveuUjawHvH_XOtnkg.0Pf9RCSvNNbZiEqybl7EOt2b49lmHFcWSbFeBpnQG2Yg.PNG.cjswo9207/fig_6-6.png?type=w2)
  
    그림을 보면 최솟값을 향해 효율적으로 움직이는 것을 알 수 있습니다.
  
    y축 방향은 기울기가 커서 크게 움직이지만, 그 큰 움직임에 비례해 갱신 정도도 큰 폭으로 작아지도록 조정됩니다.
  
    그래서 y축 방향으로 갱신 강도가 빠르게 약해지고, 지그재그 움직임이 줄어듭니다.
  
  * **Adam**
  
    모멘텀은 공이 그릇 바닥을 구르는 듯한 움직임을 보였습니다.
  
    AdaGrad는 매개변수의 원소마다 적응적으로 갱신 정도를 조정했습니다.
  
    이 두 기법을 융합한 것이 Adam입니다.
  
    Adam은 2015년에 제안된 새로운 방법입니다.
  
    그 이론은 다소 복잡하지만 직관적으로는 모멘텀과 AdaGrad를 융합한 듯한 방법입니다.
  
    이 두 방법의 이점을 조합했다면, 매개변수 공간을 효율적으로 탐색해줄 것으로 기대할 수 있습니다.
  
    또, 하이퍼파라미터의 편향 보정이 진행된다는 점도 Adam의 특징입니다.
  
    그럼 Adam을 사용해서 최적화 문제를 풀어보겠습니다. 결과는 아래 그림과 같습니다.
  
    ![img](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMTY2/MDAxNTAxMTI0OTE5MzQz.uHy8VlKoQt9RgdMVQW0MZeae_puzDlTicctRfuKwwHIg.fET0kHfJPIXGMn261G-pdMItHWRM_hE6Y9tX2EMSa20g.PNG.cjswo9207/fig_6-7.png?type=w2)
  
  * **어느 갱신 방법을 이용할 것인가?**
  
    지금까지 매개변수의 갱신 방법을 4개 살펴봤습니다.
  
    이번 절에서 ㅇ는 이들 네 기법의 결과를 비교해보겠습니다.
  
    ![img](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfNzMg/MDAxNTAxMTI1MjA3ODcx.n8cnkeeoNqZ5gll9jQrOJZ_UJ6KuTp_EKkz8r-hrqUsg.UbIaXT3BY5BVdrF2N29WLynj4XvOutQXpzfMF9UyBRQg.PNG.cjswo9207/fig_6-8.png?type=w2)
  
    그림과 같이 사용한 기법에 따라 갱신 경로가 다릅니다. 이 그림만 보면 AdaGrad가 가장 나은 것 같은데, 사실 그 결과는 풀어야 할 문제가 무엇이냐에 따라 달라집니다.
  
    또, 당연하지만 하이퍼파라미터를 어떻게 설정하느냐에 따라서도 결과가 달라집니다.
  
  * **MNIST 데이터셋으로 본 갱신 방법 비교**
  
    손글씨 숫자 인식을 대상으로 지금까지 설명한 네 기법을 비교해봅시다.
  
    각 방법의 학습 진도가 얼마나 다른지를 그림에 그려보았습니다.
  
    ![img](data:image/png;base64,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%0AAABJRU5ErkJggg==)
  
    이 실험은 각 층이 100개의 뉴런으로 구성된 5층 신경망에서 ReLU를 활성화 함수로는 사용해 측정했습니다.
  
    그림의 결과를 보면 SGD의 학습 진도가 가장 느립니다.
  
    나머지 세 기법의 진도는 비슷한데, 잘 보면 AdaGrad가 조금 더 빠른 것 같습니다.
  
    이 실험에서 주의할 점은 하이퍼파라미터인 학습률과 신경망의 구조에 따라 결과가 달라진다는 것입니다.
  
    다만 일반적으로 SGD보다 다른 세 기법이 빠르게 학습하고, 때로는 최종 정확도도 높게 나타납니다.