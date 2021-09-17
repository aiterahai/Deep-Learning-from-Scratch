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