# Pyomo Tutorial v1 (2025.03.19) 

[Test_Pyomo.py](./Test_Pyomo.py)

## Pyomo 설명 

- Python 기반의 최적화 프로그램
  - 무료이며, 유료 및 무료 solver와 함께 사용 가능함
  - Linear Programming과 함께 Nonlinear Programming을 함께 해결할 수 있음
    - 단, Nonlinear programming은 solver의 특성을 타는 것으로 보임
  - 사이트
    - https://www.pyomo.org/
    - Ipopt solver 사용을 위하여 Conda를 이용하여 설치

## 예제 파일
- 비선형 최적화 문제 예제

$$
\begin{align*}
Minimize &\quad &x+sin(y) \\
\quad s.t. &\quad  &10 \leq x \leq 20 \\
&\quad &\pi \leq y \leq \frac{5}{2}\pi
\end{align*}
$$

  - 예제에서의 답은 $x$가 10이고, $sin(y)$가 -1이 되는 지점일 것이다.
    - Optimal Point: $x=10, y=\frac{3}{2}\pi$
- Code step by step
  - Module 불러오기
    - ```
      import pyomo.environ as pyo 
      ```
  - Model 만들기
    - ```
      model = pyo.ConcreteModel()
      ```
  - 변수 만들기 (Variables)
    - ```
      model.x = pyo.Var(within=pyo.NonNegativeReals,initialize=10.0)
      model.y = pyo.Var(within=pyo.NonNegativeReals,initialize=math.pi/2)
      ```
      - x와 y에 초기값 지정
  - 제약조건(Constraints)
    - ```
      def xregion1(model):
          return model.x>=10 # return model.x==10
      model.Boundx1 = pyo.Constraint(rule=xregion1)

      def xregion2(model):
          return model.x<=20 # return model.x==20
      model.Boundx2 = pyo.Constraint(rule=xregion2)

      def yregion1(model):
          return model.y<=math.pi/2*5
      model.Boundy1 = pyo.Constraint(rule=yregion1)

      def yregion2(model):
          return model.y>=math.pi/2*2
      model.Boundy2 = pyo.Constraint(rule=yregion2)
      ```
      - 제약조건은 함수로 define하여 입력해야 함
      - 제약조건 이름은 'model.(제약조건 이름)'으로 설정
      - 넣고 싶은 제약조건은 'pyo.Constraint'뒤에 (rule=제약조건으로 사용할 함수 이름) 으로 넣기
  - 목적함수
    - ```
      def obj_rule(model):                                        
          return  model.x + pyo.sin(model.y) # x + sin(y)
      model.obj = pyo.Objective(rule=obj_rule,sense=pyo.minimize) # if you want to maximize objective, use 'sense=pyo.maximize'
      ```
    - 제약조건과 마찬가지로 함수로 define
    - 목적함수의 이름과 넣는 방법은 동일

  - 결과 출력 부분
    - ```
      print('x= ' + str(model.x.value))
      print('y= ' + str(model.y.value))
      print('Objective function= ' + str(model.obj.expr()))
      ```