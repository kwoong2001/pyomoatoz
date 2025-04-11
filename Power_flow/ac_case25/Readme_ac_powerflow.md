# Optimization model for calculating AC power flow

## Activity
이번 활동의 목적은 반복 계산법으로 해결하던 AC power flow 계산을 비선형 최적화 문제로 구현하는 것에 있음

## Formula

### Nomenclature
#### Indices
- $i, j$: Bus
#### Parameters
- $\alpha_{i}$: PV Bus indicator ($i$모선이 PV 모선 이면 1, 그렇지 않으면 0)
- $\beta_{i}$: PQ Bus indicator ($i$모선이 PQ 모선 이면 1, 그렇지 않으면 0)
- $P^{known}_{i}$: Known value of active power in Bus $i$
- $Q^{known}_{i}$: Known value of reactive power in Bus $i$
- $V^{known}_{i}$: Known value of voltage magnitude in Bus $i$
- $G_{ij}$: Conductance of the line between Bus $i$ and Bus $j$
- $B_{ij}$: Susceptance of the line between Bus $i$ and Bus $j$

#### Varaiables
- $P^{cal}_{i}$: Active power in Bus $i$
- $Q^{cal}_{i}$: Reactive power in Bus $i$
- $V^{cal}_{i}$: Voltage magnitude in Bus $i$
- $\theta^{cal}_{i}$: Voltage angle in Bus $i$

### Opitmization Formula

#### Overview
$$
\begin{align*}
Minimize &\quad \sum_{\forall i}{ (\alpha_{i} + \beta_{i})(P^{known}_{i} - P^{cal}_{i}) + \beta_{i}(Q^{known}_{i} - Q^{cal}_{i}) +\alpha_{i}(V^{known}_{i} - V^{cal}_{i}) }\\

\quad s.t. &\quad  P^{cal}_{i} = \sum_{\forall j}{V^{Cal}_{i}V^{Cal}_{j}(G_{ij}cos(\theta^{cal}_{i} - \theta^{cal}_{j}) + B_{ij}sin(\theta^{cal}_{i} - \theta^{cal}_{j}))}, \quad \forall i  \\

&\quad  Q^{cal}_{i} = \sum_{\forall j}{V^{Cal}_{i}V^{Cal}_{j}(G_{ij}sin(\theta^{cal}_{i} - \theta^{cal}_{j}) - B_{ij}cos(\theta^{cal}_{i} - \theta^{cal}_{j}))}, \quad \forall i \\

&\quad  P^{cal}_{i} \geq 0, \quad \forall i\\
&\quad  Q^{cal}_{i} \geq 0, \quad \forall i\\
&\quad  V^{cal}_{i} \geq 0, \quad \forall i\\
&\quad  \theta^{cal}_{i} \geq 0, \quad \forall i\\
\end{align*}
$$

#### Objective function
- 목적함수는 아는 값과 계산 값의 차이를 최소화하는 것임

$$
\begin{align*}
Minimize &\quad \sum_{\forall i}{ (\alpha_{i} + \beta_{i})(P^{known}_{i} - P^{cal}_{i}) + \beta_{i}(Q^{known}_{i} - Q^{cal}_{i}) +\alpha_{i}(V^{known}_{i} - V^{cal}_{i}) }\\
\end{align*}
$$

#### Constraints
- 전력방정식이 제약조건으로 입력됨
$$
\begin{align*}
\quad s.t. &\quad  P^{cal}_{i} = \sum_{\forall j}{V^{Cal}_{i}V^{Cal}_{j}(G_{ij}cos(\theta^{cal}_{i} - \theta^{cal}_{j}) + B_{ij}sin(\theta^{cal}_{i} - \theta^{cal}_{j}))}, \quad \forall i  \\

&\quad  Q^{cal}_{i} = \sum_{\forall j}{V^{Cal}_{i}V^{Cal}_{j}(G_{ij}sin(\theta^{cal}_{i} - \theta^{cal}_{j}) - B_{ij}cos(\theta^{cal}_{i} - \theta^{cal}_{j}))}, \quad \forall i \\

&\quad  P^{cal}_{i} \geq 0, \quad \forall i\\
&\quad  Q^{cal}_{i} \geq 0, \quad \forall i\\
&\quad  V^{cal}_{i} \geq 0, \quad \forall i\\
&\quad  \theta^{cal}_{i} \geq 0, \quad \forall i\\
\end{align*}
$$
