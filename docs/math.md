# Reward 与优化目标

## 基本定义

记状态为 $s_t$，末端当前位置为 $p_t$，目标点为 $p^\*$。

当前阶段的任务误差定义为

$$
d_t = d_{\text{task}}(s_t) = \|p_t - p^\*\|_2,\qquad d_t \ge 0
$$

记势函数为

$$
\phi_t = \phi(d_t)
$$

强化学习的标准折扣目标写为

$$
J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],\qquad 0<\gamma<1
$$

其中 $k>0$ 为 reward 缩放系数。

## 1. 无 $\gamma$ 的差分势函数

reward 定义为

$$
r_t = k\,[\phi(d_t)-\phi(d_{t+1})]
$$

代入标准折扣目标：

$$
J(\pi)
=
k\,\mathbb{E}_{\pi}\left[
\sum_{t=0}^{T-1}\gamma^t(\phi_t-\phi_{t+1})
\right]
$$

整理得

$$
J(\pi)
=
k\,\mathbb{E}_{\pi}\left[
\phi_0
-(1-\gamma)\sum_{t=1}^{T-1}\gamma^{t-1}\phi_t
- \gamma^{T-1}\phi_T
\right]
$$

由于 $\phi_0$ 与策略无关，等价于最小化

$$
\min_{\pi}\;
\mathbb{E}_{\pi}\left[
(1-\gamma)\sum_{t=1}^{T-1}\gamma^{t-1}\phi(d_t)
\;+\;
\gamma^{T-1}\phi(d_T)
\right]
$$

这一形式同时包含：

$$
\text{路径项} \;+\; \text{终点项}
$$

## 2. 带 $\gamma$ 的差分势函数

reward 定义为

$$
r_t = k\,[\phi(d_t)-\gamma\,\phi(d_{t+1})]
$$

代入标准折扣目标：

$$
J(\pi)
=
k\,\mathbb{E}_{\pi}\left[
\sum_{t=0}^{T-1}\gamma^t(\phi_t-\gamma\phi_{t+1})
\right]
$$

利用望远镜求和可得

$$
J(\pi)
=
k\,\mathbb{E}_{\pi}\left[
\phi(d_0)-\gamma^T\phi(d_T)
\right]
$$

由于 $\phi(d_0)$ 与策略无关，等价于最小化

$$
\min_{\pi}\;
\mathbb{E}_{\pi}\left[\gamma^T\phi(d_T)\right]
$$

这一形式主要对应

$$
\text{终点误差最小化}
$$

## 3. 直接阶段代价

reward 定义为

$$
r_t = -k\,\phi(d_t)
$$

代入标准折扣目标：

$$
J(\pi)
=
-k\,\mathbb{E}_{\pi}\left[
\sum_{t=0}^{T-1}\gamma^t\phi(d_t)
\right]
$$

等价于最小化

$$
\min_{\pi}\;
\mathbb{E}_{\pi}\left[
\sum_{t=0}^{T-1}\gamma^t\phi(d_t)
\right]
$$

这一形式主要对应

$$
\text{全过程跟踪误差最小化}
$$

## `log` 势函数

为增强目标附近的分辨率，可以采用

$$
\phi(d)=\log\left(1+\frac{d}{\varepsilon}\right)
$$

其中 $\varepsilon>0$ 为距离尺度参数。当前实现中可取

$$
\varepsilon = d_{\text{tol}}
$$

即

$$
\phi(d_t)=\log\left(1+\frac{d_t}{d_{\text{tol}}}\right)
$$

于是三类 reward 分别写为

$$
r_t^{(1)}
=
k\left[
\log\left(1+\frac{d_t}{\varepsilon}\right)
-
\log\left(1+\frac{d_{t+1}}{\varepsilon}\right)
\right]
$$

$$
r_t^{(2)}
=
k\left[
\log\left(1+\frac{d_t}{\varepsilon}\right)
-
\gamma\log\left(1+\frac{d_{t+1}}{\varepsilon}\right)
\right]
$$

$$
r_t^{(3)}
=
-k\log\left(1+\frac{d_t}{\varepsilon}\right)
$$

## 无 `log` 的线性消融

若直接取

$$
\phi(d)=d
$$

则得到线性版本：

$$
r_t^{(1)} = k(d_t-d_{t+1})
$$

$$
r_t^{(2)} = k(d_t-\gamma d_{t+1})
$$

$$
r_t^{(3)} = -k d_t
$$

## 当前默认形式

当前代码默认使用

$$
r_t = k\,[\phi(d_t)-\phi(d_{t+1})],
\qquad
\phi(d)=\log\left(1+\frac{d}{d_{\text{tol}}}\right)
$$

对应的等价优化目标为

$$
\min_{\pi}\;
\mathbb{E}_{\pi}\left[
(1-\gamma)\sum_{t=1}^{T-1}\gamma^{t-1}\phi(d_t)
\;+\;
\gamma^{T-1}\phi(d_T)
\right]
$$
