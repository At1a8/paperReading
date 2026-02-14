# Mamba: Linear-Time Sequence Modeling with Selective State Spaces
视频讲解：https://www.bilibili.com/video/BV1JCrVYjEPN/?spm_id_from=333.337.search-card.all.click&vd_source=438c274850337b47a0be54ae5ef6d6ca 

论文：https://arxiv.org/pdf/2312.00752

## 公式推导
经典的线性时不变系统，输入为$x(t)$, 系统状态为$h(t)$, 输出为$y(t)$, 系统方程（原理可参考上述视频或现代控制理论）可以表示为:<center>

$\dot{h}(t) = Ah(t) + Bx(t)$ (1)

$y(t) = Ch(t) + Dx(t)$ (2) </center>

为了能够适配LLM，token by token的输出，需要对上述连续系统进行离散化，得到通过 $k$ 时刻的 $h(t_k)$ 来表达 $k+1$ 时刻的 $h(t_{k+1})$ 的方案，即 $h_{k+1}=f(h_k)$

首先观察公式 (1) ，重写为<center>

$\dot{h}(t) - Ah(t) = Bx(t)$ (3) </center>

易看出函数 $F(t) = e^{-At}h(t)$，则有 $\dot{F}(t) = -Ae^{-At}h(t) + e^{-At}\dot{h}(t)$, 公式(3)两边同时乘以 $e^{-At}$ 得到：<center>

$e^{-At}\dot{h}(t) - Ae^{-At}h(t) = e^{-At}Bx(t)$ </center>

因此 $\dot{F}(t) = e^{-At}Bx(t)$

易知: $F(t) = F(\lambda) + \int_\lambda^t\dot{F}(\tau)d\tau$，设 $\lambda = 0 $, 则有<center>

$` F(t)=e^{-At}h(t)=F(0)+\int_0^te^{-A\tau}Bx(\tau)d\tau=h(0)+\int_0^te^{-A\tau}Bx(\tau)d\tau `$ (4) </center>

公式(4)两边同时乘以 $` e^{At} `$，可得<center>

$` h(t)=e^{At}h(0)+e^{At}\int_0^te^{-A\tau}Bx(\tau)d\tau `$ (5) </center>

考虑离散化形式，令 $` t=t_{k+1} `$<center>

$` h(t_{k+1})=e^{At_{k+1}}h(0)+e^{At_{k+1}}\int_0^{t_{k+1}}e^{-A\tau}Bx(\tau)d\tau `$ (6) </center>

设 $` t_{k+1}=t_k+\Delta `$, 则有

$` h(t_{k+1})=e^{A(t_k+\Delta)}h(0)+e^{A(t_k+\Delta)}[\int_0^{t_k}+\int_{t_k}^{t_{k+1}}] `$

$` h(t_{k+1})=e^{At_k}*e^{A\Delta}*h(0)+e^{At_k}*e^{A\Delta}*[\int_0^{t_k}+\int_{t_k}^{t_{k+1}}] `$

$` h(t_{k+1})=e^{A\Delta}[e^{At_k}*h(0)+e^{At_k}*\int_0^{t_k}]+e^{At_{k+1}}\int_{t_k}^{t_{k+1}}  `$ (7)

公式(7)中括号中的部分恰好等于 $` h(t_k) `$ ,化简可得<center>

$` h(t_{k+1})=e^{A\Delta}h(t_k)+e^{At_{k+1}}\int_{t_k}^{t_{k+1}}e^{-A\tau}Bx(\tau)d\tau `$ (8)</center>

离散场景中假设 $` \Delta `$ 足够小，我们认为 $` x(\tau) `$ 等价于 $` x(t_{k+1}) `$, 公式(8)可进一步化简：

$` h(t_{k+1})=e^{A\Delta}h(t_k)+e^{At_{k+1}}x(t_{k+1})\int_{t_k}^{t_{k+1}}e^{-A\tau}Bd\tau `$
$` h(t_{k+1})=e^{A\Delta}h(t_k)+e^{At_{k+1}}x(t_{k+1})[-\frac{1}{A}(e^{-At_{k+1}}-e^{-At_k})] `$
$` h(t_{k+1})=e^{A\Delta}h(t_k)-A^{-1}[1-e^{A\Delta}]Bx(t_{k+1}) `$ (9)

令公式(9)中 $` \bar{A}=e^{A\Delta}, \bar{B}=A^{-1}[e^{A\Delta}-1]B `$, 则有<center>

$` h(t_{k+1})=\bar{A}h(t_k)+\bar{B}x(t_{k+1}) `$</center>

以上得到论文中公式 (1a), (2a), (4)，对于输出，如果认为输出只与状态有关, 可以直接离散化为<center>

$` y(t_{k+1})=Ch(t_{k+1}) `$</center>

以上为论文中公式 (1b), (2b)

SSM 假定 $` h(t_0)=\bar{B}x(t_0) `$ , 那么当 $` k=0,1,2 `$ 时刻时，

$`
y(t_0)=Ch(t_0)=C\bar{B}x(t_0) \\
y(t_1)=Ch(t_1)=C(\bar{A}h(t_0)+\bar{B}x(t_1))=C\bar{A}\bar{B}x(t_0)+C\bar{B}x(t_1) \\
y(t_2)=Ch(t_2)=C(\bar{A}h(t_1)+\bar{B}x(t_2))=C\bar{A}(\bar{A}h(t_0)+\bar{B}x(t_1))+C\bar{B}x(t_2)=C\bar{A}^2\bar{B}x(t_0)+C\bar{A}\bar{B}x(t_1)+C\bar{B}x(t_2) `$

因此<center>

$` y_2 = [C\bar{A}^2\bar{B}, C\bar{A}\bar{B}, C\bar{B}][x_0, x_1, x_2]^T `$ (10)</center>

当推广到第 $` k `$ 次项 $` y_k `$ 时，令 $` K=[C\bar{A}^k\bar{B}, C\bar{A}^{k-1}\bar{B}, ...], x=[x_0, x_1, ...] `$ , 则<center>

$` y_k= K * x `$ (11)</center>

以上为论文公式 (3a), (3b)

