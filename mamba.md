# Mamba: Linear-Time Sequence Modeling with Selective State Spaces
视频讲解：https://www.bilibili.com/video/BV1JCrVYjEPN/?spm_id_from=333.337.search-card.all.click&vd_source=438c274850337b47a0be54ae5ef6d6ca 

论文：https://arxiv.org/pdf/2312.00752

## 公式推导
经典的线性时不变系统，输入为$x(t)$, 系统状态为$h(t)$, 输出为$y(t)$, 系统方程（原理可参考上述视频或现代控制理论）可以表示为: \

$\dot{h}(t)=Ah(t)+Bx(t)$ (1) \
$y(t)=Ch(t)+Dx(t)$ (2) \

为了能够适配LLM，token by token的输出，需要对上述连续系统进行离散化，得到通过$k$时刻的$h(t_k)$来表达$k+1$时刻的$h(t_{k+1})$的方案，即$h_{k+1}=f(h_k)$ 