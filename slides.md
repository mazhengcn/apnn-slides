---
theme: seriph
background: https://source.unsplash.com/collection/94734566/1920x1080
# class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
drawings:
  persist: false
transition: slide-left
title: Solving Transport Equations by DNNs
mdc: true
colorSchema: light
fonts:
  sans: inter
  mono: Fira Code
---

# Solve Multiscale Kinetic Equations with Deep Learning

<br>

Zheng Ma

Shanghai Jiao Tong University

Joint work with Shi Jin, Keke Wu, Tianai Zhang and Xiongbin Yan

<!-- <div class="abs-br m-6 flex gap-2">
  <img src="/affiliation.png" class="h-12">
</div> -->

<!-- https://sli.dev/ -->


---

# Background

Kinetic equations are important in many areas 

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>
Neutron Transport:

<br>

<div class="flex">
  <img src="/nte.png" class="h-60 rounded-lg b-1 b-b">
</div>

Nuclear fission, ...

</div>

<div>
Radiative Transfer:

<br>

<div class="flex">
  <img src="/rte.png" class="h-60 rounded-lg b-1 b-b">
</div>

Radiation Therapy, nuclear fusion, ...
</div>

</div>

Solving the neutron transport, radiative transfer and Boltzmann eqautions are the core part of these problems.

---

# Multiscale Kinetic Equations

<br>  

$$
\partial_t f +  v \cdot \nabla_x f = \frac{1}{\varepsilon} Q(f)
$$

<div class="top-25 right-8 absolute">
  <img src="/transport.png" class="mx-auto h-30 rounded-lg b-1 b-b">
</div>

- **$f$**: distribution function of particles at time $t$ in phase sapce $(x, v)$
- **$Q$**: collision operator
- **$\varepsilon$**: Knudsen number (ratio between mean free path and characteristic length) 

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>

###### Linear Transport Equation

$$
  \begin{equation*}
  \varepsilon \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( \frac{1}{2} \int_{-1}^{1} f \mathrm{d}{d} v' - f \right )
  \end{equation*}
$$

</div>

<div>

###### Boltzmann-BGK Equation

$$
  \begin{equation*}
  \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left (\textcolor{red}{M(U)}  - f \right )
  \end{equation*}
$$

</div>
</div>

Multiscale problem: $\varepsilon$ can be $O(1)$ (kinetic regime) or $\ll 1$ (limiting macroscopic regime)

---

# Numerical Chanllenge and Strategy

"High dimensional problem": we need to evolve the density function $f(x, v, t)$ in phase space

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>

**Knudsen number**
  - stability issues for small $\varepsilon$ (stiffness)
  - consistency of the scheme with limiting model as $\varepsilon \to 0$

**Collision Operator**
  - special discretization needed for the collision operator
  - high computational cost for more complicated collision
  - ray effect
</div>

<div>

**Probabilistic approach**

- Monte Carlo methods — easy implementation, efficient, half-order accuracy, random fluctuations

**Deterministic approach**

- discrete velocity/ordinate methods --- expensive, first or second order accuracy, maintain conservation.
- spectral methods --- relatively expensive, spectral accuracy, do not maintain conservation.

</div>

</div>

<div class="color-red bold text-center">

*Deep learning methods may become new approach!*

</div>

---

# Solve PDEs with Deep Learning

Key components and core ideas of solving PDEs by Deep Neural Networks

- **Architecture**: build a deep neural network (function class) as the trail function
  - approximate solution (PINNs)
  - approximate solution operator (DeepONet, FNO)
  - mapping from equations (as a computational graph) to solutions (PDEFormer)
- **Constraints (as Loss)**:
  - **Model**: PDE / physical information needed (e.g., PINNs, DeepRitz, Deep-Galerkin)
  - Data: pure supervised or as a priori information
  - IC (initial conditions) and BC (boundary conditions)
  - Other constraints: **conservation**, symmetry, etc.
- **Optimization**: minimize loss over the parameter space, usually SGD, Adam, LBFGS, etc.


---
layout: center
class: text-center
---

# What are models for multiscale kinetic equations?

PINNs? DeepRitz? ...

---

# Linear Transport Equation

We use a $1$-D linear transport equation as an example

$$
  \begin{equation*}
    \varepsilon \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( \frac{1}{2} \int_{-1}^{1} f \,\mathrm{d} v' - f \right )
  \end{equation*}
$$

<br>

#### Physics Informed Neural Networks (PINNs):

<br>

**Architecture** - approximate the solution by a deep neural network
$$
f_\theta^{\text{NN}}(t,x,v) \approx f(t,x,v)
$$

**Model** - take the *least square* of the linear transport equation residual (strong form) as loss 

$$
  \mathcal{R}_{\text{PINN}}^{\varepsilon} = 
  \frac{1}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}\times\mathcal{D}\times\Omega} \left| \varepsilon^2 \partial_t f^{\text{NN}}_{\theta} + \varepsilon {v} \cdot \nabla_x f^{\text{NN}}_{\theta} - \left ( \frac{1}{2} \int_{-1}^{1} f^{\text{NN}}_{\theta} \mathrm{d} v' - f^{\text{NN}}_{\theta} \right ) \right|^2 \, \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t}
$$

<!-- Physics Informed Neural Networks(PINNs)[^footnote1] -->

<!-- 
[^footnote1]: Maziar Raissi, Paris Perdikaris, and George E Karniadakis.  Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 2019 -->


<!-- background: '/carton8.png'  png。format -->

<!-- <style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style> -->

---

# Examples

<!-- <div class="top-0 right-10 absolute">

$$
\begin{equation*}
\varepsilon \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( \frac{1}{2} \int_{-1}^{1} f \mathrm{d}{d} v' - f \right )
\end{equation*}
$$

</div> -->

$$
\varepsilon \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( \rho(t,x) - f \right )
, \quad \rho(t,x) =  \frac{1}{2} \int_{-1}^{1} f \, \mathrm{d} v'
$$

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>

Kinetic regime: $\varepsilon=1$

$$
  f_0(x, v) = \frac{1 + \cos (4 \pi x)}{\sqrt{2\pi}}e^{-\frac{v^2}{2}}
$$

<!-- <img src="/ex1_pinns.png" width="400" height="300" class="h-40 float-left ml-5"/> -->
<img src="/ex1_pinns.png" class="h-50 mx-auto rounded-lg b-1 b-b"/>

</div>

<div>

Parabolic regime: $\varepsilon=10^{-8}$

$$
  f_0(x, v) = 0
$$

<br>

<img src="/ex2_pinns.png" class="h-50 mx-auto rounded-lg b-1 b-b" />

<div class="color-red text-center" >
  PINN fails!
</div>

</div>

</div>

---

# Failure of PINN Loss to Resolve Small Scales

<!-- <div v-click-hide> 

<img src="/loss_1e-8_pinns.png" width="300" height="500" class="h-40 mx-auto"/>

</div> -->

<img src="/loss_1e-8_pinns.png" class="h-40 mx-auto rounded-lg b-1 b-b" />

<div class="overflow-auto h-60">

$$
  \begin{aligned}
      \mathcal{R}^{\varepsilon}_{\text{PINN}} = & \frac{1}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}\times\mathcal{D}\times\Omega} \left| \varepsilon^2 \partial_t f^{\text{NN}}_{\theta} + \varepsilon {v} \cdot \nabla_x f^{\text{NN}}_{\theta} - \left ( \frac{1}{2} \int_{-1}^{1} f^{\text{NN}}_{\theta} \mathrm{d} v' - f^{\text{NN}}_{\theta} \right ) \right|^2 \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t} \\
                                                & +  \frac{\lambda_1}{|\mathcal{T} \times \partial \mathcal{D} \times \Omega|}  \int_{\mathcal{T}\times\partial\mathcal{D} \times\Omega} |\mathcal{B}f^{\text{NN}}_{\theta} - F_{\text{B}}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t} \\
                                                & +  \frac{\lambda_2}{|\mathcal{D}\times\Omega|} \int_{\mathcal{D} \times\Omega} |\mathcal{I}f^{\text{NN}}_{\theta} - f_{0}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}}.
  \end{aligned}
$$

We only need to focus on the first term and taking $\varepsilon \to 0$, this will lead to  

$$
\begin{equation*}
    \mathcal{R}_{\text{PINN}}^0 = \frac{1}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}} \int_{\mathcal{D}} \int_\Omega \left| - \left ( \frac{1}{2} \int_{-1}^{1} f^{\text{NN}}_{\theta} \mathrm{d} v' - f^{\text{NN}}_{\theta} \right ) \right|^2  \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t}
\end{equation*}
$$

which can be viewed as the PINN loss of the steady state

$$
f^{\text{NN}}_{\theta} = \frac{1}{2} \int_{-1}^{1} f^{\text{NN}}_{\theta} \mathrm{d} v'
$$

</div>

---

# Diffusion Limit: Micro-Macro Decomposition

<div class="overflow-auto h-100">

$$
\varepsilon \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( \rho(t,x) - f \right )
, \; \rho(t,x) = \left \langle f \right \rangle := \frac{1}{2} \int_{-1}^{1} f \mathrm{d} v' 
$$

Decompose $f$ into the equilibrium $\rho(t,x)$ and the non-equilibrium part $g(t,x,v)$:

$$
f(t,x,v) = \rho(t,x) + \varepsilon g(t,x,v). 
$$

The non-equilibrium part $g$ clearly satisfies $\left \langle g \right \rangle = 0$.

<!-- Subsititing $f = \rho + \varepsilon g$ into the linear transport equation yields

$$
\begin{equation}
  \varepsilon \partial_t \rho + \varepsilon^2 \partial_t g + v \cdot \nabla_x \rho + \varepsilon v \cdot \nabla_x g = - g. 
\end{equation}
$$

Integrating this equation with respect to $v$:

$$
\left \langle \varepsilon \partial_t \rho + \varepsilon^2 \partial_t g + v \cdot \nabla_x \rho + \varepsilon v \cdot \nabla_x g \right \rangle = 0，
$$

i.e.,

$$
 \varepsilon \partial_t \rho + \varepsilon^2 \left \langle  \partial_t g \right \rangle + \left \langle v  \right \rangle \cdot \nabla_x \rho + \varepsilon \left \langle  v \cdot \nabla_x g  \right \rangle  = 0,
$$

i.e.,

$$
 \varepsilon \partial_t \rho + \varepsilon^2 \partial_t \left \langle g \right \rangle +  \varepsilon \left \langle  v \cdot \nabla_x g  \right \rangle  = 0,
$$

i.e.,

$$
 \partial_t \rho  +  \left \langle  v \cdot \nabla_x g  \right \rangle  = 0.
$$

Define operator $\Pi(\cdot)(v): \left \langle \cdot \right \rangle$ and $I$ the identity operator, then apply the projection operator $I - \Pi$ to equation (1):

$$
\left ( I - \Pi \right ) \left ( \varepsilon \partial_t \rho + \varepsilon^2 \partial_t g + v \cdot \nabla_x \rho + \varepsilon v \cdot \nabla_x g \right ) =  - \left ( I - \Pi \right ) (g),
$$

i.e.,

$$
\varepsilon^2 \partial_t g + \varepsilon \left ( I - \Pi \right ) \left ( v \cdot \nabla_x g \right ) + v \cdot \nabla_x \rho  =  - g.
$$

Thus, one can get the micro-macro system for the linear trasport equation -->

Subsititing $f = \rho + \varepsilon g$ into the linear transport equation yields

$$
  \left\{
  \begin{aligned}
      & \partial_t \rho  +  \left \langle  v \cdot \nabla_x g  \right \rangle = 0, \\     
      \\
      & \varepsilon^2 \partial_t g + \varepsilon \left ( I - \Pi \right ) \left ( v \cdot \nabla_x g \right ) + v \cdot \nabla_x \rho  =  - g.
  \end{aligned}
  \right.
$$

where $I$ is the identity operator and $\Pi(\cdot)(v):=\left \langle \cdot \right \rangle$ is the projection operator.

<!-- Hilbert expansion:

$$
\begin{equation*}
    \begin{aligned}

        f(t,x,v) &= f_0(t,x,v) + \varepsilon f_0(t,x,v) + \varepsilon^2 f_1(t,x,v) + \cdots \\
        O(\frac{1}{\varepsilon}): & \quad  f_0 = \rho(t,x) \\
        O(1): & \quad f_1 = - v \cdot f_0 \\
        O(\varepsilon): & \quad f_2 = - (\partial_t f_0 + v \cdot \nabla_x f_1) \\
        & \vdots
    \end{aligned} 
\end{equation*}
$$ -->


</div>

---

# Micro-macro System

Classical numerical schemes based on this system are usually AP schemes

The micro-macro system for the linear trasport equation

$$
  \left\{
  \begin{aligned}
      & \partial_t \rho  +  \left \langle  v \cdot \nabla_x g  \right \rangle   = 0, \\     
        \\       
      & \varepsilon^2 \partial_t g + \varepsilon \left ( I - \Pi \right ) \left ( v \cdot \nabla_x g \right ) + v \cdot \nabla_x \rho  =  - g.
  \end{aligned}
  \right.
$$

Sending $\varepsilon \to 0$, the above system formally approaches 

$$
  \left\{
  \begin{aligned}
      & \partial_t \rho  +  \left \langle  v \cdot \nabla_x g  \right \rangle  = 0, \\          
      & - v \cdot \nabla_x \rho = g.
  \end{aligned}
  \right.
$$

Plugging the second equation into the first equation gives the diffusion equation

$$
 \rho_t - \frac{1}{3} \rho_{xx} = 0.
$$

---
layout: center
class: text-center
---

# What kind of loss is "good"?

Conservation, symmetry, parity, etc


---

# Asymptotic-Preserving Neural Networks

<img src="/apnns.png" class="h-70 mx-auto rounded-lg b-1 b-b" />

$\mathcal{F^{\varepsilon}}$ is the microscopic equation that depends on the small scale parameter $\varepsilon$ and $\mathcal{F}^{0}$ is its macroscopic limit as $\varepsilon \to 0$, which is independent of $\varepsilon$. The latent solution of $\mathcal{F^{\varepsilon}}$ is approximated by neural networks with its measure denoted by $\mathcal{R}(\mathcal{F^{\varepsilon}})$. The asymptotic limit of $\mathcal{R}(\mathcal{F^{\varepsilon}})$ as $\varepsilon \to 0$, if exists, is denoted by $\mathcal{R}(\mathcal{F}^{0})$. If $\mathcal{R}(\mathcal{F}^{0})$ is a good measure of $\mathcal{F}^{0}$, then it is called asymptotic-preserving (AP).

---

# Micro-macro System

Classical numerical schemes based on this system are usually AP schemes

The micro-macro system for the linear trasport equation

$$
  \left\{
  \begin{aligned}
      & \partial_t \rho  +  \left \langle  v \cdot \nabla_x g  \right \rangle   = 0, \\     
        \\       
      & \varepsilon^2 \partial_t g + \varepsilon \left ( I - \Pi \right ) \left ( v \cdot \nabla_x g \right ) + v \cdot \nabla_x \rho  =  - g.
  \end{aligned}
  \right.
$$

Sending $\varepsilon \to 0$, the above system formally approaches 

$$
  \left\{
  \begin{aligned}
      & \partial_t \rho  +  \left \langle  v \cdot \nabla_x g  \right \rangle  = 0, \\          
      & - v \cdot \nabla_x \rho = g.
  \end{aligned}
  \right.
$$

Plugging the second equation into the first equation gives the diffusion equation

$$
 \rho_t - \frac{1}{3} \rho_{xx} = 0.
$$

---

# APNN v1: based on Micro-macro decomposition

Re-design of loss

$$
    \begin{aligned}
        \mathcal{R}^{\varepsilon}_{\text{APNN}} = & \frac{1}{|\mathcal{T} \times \mathcal{D}|} \int_{\mathcal{T}} \int_{\mathcal{D}} | \partial_t \rho^{\text{NN}}_{\theta} + \nabla_x \cdot \left \langle   {v} g^{\text{NN}}_{\theta} \right \rangle|^2 \mathrm{d}{{x}}  \mathrm{d}{t}                                      \\
                                                  & + \frac{1}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}} \int_{\mathcal{D}} \int_\Omega | \varepsilon^2 \partial_t g^{\text{NN}}_{\theta}  + \varepsilon (I - \Pi)({v} \cdot \nabla_x g^{\text{NN}}_{\theta})                                      \\
                                                  & \quad  + {v} \cdot  \nabla_{{x}} \rho^{\text{NN}}_{\theta} +  g^{\text{NN}}_{\theta}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t}                                                                                                      \\
                                                  & +  \frac{\lambda_1}{\mathcal{T} \times\partial \mathcal{D} \times \Omega|}  \int_{\mathcal{T}} \int_{\partial \mathcal{D}} \int_\Omega |\mathcal{B}(\rho^{\text{NN}}_{\theta} + \varepsilon g^{\text{NN}}_{\theta}) - F_{\text{B}}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t} \\
                                                  & +  \frac{\lambda_2}{|\mathcal{D} \times \Omega|} \int_{\mathcal{D}} \int_\Omega |\mathcal{I}(\rho^{\text{NN}}_{\theta} + \varepsilon g^{\text{NN}}_{\theta}) - f_{0}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}}.
    \end{aligned}
$$

$$
    \begin{aligned}
        \mathcal{R}^{0}_{\text{APNN}} = & \frac{1}{|\mathcal{T} \times \mathcal{D}|} \int_{\mathcal{T}} \int_{\mathcal{D}} | \partial_t \rho^{\text{NN}}_{\theta} + \nabla_x \cdot \left \langle   {v} g^{\text{NN}}_{\theta} \right \rangle |^2 \mathrm{d}{{x}}  \mathrm{d}{t}                                      \\
                                                  & + \frac{1}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}} \int_{\mathcal{D}} \int_\Omega | {v} \cdot  \nabla_{{x}} \rho^{\text{NN}}_{\theta} +  g^{\text{NN}}_{\theta} |^2.
    \end{aligned}
$$

---

# APNN v1: based on Micro-macro decomposition


<img src="/APNNs.jpg" class="h-90 mx-auto rounded-lg b-1 b-b" />

Mass conservation mechanism $g^{\text{NN}}_{\theta} = \tilde{g}^{\text{NN}}_{\theta}-\left \langle \tilde{g}^{\text{NN}}_{\theta} \right\rangle$ is also important!


---

# Test examples

<div>


</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>

###### 
Ex 1: Periodic boundary condition ($\varepsilon=1$)

$$
\begin{equation*}
\begin{aligned}
  f(t, x_L, v) &= f(t, x_R, v), \\
  f_0(x, v) &= \frac{1 + \cos (4 \pi x)}{\sqrt{2\pi}}e^{-\frac{v^2}{2}}.
\end{aligned}
\end{equation*}
$$

<img src="/ex1_apnns.png" class="h-50 mx-auto rounded-lg b-1 b-b" />

</div><div>

###### 
Ex 2: Inflow boundary condition ($\varepsilon=10^{-8}$)

$$
\begin{equation*}
\begin{aligned}
  f(t, x_L, v) &= 1 \; \text{for} \; v > 0, \\
  f(t, x_R, v) &= 0 \; \text{for} \; v < 0, \\
  f_0(x, v) &= 0.
\end{aligned}
\end{equation*}
$$

<img src="/ex2_apnns.png" class="h-50 mx-auto rounded-lg b-1 b-b"/>


</div></div>

<div>

One can observed that APNN works for both $\varepsilon=1$ and $\varepsilon=10^{-8}$.

</div>

---

# Mass conservation mechanism

<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>

###### 
Ex 3: Inflow boundary condition ($\varepsilon=10^{-8}$)

For the constraint $\left \langle g \right \rangle = 0$, one way is to construct a novel neural network for $g$ such that it  exactly satisfies $\left \langle g \right \rangle = 0$. 

The other way is to treat it as a soft constraint with parameter $\lambda_3$, we use $\hat{g}_{\theta}^{\text{NN}}$ and modifies the loss as
$$
\begin{equation*}
  \mathcal{R}_{\text{APNN}} +  \frac{\lambda_3}{|\mathcal{T} \times \mathcal{D}|} \int_{\mathcal{T}} \int_{\mathcal{D}} | \left \langle  \hat{g}^{\text{NN}}_{\theta} \right \rangle - 0|^2 \mathrm{d}{{x}}  \mathrm{d}{t}.
\end{equation*}
$$



</div><div>

###### 
Plot of density $\rho$ at $t = 0.1$: APNNs with soft constraint (marker) vs. Ref (line). 

<img src="/ex2_1e-8_noexact.png" class="h-50 mx-auto rounded-lg b-1 b-b"/>

Mass conservation mechanism $g^{\text{NN}}_{\theta} = \tilde{g}^{\text{NN}}_{\theta} -  \left \langle \tilde{g}^{\text{NN}}_{\theta} \right \rangle$ is important!

</div></div>

<div>


</div>

---

# Even and odd parity method
Alternative method for constructing AP schemes in classical numerical methods

$$
\begin{equation*}
\varepsilon \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( \frac{1}{2} \int_{-1}^{1} f \mathrm{d} v' - f \right ), \; -1 \le v \le 1
\end{equation*}
$$

By splitting equation and define even- and odd-parities as

$$
\begin{equation*}
    \begin{aligned}
        r(t, x, v) & = \frac{1}{2}[f(t, x, v) + f(t, x, -v)], \; 0 \le v \le 1,  \\
        j(t, x, v) & = \frac{1}{2\varepsilon}[f(t, x, v) - f(t, x, -v)],  \; 0 \le v \le 1,
    \end{aligned}
\end{equation*}
$$

one can obtain

$$
\begin{equation*}
    \left\{
    \begin{aligned}
         & \partial_t r + v\partial_x j = \frac{1}{\varepsilon^2}(\rho - r), \\
         & \partial_t j + \frac{1}{\varepsilon^2} v \partial_x r = -\frac{1}{\varepsilon^2} j,
    \end{aligned}
    \right.
\end{equation*}
$$

where $\rho = \left \langle r \right \rangle := \int_0^1 r(t, x, v)  \mathrm{d} v$.


---

# Even and odd parity method
Not all traditional AP schemes work for neural networks!

$$
\begin{equation*}
    \left\{
    \begin{aligned}
         & \partial_t r + v\partial_x j = \frac{1}{\varepsilon^2}(\rho - r), \\
         & \partial_t j + \frac{1}{\varepsilon^2} v \partial_x r = -\frac{1}{\varepsilon^2} j.
    \end{aligned}
    \right.
\end{equation*}
$$

So far, we've got the even-odd system, however, it is not AP when applying neural networks for $r$ and $j$. 

To make it AP, we next introduce $\rho$ into this system as a bridge between $r$ and $j$.

By integrating over $v$, the first equation gives

$$
\begin{equation*}
    \partial_t \left \langle r \right \rangle  + \int_0^1  v\partial_x j  \mathrm{d} v = \frac{1}{\varepsilon^2} (\rho - \left \langle r \right \rangle),
\end{equation*}
$$

and due to $\rho = \left \langle r \right \rangle$, one can write as follows

$$
\begin{equation*}
    \partial_t \rho + \left \langle v\partial_x j \right \rangle = 0.
\end{equation*}
$$



---

# Even and odd parity method

<br>

The even-odd parity system for the linear trasport equation

$$
\begin{equation*}
    \left\{
    \begin{aligned}
         & \partial_t r + v\partial_x j = \frac{1}{\varepsilon^2}(\rho - r), \\
         & \partial_t j + \frac{1}{\varepsilon^2} v \partial_x r = -\frac{1}{\varepsilon^2} j, \\
         & \partial_t \rho + \left \langle v\partial_x j \right \rangle = 0. 
    \end{aligned}
    \right.
\end{equation*}
$$

Sending $\varepsilon \to 0$, the above system formally approaches 

$$
\begin{equation*}
    \left\{
    \begin{aligned}
        \rho & = r, \\          
        v \partial_x r&  = - j, \\
       \partial_t \rho + \left \langle v\partial_x j \right \rangle & = 0. 
    \end{aligned}
    \right.
\end{equation*}
$$

Plugging the first two equations into the third equation gives the diffusion equation $\rho_t - \frac{1}{3} \rho_{xx} = 0.$

---

# APNN v2: based on even and odd parity 

More general approach

**The equation of local conservation law $\partial_t \rho +  \left \langle v \partial_x j \right \rangle = 0$ is important in constructing the APNN loss. By coupling these equations of $r, j$ and $\rho$, one can obtain the loss for the diffusion limit equation.**

For solving the linear transport equation by deep neural networks, we need to use DNN to parametrize three functions $\rho(t, x), r(t, x, v)$ and $j(t, x, v)$. 

So here three networks are used:

$$
\begin{equation*}
    \rho^{\text{NN}}_{\theta}(t, x) := \exp \left( -\tilde{\rho}^{\text{NN}}_{\theta}(t, x)\right) \approx \rho(t, x),
\end{equation*}
$$

$$
\begin{equation*}
    r^{\text{NN}}_{\theta}(t, x, v) := \exp \left( -
    \frac{1}{2} (\tilde{r}^{\text{NN}}_{\theta}(t, x, v) + \tilde{r}^{\text{NN}}_{\theta}(t, x, -v) ) \right) \approx r(t, x, v),
\end{equation*}
$$

$$
\begin{equation*}
    j^{\text{NN}}_{\theta}(t, x, v) :=
    \tilde{j}^{\text{NN}}_{\theta}(t, x, v) - \tilde{j}^{\text{NN}}_{\theta}(t, x, -v)  \approx j(t, x, v).
\end{equation*}
$$

---

# APNN v2: based on even- and odd- parity 

<!-- Then we propose the least square of the residual of the even-odd system as the APNN loss -->

$$
\begin{equation*}
    \begin{aligned}
        \mathcal{R}^{\varepsilon}_{\text{APNN}} = & \frac{\lambda_1}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}} \int_{\mathcal{D}} \int_{\Omega} | \varepsilon^2 \partial_t r^{\text{NN}}_{\theta} + \varepsilon^2 v\partial_x j^{\text{NN}}_{\theta} - (\rho^{\text{NN}}_{\theta} - r^{\text{NN}}_{\theta}) |^2 \mathrm{d}{{v}} \mathrm{d}{{x}}  \mathrm{d}{t} \\
                                                  & + \frac{\lambda_2}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}} \int_{\mathcal{D}} \int_{\Omega} |\varepsilon^2 \partial_t j^{\text{NN}}_{\theta} + v \partial_x r^{\text{NN}}_{\theta} - (-j^{\text{NN}}_{\theta}) |^2 \mathrm{d}{{v}} \mathrm{d}{{x}}  \mathrm{d}{t}                                        \\
                                                  & + \frac{\lambda_3}{|\mathcal{T} \times \mathcal{D}|} \int_{\mathcal{T}} \int_{\mathcal{D}} | \partial_t \rho^{\text{NN}}_{\theta} +  \left \langle v\partial_x j^{\text{NN}}_{\theta} \right \rangle |^2   \mathrm{d}{{x}}  \mathrm{d}{t}
        \\
                                                  & + \frac{\lambda_4}{|\mathcal{T} \times \mathcal{D}|} \int_{\mathcal{T}} \int_{\mathcal{D}} |\rho^{\text{NN}}_{\theta} -  \left \langle r^{\text{NN}}_{\theta} \right \rangle |^2   \mathrm{d}{{x}}  \mathrm{d}{t}
        \\
                                                  & +  \frac{\lambda_5}{|\mathcal{D}|} \int_{\mathcal{D}} |\rho^{\text{NN}}_{\theta}(0, x) - \left \langle f_{0} \right \rangle |^2 \mathrm{d}{\bm{x}}
 +  \frac{\lambda_6}{|\mathcal{D} \times \Omega|} \int_{\mathcal{D}} \int_\Omega |\mathcal{I}(r^{\text{NN}}_{\theta} + \varepsilon j^{\text{NN}}_{\theta}) - f_{0}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}}
        \\
                                                  & +  \frac{\lambda_7}{|\mathcal{T} \times\partial \mathcal{D} \times \Omega|}  \int_{\mathcal{T}} \int_{\partial \mathcal{D}} \int_\Omega |\mathcal{B}(r^{\text{NN}}_{\theta} + \varepsilon j^{\text{NN}}_{\theta}) - F_{\text{B}}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t}.
    \end{aligned}
\end{equation*}
$$

Notice that the constraint $\textcolor{red}{\rho = \left \langle r \right \rangle}$ is also added into the APNN loss.

---

# Test examples - Case I

Inflow boundary condition ($\varepsilon=10^{-3}$)

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>

$$
\begin{equation*}
\begin{aligned}
  f(t, x_L, v) &= 1 \; \text{for} \; v > 0, \\
  f(t, x_R, v) &= 0 \; \text{for} \; v < 0, \\
  \rho_0(x) &= 0, \\
  f_0(x, v) &= 0.
\end{aligned}
\end{equation*}
$$

Note that the function $f$ has a jump at $t = 0$ since $F_L(v) = 1, F_R(v) = 0$ but $f_0(x, v) = 0$. 

<u>For better numerical performance,  $\rho^{\text{NN}}_{\theta}$ can be further constructed to automatically satisfies initial condition:</u>

$\rho^{\text{NN}}_{\theta}(t, x) := t \cdot \exp \left( -\tilde{\rho}^{\text{NN}}_{\theta}(t, x)\right) \approx \rho(t, x)$


</div>

<div class="top-1">

Plot of density $\rho$ at $t = 0, 0.05, 0.1$: APNNs (marker) vs. Ref (line). 

<img src="/dirichlet10_sol.png" class="h-40 mx-auto rounded-lg b-1 b-b"/>

<!-- FCNet with units $[2, 128, 128, 128, 128, 1]$ for $\rho$ and $[3, 256, 256, 256, 256, 1]$ both for $r$ and $j$. Batch size is $512$ in domain, $1024 \times 2$ on boundary and $512$ on initial, the number of quadrature points is $30$. $\lambda_1 = \lambda_2 = \lambda_3 = \lambda_4 = \lambda_6 = 1, \lambda_7 = 10$. -->

Relative $\ell^2$ error of APNNs is $9.87 \times 10^{-3}$. 

</div></div>

---

<div>

The numerical performance of enforcement of initial condition and the soft constraint $\rho = \left \langle r \right \rangle$ are discussed as follows.

Plot of density $\rho$ at $t = 0, 0.05, 0.1$: APNNs (marker) vs. Ref (line).

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>

##### &emsp; &emsp; Figure 1:  without the enforcement of initial condition

<img src="/dirichlet10_sol_2.png" class="h-60 mx-auto rounded-lg b-1 b-b" />

Due to the poor approximate of initial and boundary layer effect, it gives wrong solution at time $t = 0, 0.05, 0.1$.

</div><div>

##### &emsp; &emsp; &emsp; &emsp; Figure 2: without the constraint $\rho = \left \langle r \right \rangle$ 

<img src="/dirichlet10_sol_1.png" class="h-60 mx-auto rounded-lg b-1 b-b" />

The solutions are also wrong at time $t = 0, 0.05, 0.1$, therefore, we consider this constraint into our APNN loss.

</div></div>


---

# Test examples - Case II

UQ problems with inflow condition ($\varepsilon=10^{-5}$)


$$
\begin{equation*}
    \varepsilon \partial_t f + v\partial_x f = \frac{\sigma_S(\bm z)}{\varepsilon}\left ( \frac{1}{2} \int_{-1}^1 f \, dv' - f \right ), \quad x_L < x < x_R, \quad -1 \leq v \leq 1,
\end{equation*}
$$
$$
\begin{equation*}
    \sigma_S(\bm z) = 1 + \frac{1}{10} \prod_{i=1}^{20} \sin(\pi z^i), \;\bm{z} = (z^1, z^2, \cdots, z^{20}) \sim \mathcal{U}([-1, 1]^{20}),
\end{equation*}
$$


<div>

<img src="/uq_1e-5.png" class="h-60 mx-auto rounded-lg b-1 b-b"/>
</div>

<!-- ---

# APNN <MarkerCore />

- Linear Transport Equation: 

&emsp; &emsp;
boundary and initial condition, 
conservation, 
symmetry, 
parity, 
...

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### Micro-macro decomposition

$$
\begin{equation*}
    \left\{
    \begin{aligned}
        \partial_t \rho  +  \left \langle  v \cdot \nabla_x g  \right \rangle  & = 0, \\     
         \\       
        \varepsilon^2 \partial_t g + \varepsilon \left ( I - \Pi \right ) \left ( v \cdot \nabla_x g \right ) + v \cdot \nabla_x \rho  & =  - g.
    \end{aligned}
    \right.
\end{equation*}
$$

</div><div v-click>

###### Odd-even Parity

$$
\begin{equation*}
    \left\{
    \begin{aligned}
          \varepsilon^2 \partial_t r + \varepsilon^2 v\partial_x j &= \rho - r, \\
          \varepsilon^2 \partial_t j + v \partial_x r &= - j,                   \\
          \partial_t \rho +  \left \langle v \partial_x j \right \rangle &= 0.

    \end{aligned}
    \right.
\end{equation*}
$$

</div></div> -->

---

# Bhatnagar-Gross-Krook (BGK) equation
<br>

 $$
 \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( M(U) - f \right ),  \quad v \in \mathbb{R},
 $$

where $M(U)$ denotes the local Maxwellian distribution function given by
$$
\begin{equation*}
M(U) = \frac{\rho}{(2 \pi T)^{\frac{1}{2}}} \exp \left ( - \frac{|v - u|^2}{2 T}\right ),
\end{equation*}
$$

and $\rho(t, x), u(t, x)$ and $T(t, x)$ are the density, macroscopic velocity and temperature which are related with the moments of $f$:

$$
\begin{equation*}
U :=
 \begin{pmatrix}
 \rho \\
 \rho u \\
\frac{1}{2} \rho |u|^2 + \frac{1}{2} \rho T
\end{pmatrix} = \int_{\mathbb{R}} m  f \mathrm{d} v, \; m =  {\left ( 1, v, \frac{1}{2} v^2\right )}^T.
\end{equation*}
$$

<font color=red> Notice that the Boltzmann-BGK equation is an integro-differential equation with its nonlinear and non-local collision operator. </font>


---

# Density, macroscopic velocity and temperature

<br>

- density 
  $$
  \rho = 
  \int {f}(t,x,v) \mathrm{d} v.
  $$

- velocity 
  $$
  u = 
  \int v \cdot \frac{f(t,x,v)}{ \int {f}(t,x,v) \mathrm{d} v} \mathrm{d} v = \frac{1}{\rho} \int v f \mathrm{d} v. 
  $$

- thermodynamics energy
   $$
  \int \frac{1}{2} (v - u)^2 \cdot \frac{f(t,x,v)}{ \int {f}(t,x,v) \mathrm{d} v} \mathrm{d} v = \frac{1}{\rho} \left [ \frac{1}{2} \int v^2 f \mathrm{d} v -  \frac{1}{2} \rho u^2 \right ],
  $$ 
  $$
  \Rightarrow \frac{1}{2} \rho T  = \frac{1}{2}\int_{\mathbb{R}} v^2  f \mathrm{d} v - \frac{1}{2} \rho |u|^2 \; (\text{The ideal gas law}).
  $$

  
---

# Local conservation laws

<br>

$$
\partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( M(U) - f \right ),  \quad v \in \mathbb{R},
$$

$$
\begin{equation*}
U :=
 \begin{pmatrix}
 \rho \\
 \rho u \\
\frac{1}{2} \rho |u|^2 + \frac{1}{2} \rho T
\end{pmatrix} = \int_{\mathbb{R}} m  f \mathrm{d} v, \; m = {\left ( 1, v, \frac{1}{2} v^2\right )}^T.
\end{equation*}
$$

Due to the properties of conserving mass, momentum and energy of collision operator, one can multiply the BGK equation by $m(v)$ and integrate with respect to $v$ to obtain the equations of local conservation laws

$$
\begin{equation*}
\partial_t \left \langle m f \right \rangle + \nabla_x \cdot \left \langle v m f \right \rangle = 0, \; \text{where} \; \left \langle g \right \rangle =  \int_{\mathbb{R}} g(v) \mathrm{d} v,
\end{equation*}
$$

i.e.,

$$
\begin{equation*}
\partial_t 
\begin{pmatrix}
 \rho \\
 \rho u \\
\frac{1}{2} \rho |u|^2 + \frac{1}{2} \rho T
\end{pmatrix}
 + \nabla_x \cdot \left \langle v m f \right \rangle = 0.
\end{equation*}
$$


---

# APNN System 

<br>

The systems of Boltzmann-BGK model

$$
\begin{equation*}
    \left\{
    \begin{aligned}
         & \varepsilon \left ( \partial_t f + v \partial_x f \right ) = M(U) - f, \\
         &  \partial_t 
            \begin{pmatrix}
             \rho \\
             \rho u \\
            \frac{1}{2} \rho |u|^2 + \frac{1}{2} \rho T
            \end{pmatrix}
             + \nabla_x \cdot \left \langle v m f \right \rangle = 0, \\
        & \begin{pmatrix}
             \rho \\
             \rho u \\
            \frac{1}{2} \rho |u|^2 + \frac{1}{2} \rho T
            \end{pmatrix} = \int_{\mathbb{R}} m  f \mathrm{d} v.
    \end{aligned}
    \right.
\end{equation*}
$$

Sending $\varepsilon \to 0$, one have $f = M(U)$ and the local conservation laws becomes compressible Euler equation:
    
$$
\begin{equation*}
  \partial_t 
            \begin{pmatrix}
             \rho \\
             \rho u \\
            \frac{1}{2} \rho |u|^2 + \frac{1}{2} \rho T
            \end{pmatrix}
             + \nabla_x \cdot \left \langle v m M \right \rangle = 0.
\end{equation*}
$$

---

# Boundary and initial conditions
<br>

The boundary conditions of $\rho, u, T$ are set as constants:

$$
\begin{equation*}
\begin{aligned}
& \rho(t, x_L) = \rho_L, \rho(t, x_R) = \rho_R, \\
& u(t, x_L) = u_L = 0, u(t, x_R) = u_R = 0, \\
& T(t, x_L) = T_L, T(t, x_R) = T_R.
\end{aligned}
\end{equation*}
$$


The initial condition of $f$ is computed by the initial functions $\rho_0(x), u_0(x), T_0(x)$:

$$
\begin{equation*}
f(0, x, v) = \frac{\rho_0}{(2 \pi T_0)^{\frac{1}{2}}} \exp \left ( - \frac{|v - u_0|^2}{2 T_0} \right ) := f_0 (x, v).
\end{equation*}
$$

Here, time $t \in \mathcal{T} := [0, T]$, space point $x \in \mathcal{D} := [x_L, x_R]$ and
 we restrict the range of velocity to a bounded symmetrical domain $\Omega = [-V, V]$ with $V = 10$ since this assumption might be realistic in many studies.
 
 
---

# APNN v2 for Boltzmann-BGK equation 
<br>

$$
\begin{equation*}
    \left\{
    \begin{aligned}
         & \varepsilon \left ( \partial_t f + v \partial_x f \right ) = M(U) - f, \\
         &  \partial_t 
            \begin{pmatrix}
             \rho \\
             \rho u \\
            \frac{1}{2} \rho |u|^2 + \frac{1}{2} \rho T
            \end{pmatrix}
             + \nabla_x \cdot \left \langle v m f \right \rangle = 0, \\
        & \begin{pmatrix}
             \rho \\
             \rho u \\
            \frac{1}{2} \rho |u|^2 + \frac{1}{2} \rho T
            \end{pmatrix} = \int_{\mathbb{R}} m  f \mathrm{d} v, \\
        & \rho(t, x_L) = \rho_L, \;  \rho(t, x_R) = \rho_R, \\
        & u(t, x_L) = u_L, \; u(t, x_R) = u_R, \\
        & T(t, x_L) = T_L, \; T(t, x_R) = T_R, \\
        & f(0, x, v) = f_0(x, v), \\
        & \rho(0, x) = \rho_0(x), u(0, x) = u_0(x),  T(0, x) = T_0(x).
    \end{aligned}
    \right.
\end{equation*}
$$

---

# APNN v2 for Boltzmann-BGK equation 
<br>

First we parametrize four functions $f(t, x, v), \rho(t, x), u(t, x)$ and $T(t, x)$ with four networks. The time and velocity variable $t, v$ are normalized into $[0, 1]$ and $[-1, 1]$ with scaling $\bar{t} = t / T, \bar{v} = v / V$ and we construct four DNNs as follows:

$$
\begin{equation*}
f^{\text{NN}}_{\theta}(t, x, v) := \ln \left(1 + \exp (\tilde{f}^{\text{NN}}_{\theta}(\bar{t}, x, \bar{v})) \right) > 0, 
\end{equation*}
$$

$$
\begin{equation*}
\begin{aligned}
& \rho ^{\text{NN}}_{\theta}(t, x) := \exp \left ( (x-x_L)(x_R - x) \cdot  \tilde{\rho}^{\text{NN}}_{\theta}(\bar{t}, x) + \log(\rho_L) \frac{x_R - x}{x_R - x_L} + \log(\rho_R) \frac{x - x_L}{x_R - x_L}\right ) > 0, \\
& u ^{\text{NN}}_{\theta}(t, x) := \sqrt{(x-x_L)(x_R - x)} \cdot \tilde{u}^{\text{NN}}_{\theta}(\bar{t}, x), \\
& T ^{\text{NN}}_{\theta}(t, x) := \exp \left ( (x-x_L)(x_R - x) \cdot  \tilde{T}^{\text{NN}}_{\theta}(\bar{t}, x) + \log(T_L) \frac{x_R - x}{x_R - x_L} + \log(T_R) \frac{x - x_L}{x_R - x_L}\right ) > 0,
\end{aligned}
\end{equation*}
$$

which $\rho ^{\text{NN}}_{\theta}, u ^{\text{NN}}_{\theta}, T ^{\text{NN}}_{\theta}$ automatically satisfy the boundary conditions. In this problem, to keep $f$ positive, $\ln (1 + \exp(\cdot))$ is applied for constructing $f^{\text{NN}}_{\theta}$. The benefit of this construction is that the values of $f^{\text{NN}}_{\theta}$ and $\tilde{f}^{\text{NN}}_{\theta}$ are at the same level. 

---

# APNN loss for Boltzmann-BGK

<div class="overflow-auto h-100">

$$
\begin{equation*}
    \begin{aligned}
        \mathcal{R}^{\varepsilon}_{\text{APNN, BGK}} = & \frac{\lambda_1}{N_1^{(1)}} \sum_{i=1}^{N_1^{(1)}}
        | \varepsilon (\partial_t f^{\text{NN}}_{\theta}(t_i,x_i,v_i) +  v\nabla_x f^{\text{NN}}_{\theta}(t_i,x_i,v_i)) - \left ( M(U^{\text{NN}}_{\theta}) - f^{\text{NN}}_{\theta} \right )(t_i,x_i,v_i) |^2  \\
                                        & + \frac{\lambda_2}{N_1^{(2)}} \sum_{i=1}^{N_1^{(2)}} |\partial_t \rho^{\text{NN}}_{\theta}(t_i,x_i) + \nabla_x \left \langle v f^{\text{NN}}_{\theta} \right \rangle (t_i,x_i)|^2  \\
                                        & + \frac{\lambda_3}{N_1^{(3)}} \sum_{i=1}^{N_1^{(3)}} |\partial_t (\rho^{\text{NN}}_{\theta}(t_i,x_i) u^{\text{NN}}_{\theta}(t_i,x_i)) + \nabla_x \left \langle v^2 f^{\text{NN}}_{\theta} \right \rangle (t_i,x_i) |^2 \\
                                        & + \frac{\lambda_4}{N_1^{(4)}} \sum_{i=1}^{N_1^{(4)}} |\partial_t \left (\frac{1}{2} \rho^{\text{NN}}_{\theta}(t_i,x_i) (u^{\text{NN}}_{\theta}(t_i,x_i))^2 + \frac{1}{2} \rho^{\text{NN}}_{\theta}(t_i,x_i) T^{\text{NN}}_{\theta}(t_i,x_i)\right ) + \\
                                                & \quad \quad \quad \quad \quad \quad \nabla_x \left \langle \frac{1}{2} v^3 f^{\text{NN}}_{\theta} \right \rangle (t_i,x_i) |^2 \\
                                        & + \frac{\lambda_5}{N_2^{(1)}} \sum_{i=1}^{N_2^{(1)}} |\rho^{\text{NN}}_{\theta}(t_i,x_i) - \left \langle f^{\text{NN}}_{\theta} \right \rangle (t_i,x_i)|^2 \\
                                        & + \frac{\lambda_6}{N_2^{(2)}} \sum_{i=1}^{N_2^{(2)}} |\rho^{\text{NN}}_{\theta}(t_i,x_i) u^{\text{NN}}_{\theta}(t_i,x_i) - \left \langle v f^{\text{NN}}_{\theta} \right \rangle (t_i,x_i) |^2  \\
                                        & + \frac{\lambda_7}{N_2^{(3)}} \sum_{i=1}^{N_2^{(3)}} |\partial_t \left (\frac{1}{2} \rho^{\text{NN}}_{\theta} (u^{\text{NN}}_{\theta})^2 + \frac{1}{2} \rho^{\text{NN}}_{\theta} T^{\text{NN}}_{\theta}\right )(t_i,x_i) - \left \langle \frac{1}{2} v^2 f^{\text{NN}}_{\theta} \right \rangle (t_i,x_i) |^2  \\
                                        & + \frac{\lambda_8}{N_3^{(1)}} \sum_{i=1}^{N_3^{(1)}} |\rho^{\text{NN}}_{\theta}(t_i, x_L) - \rho_L|^2 + |\rho^{\text{NN}}_{\theta}(t_i, x_R) - \rho_R|^2 \\
                                        & + \frac{\lambda_9}{N_3^{(2)}} \sum_{i=1}^{N_3^{(2)}} |u^{\text{NN}}_{\theta}(t_i, x_L) - u_L|^2 + |u^{\text{NN}}_{\theta}(t_i, x_R) - u_R|^2 \\
                                        & + \frac{\lambda_{10}}{N_3^{(3)}} \sum_{i=1}^{N_3^{(3)}} |T^{\text{NN}}_{\theta}(t_i, x_L) - T_L|^2 + |T^{\text{NN}}_{\theta}(t_i, x_R) - T_R|^2 \\
                                        & + \frac{\lambda_{11}}{N_4^{(1)}} \sum_{i=1}^{N_4^{(1)}} |f^{\text{NN}}_{\theta}(0, x_i, v_i) - f_0(x_i, v_i)|^2  \\
                                        & + \frac{\lambda_{12}}{N_4^{(2)}} \sum_{i=1}^{N_4^{(2)}} |\rho^{\text{NN}}_{\theta}(0, x_i) - \rho_0(x_i)|^2 \\
                                        & + \frac{\lambda_{13}}{N_4^{(3)}} \sum_{i=1}^{N_4^{(3)}} |u^{\text{NN}}_{\theta}(0, x_i) - u_0(x_i)|^2  \\
                                        & + \frac{\lambda_{14}}{N_4^{(4)}} \sum_{i=1}^{N_4^{(4)}} |T^{\text{NN}}_{\theta}(0, x_i) - T_0(x_i)|^2 .
    \end{aligned}
\end{equation*}
$$

</div>

---

# Test example

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>

###### 
Initial conditions ($\varepsilon=10^{-3}$)

$$
\begin{equation*}
\begin{aligned}
& \rho_0(x) = 1.5 + (0.625 - 1.5) \cdot \frac{\sin(\pi x) + 1}{2}, \\
& u_0(x) =0, \\
& T_0(x) = 1.5 + (0.75 - 1.5) \cdot \frac{\sin(\pi x) + 1}{2},
\end{aligned}
\end{equation*}
$$

$$
\begin{equation*}
f_0 (x, v) = \frac{\rho_0}{(2 \pi T_0)^{\frac{1}{2}}} \exp \left ( - \frac{|v - u_0|^2}{2 T_0} \right ).
\end{equation*}
$$

The reference solutions are the density, momentum and energy:

$$\rho, \rho u, \frac{1}{2} \rho u^2 + \frac{1}{2} \rho T.$$

</div><div>

###### 
Setting: ResNet with units $[3, 128, 128, 128, 128, 128, 128, 1]$ for $f$ and $[2, 64, 64, 64, 64, 64, 64, 1]$ both for $\rho, u$ and $T$. Batch size is $512$ in domain, and $256$ on initial, the number of quadrature points is $64$. 

$\lambda_9 = 0.1, \lambda_{11} = \lambda_{13} = \lambda_{14} = 10$ 
and others are set to be $1$. 

For $t = 0:$ mean square error of density, momentum and energy are $7.16\text{e-8}, 4.50\text{e-6}, 8.13\text{e-7}$. 

For $t = 0.1:$ relative $l^2$ error of density, momentum and energy are $5.43\text{e-3}, 6.35\text{e-3}, 4.47\text{e-2}$.

</div></div>


---


<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0$

<img src="/ex1_sol_f_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>

<img src="/ex1_sol_macro_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>



</div><div>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0.1$

<img src="/ex1_sol_f_t1.png" width="400" height="300" class="h-45 float-right ml-5"/>

<img src="/ex1_sol_macro_t1.png" width="400" height="300" class="h-45 float-right ml-5"/>


</div></div>

---

# APNNs for GRTE

APNNs for Multiscale Gray Radiative Transfer Equations

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>

Original system

$$
  \left \{
    \begin{aligned}
      & \frac{\varepsilon^2}{c} \frac{\partial I}{\partial t}+\varepsilon \Omega \cdot \nabla I =\sigma \left(\frac{1}{4 \pi}acT^4-I\right),    \\
      & \varepsilon^2 C_v\frac{\partial T}{\partial t} = \sigma \left(\int_{S^2} I \mathrm{d}\Omega - a c T^4\right), \\
      & \mathcal{B} I = 0, \; \\
      & I(t=0, x, \Omega) = I_0(x, \Omega), \\
      & T(t=0, x) = T_0(x),
    \end{aligned}
  \right.
$$

</div>

<div>

APNN based on even-odd decomposition

$$
\left\{
    \begin{aligned}
         & \frac{\varepsilon^2}{c} \partial_t r + \frac{\varepsilon^2}{\sqrt{\sigma_0}} \cdot \mu \partial_x j = \sigma \left ( \frac{1}{2} a c T^4 - r \right ), \\
         & \frac{\varepsilon^2}{c \sqrt{\sigma_0}} \partial_t j + \mu \partial_x r = -\frac{\sigma}{\sqrt{\sigma_0}} j, \\
         &  \frac{1}{c} \partial_t \rho  + \frac{1}{\sqrt{\sigma_0}} \cdot \left \langle  \mu \partial_x j \right \rangle = - \frac{1}{2} C_v\frac{\partial T}{\partial t}, \\
         & \varepsilon^2 C_v\frac{\partial T}{\partial t} = \sigma \left(2 \rho -acT^4 \right), \\
         & \rho = \left \langle r \right \rangle.
    \end{aligned}
    \right.
$$
</div>
</div>

---
layout: image-right
image: /apnn_grte.png
---

# APNNs for GRTE
Numerics and schema

<br>
<!-- <div class="grid grid-cols-2 gap-x-4 mt-4"> -->

<div>
  <img src="/grte.png" class="mx-auto rounded-lg b-1 b-b">
</div>

---

# APNNs for Vlasov-Poisson-Fokker-Planck system

<br>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div>

VPFP System

$$
\begin{aligned}
  & \partial_t f+v\cdot\nabla_x f-\frac{1}{\varepsilon}\nabla_x \phi \cdot \nabla_v f =\frac{1}{\varepsilon} \nabla_{v} \cdot\left[v f+\nabla_{v} f\right], \\
  & -\nabla_x \phi(t, x) =\rho(t, x)-h(x),
\end{aligned}
$$
where 
$$
\rho(t, x)=\int_{\mathbb{R}^N} f(t, x, v) \mathrm{d} v
$$
</div>

<div>
  <img src="/vpfp.png" class="mx-auto rounded-lg b-1 b-1">
</div>
</div>

---

# APCONs
Asymptotic-Preserving Convolutional Deep Operator Networks

<div class="grid grid-cols-2 gap-x-4 mt-4">
<div>

$$
  \left\{
      \begin{aligned}
        & \varepsilon \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon}\mathcal{L} f, \; (t, x, v) \in \mathcal{T} \times \mathcal{D} \times {\Omega}, \\
        & \mathcal{B} f(t, x, v) = 0,      \\
        & f(0, x, v) = f_0(x, v).
  \end{aligned}
  \right.
$$

<br>

Our goal is to learn the mapping $\mathcal{G}$ from 

$$
\mathcal{G}: f_0(x, v) \to f(t, x, v)
$$

</div>
<div class="text-center">
<br>
<img src="/cons.png" class="h-70 mx-auto rounded-lg b-1 b-b">

Convolutional DeepONets

</div>
</div>

---

# Schematic of APCONs

<div class="grid grid-cols-2 gap-x-4 mt-4">
<div>

APCON based on Micro-macro decomposition

<img src="/apcon_mm.png" class="h-70 mx-auto rounded-lg b-1 b-b">

</div>

<div class="text-center">

APCON based on even-odd decomposition

<img src="/apcon_eo.png" class="h-70 mx-auto rounded-lg b-1 b-b">

</div>
</div>

---

# APCONs Performance

<div class="grid grid-cols-2 gap-x-4 mt-4">
<div>

Accuracy and parameters 

<img src="/apcon_table.png" class="h-70 mx-auto rounded-lg b-1 b-b">

</div>

<div class="text-center">

APCON efficiency

<img src="/apcon_e.png" class="h-70 mx-auto rounded-lg b-1 b-b">

</div>
</div>

---

# Conclusions

We propose several Asymptotic-Preserving Neural Networks for solving the multiscale time-dependent kinetic problems:

- Linear transport

  - APNN v1 based on micro-macro decomposition
  - APNN v2 based on odd-even parity method
  - APCONs: based on convolutional DeepONets and APNN

- GRTE and VFPF

  - APNN v1 and APNN v2

Boltzmann-BGK equation:

- APNN v2 based on local conservation laws

---

# References

- Lulu Zhang, Tao Luo, Yaoyu Zhang, Weinan E, Zhi-Qin John, and Zheng Ma. Mod-net: A Machine Learning Approach via Model-Operator-Data Network for Solving PDEs. Communication in Computational Physics, 32(2):299–335, 2022.
- Shi Jin, Zheng Ma and Keke Wu, Asymptotic-Preserving Neural Networks for Multiscale Time-Dependent Linear Transport Equations. Journal of Scientific Computing 94, 57 (2023)
- Shi Jin, Zheng Ma and Keke Wu, Asymptotic-Preserving Neural Networks for Multiscale Kinetic Equations, preprint, 2023
- Shi Jin, Zheng Ma and Tian-ai Zhang, Asymptotic-preserving neural networks for multiscale Vlasov-Poisson-Fokker-Planck system in the high-field, preprint, 2023
- Keke Wu, Xiong-bin Yan, Shi Jin and Zheng Ma, Capturing the Diffusive Behavior of the Multiscale Linear Transport Equations by Asymptotic-Preserving Convolutional DeepONets, preprint, 2023

---
layout: end
---

# Thanks!