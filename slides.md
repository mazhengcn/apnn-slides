---
theme: seriph
# background: 'https://images.unsplash.com/photo-1558021212-51b6ecfa0db9?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1766&q=80'
layout: cover
class: text-center
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
download: true
preload: false
title: Asymptotic-Preserving Neural Networks
---

# Asymptotic-Preserving Neural Networks for Solving Multiscale Kinetic Equations

<br>

Zheng Ma

Shanghai Jiao Tong University

<br>
<br>

Joint work with Shi Jin and Keke Wu

<!-- <div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/wukekever" target="_blank" alt="GitHub"
    class="text-xl icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div> -->

<!-- https://sli.dev/ -->

---

# Outline

<br>
<br>

#### 1. Motivation

<br>
<br>

#### 2. Method
  
  - APNN v1
  
  - APNN v2

<br>
<br>  

<!-- #### 3. Results

<br>
<br>   -->

<!-- #### 4. Conclusions -->

#### 3. Conclusions


---

# Multiscale kinetic equation <MarkerCore />

<br>  

$$
\partial_t f +  v \cdot \nabla_x f = \frac{1}{\varepsilon} Q(f, f).
$$


<div class="top-5 right-8 absolute">

<img src="/transport.png" alt="transport" title="transport" width="300" height="200" class="mx-auto">

</div>

- $f:$ distribution function of particles at time $t$, space position $x$ and traveling in direction $v$
- $Q:$ collision operator
- $\varepsilon > 0:$ Knudsen number 

<br>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### Linear Transport Equation

$$
\begin{equation*}
\varepsilon \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( \frac{1}{2} \int_{-1}^{1} f \mathrm{d}{d} v' - f \right )
\end{equation*}
$$

<!-- (linear and nonlocal operator) -->

</div><div v-click>

###### Boltzmann-BGK Equation

$$
\begin{equation*}
\partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left (\textcolor{red}{M(U)}  - f \right )
\end{equation*}
$$

<!-- (nonlinear and nonlocal operator) -->
</div></div>

<br>

<v-click>

Multiscale problem: the magnitude of $\varepsilon$ from $\varepsilon = O(1)$ to $\varepsilon \ll 1$.

</v-click>
---

# Motivation

<br>

<div class="top-0 right-20 absolute">

$$
\begin{equation*}
\varepsilon \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( \frac{1}{2} \int_{-1}^{1} f \mathrm{d} v' - f \right )
\end{equation*}
$$

</div>

Physics Informed Neural Networks(PINNs)

$$f_\theta^{\text{NN}}(t,x,v) \approx f(t,x,v)$$

Take the least square form of the linear transport equation as loss 

$$\mathcal{R}_{\text{PINN}}^{\varepsilon} = 
\frac{1}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}} \int_{\mathcal{D}} \int_\Omega \left| \varepsilon^2 \partial_t f^{\text{NN}}_{\theta} + \varepsilon {v} \cdot \nabla_x f^{\text{NN}}_{\theta} - \left ( \frac{1}{2} \int_{-1}^{1} f^{\text{NN}}_{\theta} \mathrm{d} v' - f^{\text{NN}}_{\theta} \right ) \right|^2 \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t}
$$

Procedure of solving PDEs by DNNs:

<v-clicks>

- Modeling: define the loss associated to a PDE;
- Architecture: build a deep neural network(function class) for the trail function;
- Optimization: minimize loss over the parameter space.

</v-clicks>

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

# Two illustrative examples


<!-- <div class="top-0 right-10 absolute">

$$
\begin{equation*}
\varepsilon \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( \frac{1}{2} \int_{-1}^{1} f \mathrm{d}{d} v' - f \right )
\end{equation*}
$$

</div> -->

<div>

$$
\varepsilon \partial_t f + v \cdot \nabla_x f = \frac{1}{\varepsilon} \left ( \rho(t,x) - f \right )
, \rho(t,x) =  \frac{1}{2} \int_{-1}^{1} f \mathrm{d} v'
$$
</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
Ex 1: Periodic boundary condition($\varepsilon=1$)

$$
\begin{equation*}
\begin{aligned}
  f(t, x_L, v) &= f(t, x_R, v), \\
  f_0(x, v) &= \frac{1 + \cos (4 \pi x)}{\sqrt{2\pi}}e^{-\frac{v^2}{2}}.
\end{aligned}
\end{equation*}
$$

<img src="/ex1_pinns.png" width="400" height="300" class="h-40 float-left ml-5"/>

</div><div v-click>

###### 
Ex 2: Inflow boundary condition($\varepsilon=10^{-8}$)

$$
\begin{equation*}
\begin{aligned}
  f(t, x_L, v) &= 1 \; \text{for} \; v > 0, \\
  f(t, x_R, v) &= 0 \; \text{for} \; v < 0, \\
  f_0(x, v) &= 0.
\end{aligned}
\end{equation*}
$$

<img src="/ex2_pinns.png" width="400" height="300" class="h-40 float-left ml-5"/>

PINN fails to obtain the ground truth !

</div></div>

---

# The failure of PINN Loss to resolve small scales

<!-- <div v-click-hide> 

<img src="/loss_1e-8_pinns.png" width="300" height="500" class="h-40 mx-auto"/>

</div> -->

<img src="/loss_1e-8_pinns.png" width="300" height="500" class="h-40 mx-auto"/>

<div class="overflow-auto h-60">

$$
\begin{equation*}
    \begin{aligned}
        \mathcal{R}^{\varepsilon}_{\text{PINN}} = & \frac{1}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}} \int_{\mathcal{D}} \int_\Omega \left| \varepsilon^2 \partial_t f^{\text{NN}}_{\theta} + \varepsilon {v} \cdot \nabla_x f^{\text{NN}}_{\theta} - \left ( \frac{1}{2} \int_{-1}^{1} f^{\text{NN}}_{\theta} \mathrm{d} v' - f^{\text{NN}}_{\theta} \right ) \right|^2 \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t} \\
                                                  & +  \frac{\lambda_1}{|\mathcal{T} \times \partial \mathcal{D} \times \Omega|}  \int_{\mathcal{T}} \int_{\partial \mathcal{D}} \int_\Omega |\mathcal{B}f^{\text{NN}}_{\theta} - F_{\text{B}}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t} \\
                                                  & +  \frac{\lambda_2}{|\mathcal{D} \times \Omega|} \int_{\mathcal{D}} \int_\Omega |\mathcal{I}f^{\text{NN}}_{\theta} - f_{0}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}}.
    \end{aligned}
\end{equation*}
$$


We only need to focus on the first term and taking $\varepsilon \to 0$, this will led to  

$$
\begin{equation*}
    \mathcal{R}_{\text{PINN}}^0 = \frac{1}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}} \int_{\mathcal{D}} \int_\Omega \left| - \left ( \frac{1}{2} \int_{-1}^{1} f^{\text{NN}}_{\theta} \mathrm{d} v' - f^{\text{NN}}_{\theta} \right ) \right|^2  \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t} ,
\end{equation*}
$$

which can be viewed as the PINN loss of the equilibrium equation

$$
\begin{equation*}
f^{\text{NN}}_{\theta} = \frac{1}{2} \int_{-1}^{1} f^{\text{NN}}_{\theta} \mathrm{d} v'.
\end{equation*}
$$

Next, we show the limit equation of the linear transport equation is the $\textcolor{red}{diffusion \; equation}$.

</div>

---

# The diffusion limit of the linear transport equation

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

Subsititing $f = \rho + \varepsilon g$ into the linear transport equation yields

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

Thus, one can get the micro-macro system for the linear trasport equation

$$
\begin{equation*}
    \left\{
    \begin{aligned}
        \partial_t \rho  +  \left \langle  v \cdot \nabla_x g  \right \rangle  & = 0, \\     
         \\       
        \varepsilon^2 \partial_t g + \varepsilon \left ( I - \Pi \right ) \left ( v \cdot \nabla_x g \right ) + v \cdot \nabla_x \rho   & =  - g.
    \end{aligned}
    \right.
\end{equation*}
$$

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

# The diffusion limit of the linear transport equation

<br>

The micro-macro system for the linear trasport equation

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

Sending $\varepsilon \to 0$, the above system formally approaches 

$$
\begin{equation*}
    \left\{
    \begin{aligned}
        \partial_t \rho  +  \left \langle  v \cdot \nabla_x g  \right \rangle  & = 0, \\          
        - v \cdot \nabla_x \rho  & = g.
    \end{aligned}
    \right.
\end{equation*}
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


<img src="/apnns.png" width="300" height="500" class="h-60 mx-auto"/>



Illustration of APNNs. $\mathcal{F^{\varepsilon}}$ is the microscopic equation that depends on the small scale parameter $\varepsilon$ and $\mathcal{F}^{0}$ is its macroscopic limit as $\varepsilon \to 0$, which is independent of $\varepsilon$. The latent solution of $\mathcal{F^{\varepsilon}}$ is approximated by neural networks with its measure denoted by $\mathcal{R}(\mathcal{F^{\varepsilon}})$. The asymptotic limit of $\mathcal{R}(\mathcal{F^{\varepsilon}})$ as $\varepsilon \to 0$, if exists, is denoted by $\mathcal{R}(\mathcal{F}^{0})$. If $loss(\mathcal{F}^{0})$ is a good measure of $\mathcal{F}^{0}$, then it is called asymptotic-preserving (AP).


---

# APNN v1: based on Micro-macro decomposition

<br>

<div v-click-hide> 

<img src="/APNNs.jpg" class="h-90 mx-auto"/>

Mass conservation mechanism $g^{\text{NN}}_{\theta} = \tilde{g}^{\text{NN}}_{\theta}-\left \langle \tilde{g}^{\text{NN}}_{\theta} \right\rangle$ is important!

</div>

---

# APNN v1: based on Micro-macro decomposition

<br>

$$
\begin{equation*}
    \begin{aligned}
        \mathcal{R}^{\varepsilon}_{\text{APNN}} = & \frac{1}{|\mathcal{T} \times \mathcal{D}|} \int_{\mathcal{T}} \int_{\mathcal{D}} | \partial_t \rho^{\text{NN}}_{\theta} + \nabla_x \cdot \left \langle   {v} g^{\text{NN}}_{\theta} \right \rangle - Q |^2 \mathrm{d}{{x}}  \mathrm{d}{t}                                      \\
                                                  & + \frac{1}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}} \int_{\mathcal{D}} \int_\Omega | \varepsilon^2 \partial_t g^{\text{NN}}_{\theta}  + \varepsilon (I - \Pi)({v} \cdot \nabla_x g^{\text{NN}}_{\theta})                                      \\
                                                  & \quad  + {v} \cdot  \nabla_{{x}} \rho^{\text{NN}}_{\theta} +  g^{\text{NN}}_{\theta}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t}                                                                                                      \\
                                                  & +  \frac{\lambda_1}{\mathcal{T} \times\partial \mathcal{D} \times \Omega|}  \int_{\mathcal{T}} \int_{\partial \mathcal{D}} \int_\Omega |\mathcal{B}(\rho^{\text{NN}}_{\theta} + \varepsilon g^{\text{NN}}_{\theta}) - F_{\text{B}}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}} \mathrm{d}{t} \\
                                                  & +  \frac{\lambda_2}{|\mathcal{D} \times \Omega|} \int_{\mathcal{D}} \int_\Omega |\mathcal{I}(\rho^{\text{NN}}_{\theta} + \varepsilon g^{\text{NN}}_{\theta}) - f_{0}|^2 \mathrm{d}{{v}} \mathrm{d}{{x}}.
    \end{aligned}
\end{equation*}
$$

$$
\begin{equation*}
    \begin{aligned}
        \mathcal{R}^{0}_{\text{APNN}} = & \frac{1}{|\mathcal{T} \times \mathcal{D}|} \int_{\mathcal{T}} \int_{\mathcal{D}} | \partial_t \rho^{\text{NN}}_{\theta} + \nabla_x \cdot \left \langle   {v} g^{\text{NN}}_{\theta} \right \rangle - Q |^2 \mathrm{d}{{x}}  \mathrm{d}{t}                                      \\
                                                  & + \frac{1}{|\mathcal{T} \times \mathcal{D} \times \Omega|} \int_{\mathcal{T}} \int_{\mathcal{D}} \int_\Omega | {v} \cdot  \nabla_{{x}} \rho^{\text{NN}}_{\theta} +  g^{\text{NN}}_{\theta} |^2.
    \end{aligned}
\end{equation*}
$$


---

# Test examples

<div>


</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
Ex 1: Periodic boundary condition($\varepsilon=1$)

$$
\begin{equation*}
\begin{aligned}
  f(t, x_L, v) &= f(t, x_R, v), \\
  f_0(x, v) &= \frac{1 + \cos (4 \pi x)}{\sqrt{2\pi}}e^{-\frac{v^2}{2}}.
\end{aligned}
\end{equation*}
$$

<img src="/ex1_apnns.png" width="400" height="300" class="h-40 float-left ml-5"/>

</div><div v-click>

###### 
Ex 2: Inflow boundary condition($\varepsilon=10^{-8}$)

$$
\begin{equation*}
\begin{aligned}
  f(t, x_L, v) &= 1 \; \text{for} \; v > 0, \\
  f(t, x_R, v) &= 0 \; \text{for} \; v < 0, \\
  f_0(x, v) &= 0.
\end{aligned}
\end{equation*}
$$

<img src="/ex2_apnns.png" width="400" height="300" class="h-40 float-left ml-5"/>


</div></div>

<div v-click>

One can observed that APNN work for both $\varepsilon=1$ and $\varepsilon=10^{-8}$.

</div>

---

# Mass conservation mechanism

<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
Ex 3: Inflow boundary condition($\varepsilon=10^{-8}$)

For the constraint $\left \langle g \right \rangle = 0$, one way is to construct a novel neural network for $g$ such that it  exactly satisfies $\left \langle g \right \rangle = 0$. 

The other way is to  treat it as a soft constraint with parameter $\lambda_3$, we use $\hat{g}_{\theta}^{\text{NN}}$ and modifies the loss as
$$
\begin{equation*}
  \mathcal{R}_{\text{APNN}} +  \frac{\lambda_3}{|\mathcal{T} \times \mathcal{D}|} \int_{\mathcal{T}} \int_{\mathcal{D}} | \left \langle  \hat{g}^{\text{NN}}_{\theta} \right \rangle - 0|^2 \mathrm{d}{{x}}  \mathrm{d}{t}.
\end{equation*}
$$



</div><div v-click>

###### 
Plot of density $\rho$ at $t = 0.1$: APNNs with soft constraint(marker) vs. Ref(line). 

<img src="/ex2_1e-8_noexact.png" width="400" height="300" class="h-50 float-left ml-5"/>

Mass conservation mechanism $g^{\text{NN}}_{\theta} = \tilde{g}^{\text{NN}}_{\theta} -  \left \langle \tilde{g}^{\text{NN}}_{\theta} \right \rangle$ is important!

</div></div>

<div v-click>


</div>

---

# Even- and odd- parity method

<br>

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

# Even- and odd- parity method

<br>

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

# Even- and odd- parity method

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

# APNN v2: based on even- and odd- parity 

<br>

**Here we singled out the equation of local conservation law $\partial_t \rho +  \left \langle v \partial_x j \right \rangle = 0$ is necessary in constructing the APNN loss. By coupling these equations of $r, j$ and $\rho$, one can obtain the loss for the diffusion limit equation.**


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

<br>

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

# APNN loss for odd-even parity system

<div class="overflow-auto h-100">

$$
\begin{equation*}
    \begin{aligned}
        \mathcal{R}^{\varepsilon}_{\text{APNN, parity}}=  & \frac{\lambda_1}{N_1^{(1)}} \sum_{i=1}^{N_1^{(1)}} | \varepsilon^2 \partial_t r^{\text{NN}}_{\theta}(t_i,x_i,v_i) + \varepsilon^2 v\partial_x j^{\text{NN}}_{\theta}(t_i,x_i,v_i) \\
        & \quad \quad \quad \quad \quad - (\rho^{\text{NN}}_{\theta}(t_i,x_i) - r^{\text{NN}}_{\theta}(t_i,x_i,v_i)) |^2   \\
                                                  & + \frac{\lambda_2}{N_1^{(2)}} \sum_{i=1}^{N_1^{(2)}}  |\varepsilon^2 \partial_t j^{\text{NN}}_{\theta}(t_i,x_i,v_i) + v \partial_x r^{\text{NN}}_{\theta}(t_i,x_i,v_i) - (-j^{\text{NN}}_{\theta}(t_i,x_i,v_i)) |^2                                        \\
                                                  & + \frac{\lambda_3}{N_1^{(3)}}\sum_{i=1}^{N_1^{(3)}} | \partial_t \rho^{\text{NN}}_{\theta}(t_i,x_i) +  \left \langle v\partial_x j^{\text{NN}}_{\theta} \right \rangle(t_i,x_i) |^2  
        \\
                                                  & + \frac{\lambda_4}{N_2} \sum_{i=1}^{N_2} |\rho^{\text{NN}}_{\theta}(t_i,x_i) -  \left \langle r^{\text{NN}}_{\theta} \right \rangle(t_i,x_i) |^2   
        \\
                                                  & +  \frac{\lambda_5}{N_3}\sum_{i=1}^{N_3}  |\rho^{\text{NN}}_{\theta}(0, x_i) - \left \langle f_{0} \right \rangle (x_i)|^2 
        \\
                                                  & +  \frac{\lambda_6}{N_4} \sum_{i=1}^{N_4} |\mathcal{I}(r^{\text{NN}}_{\theta} + \varepsilon j^{\text{NN}}_{\theta})(x_i,v_i) - f_{0}(x_i,v_i)|^2 
        \\
                                                  & +  \frac{\lambda_7}{N_5} \sum_{i=1}^{N_5}  |\mathcal{B}(r^{\text{NN}}_{\theta} + \varepsilon j^{\text{NN}}_{\theta})(t_i,x_i,v_i) - F_{\text{B}}(t_i,x_i,v_i)|^2 .
    \end{aligned}
\end{equation*}
$$

</div>


---

# Test examples - Case I

<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
Case I: Inflow boundary condition($\varepsilon=10^{-3}$)

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


</div><div v-click>

###### 
Plot of density $\rho$ at $t = 0, 0.05, 0.1$: APNNs(marker) vs. Ref(line). 

<img src="/dirichlet10_sol.png" width="400" height="300" class="h-40 float-left ml-5"/>

FCNet with units $[2, 128, 128, 128, 128, 1]$ for $\rho$ and $[3, 256, 256, 256, 256, 1]$ both for $r$ and $j$. Batch size is $512$ in domain, $1024 \times 2$ on boundary and $512$ on initial, the number of quadrature points is $30$. $\lambda_1 = \lambda_2 = \lambda_3 = \lambda_4 = \lambda_6 = 1, \lambda_7 = 10$.

Relative $\ell^2$ error of APNNs is $9.87 \times 10^{-3}$. 

</div></div>

<div v-click>


</div>

---

<div>

The numerical performance of enforcement of initial condition and the soft constraint $\rho = \left \langle r \right \rangle$ are discussed as follows.

Plot of density $\rho$ at $t = 0, 0.05, 0.1$: APNNs(marker) vs. Ref(line).

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

##### &emsp; &emsp; Figure 1:  without the enforcement of initial condition

<img src="/dirichlet10_sol_2.png" width="500" height="600" class="h-40 float-left ml-5"/>

Due to the poor approximate of initial and boundary layer effect, it gives wrong solution at time $t = 0, 0.05, 0.1$.

</div><div v-click>

##### &emsp; &emsp; &emsp; &emsp; Figure 2: without the constraint $\rho = \left \langle r \right \rangle$ 

<img src="/dirichlet10_sol_1.png" width="500" height="600" class="h-40 float-left ml-5"/>

The solutions are also wrong at time $t = 0, 0.05, 0.1$, therefore, we consider this constraint into our APNN loss.

</div></div>

<div v-click>


</div>

---

# Test examples - Case II

<div>


</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
Case II: Dirichlet boundary condition($\varepsilon=10^{-8}$)

$$
\begin{equation*}
\begin{aligned}
  f(t, x_L, v) &= 0 = f(t, x_R, v), \\
  \rho_0(x) &= \frac{1}{2}\left [ 1 + \sin \left ( 2 \pi x - \frac{\pi}{2} \right ) \right ] , \\
  f_0(x, v) &=  \left [ 1 + \sin \left ( 2 \pi x - \frac{\pi}{2} \right ) \right ] \cdot \frac{3}{\sqrt{2\pi}}e^{-\frac{(3v)^2}{2}}.
\end{aligned}
\end{equation*}
$$

In this case, $f$ has no jump with non-constant value of initial.


</div><div v-click>

###### 
Plot of density $\rho$ at $t = 0, 0.05, 0.1$: APNNs(marker) vs. Ref(line). 

<img src="/dirichlet00_sol.png" width="400" height="300" class="h-40 float-left ml-5"/>

FCNet with units $[2, 128, 128, 128, 128, 128, 1]$ for $\rho$ and $[3, 256, 256, 256, 256, 256, 1]$ both for $r$ and $j$. Batch size is $1024$ in domain, $512 \times 2$ on boundary and $512$ on initial, the number of quadrature points is $30$. $\lambda_1 = \lambda_2 = \lambda_5 = \lambda_6 = 1, \lambda_3 = \lambda_4 = \lambda_7 = 10$. 

Relative $\ell^2$ error of APNNs is $1.25 \times 10^{-2}$.

</div></div>

<div v-click>


</div>

---

# Test examples - Case III

<div>


</div>

<div class="grid grid-cols-1 gap-x-4 mt-4">

<div v-click>

###### 
Case III: UQ problems with inflow condition($\varepsilon=10^{-5}$)

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


</div><div v-click>

<img src="/uq_1e-5.png" width="400" height="300" class="h-60 mx-auto"/>



</div></div>

<div v-click>


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

# Euler equation 

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

# Test examples - Case I

<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
Case I: Initial conditions ($\varepsilon=10^{-3}$)

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

</div><div v-click>

###### 
Setting: ResNet with units $[3, 128, 128, 128, 128, 128, 128, 1]$ for $f$ and $[2, 64, 64, 64, 64, 64, 64, 1]$ both for $\rho, u$ and $T$. Batch size is $512$ in domain, and $256$ on initial, the number of quadrature points is $64$. 

$\lambda_9 = 0.1, \lambda_{11} = \lambda_{13} = \lambda_{14} = 10$ 
and others are set to be $1$. 

For $t = 0:$ mean square error of density, momentum and energy are $7.16\text{e-8}, 4.50\text{e-6}, 8.13\text{e-7}$. 

For $t = 0.1:$ relative $l^2$ error of density, momentum and energy are $5.43\text{e-3}, 6.35\text{e-3}, 4.47\text{e-2}$.

</div></div>

<div v-click>


</div>

---


<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0$

<img src="/ex1_sol_f_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>

<img src="/ex1_sol_macro_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>



</div><div v-click>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0.1$

<img src="/ex1_sol_f_t1.png" width="400" height="300" class="h-45 float-right ml-5"/>

<img src="/ex1_sol_macro_t1.png" width="400" height="300" class="h-45 float-right ml-5"/>


</div></div>

<div v-click>

</div>

---

# Test examples - Case II

<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
Case II: Initial conditions ($\varepsilon=10^{-3}$)

$$
\begin{equation*}
\begin{aligned}
& \rho_0(x) = 1.5 + (0.625 - 1.5) \cdot 
\frac{ 
{\sin}^{\frac{3}{7}} (\pi x) + 1
}
{2}, \\
& u_0(x) =0, \\
& T_0(x) = 1.5 + (0.75 - 1.5) \cdot 
\frac{ 
{\sin}^{\frac{3}{7}} (\pi x) + 1
}
{2}.
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

</div><div v-click>

###### 
Setting: ResNet with units $[3, 128, 128, 128, 128, 128, 128, 1]$ for $f$ and $[2, 64, 64, 64, 64, 64, 64, 1]$ both for $\rho, u$ and $T$. Batch size is $512$ in domain, and $256$ on initial, the number of quadrature points is $64$. 

$\lambda_{10} = \lambda_{11} = \lambda_{13} = \lambda_{14} = 10$ 
and others are set to be $1$. 

For $t = 0:$ mean square error of density, momentum and energy are $1.29\text{e-4}, 6.34\text{e-6}, 4.41\text{e-5}$. 

For $t = 0.1:$ relative $l^2$ error of density, momentum and energy are $1.36\text{e-2}, 2.00\text{e-2}, 3.99\text{e-2}$.

</div></div>

<div v-click>


</div>

---


<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0$

<img src="/ex2_sol_f_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>

<img src="/ex2_sol_macro_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>



</div><div v-click>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0.1$

<img src="/ex2_sol_f_t1.png" width="400" height="300" class="h-45 float-right ml-5"/>

<img src="/ex2_sol_macro_t1.png" width="400" height="300" class="h-45 float-right ml-5"/>


</div></div>

<div v-click>

</div>

---

# Test examples - Case III

<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
Case III: Initial conditions ($\varepsilon=10^{-3}$)

$$
\begin{equation*}
\begin{aligned}
& \rho_0(x) = 1.5 + (0.625 - 1.5) \cdot 
\frac{ 
\tanh (10 x) + 1
}
{2}, \\
& u_0(x) =0, \\
& T_0(x) = 1.5 + (0.75 - 1.5) \cdot 
\frac{ 
\tanh (10 x) + 1
}
{2}.
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

</div><div v-click>

###### 
Setting: ResNet with units $[3, 128, 128, 128, 128, 128, 128, 1]$ for $f$ and $[2, 64, 64, 64, 64, 64, 64, 1]$ both for $\rho, u$ and $T$. Batch size is $512$ in domain, and $256$ on initial, the number of quadrature points is $64$. 

$\lambda_{9} = 0.1$, $\lambda_{11} = \lambda_{13} = \lambda_{14} = 10$ 
and others are set to be $1$. 

For $t = 0:$ mean square error of density, momentum and energy are $4.87\text{e-8}, 1.22\text{e-6}, 3.29\text{e-8}$. 

For $t = 0.1:$ relative $l^2$ error of density, momentum and energy are $6.19\text{e-3}, 1.78\text{e-2}, 3.60\text{e-2}$.

</div></div>

<div v-click>


</div>

---


<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0$

<img src="/ex3_sol_f_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>

<img src="/ex3_sol_macro_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>



</div><div v-click>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0.1$

<img src="/ex3_sol_f_t1.png" width="400" height="300" class="h-45 float-right ml-5"/>

<img src="/ex3_sol_macro_t1.png" width="400" height="300" class="h-45 float-right ml-5"/>


</div></div>

<div v-click>

</div>

---

# Test examples - Case IV

<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
Case IV: Initial conditions ($\varepsilon=1$)

$$
\begin{equation*}
\begin{aligned}
& \rho_0(x) = 1.5 + (0.625 - 1.5) \cdot 
\frac{ 
\tanh (20 x) + 1
}
{2}, \\
& u_0(x) =0, \\
& T_0(x) = 1.5 + (0.75 - 1.5) \cdot 
\frac{ 
\tanh (20 x) + 1
}
{2}.
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

</div><div v-click>

###### 
Setting: ResNet with units $[3, 128, 128, 128, 128, 128, 128, 1]$ for $f$ and $[2, 64, 64, 64, 64, 64, 64, 1]$ both for $\rho, u$ and $T$. Batch size is $512$ in domain, and $256$ on initial, the number of quadrature points is $64$. 

$\lambda_{9} = 0.1$, $\lambda_{11} = \lambda_{13} = \lambda_{14} = 10$ 
and others are set to be $1$. 

For $t = 0:$ mean square error of density, momentum and energy are $9.89\text{e-8}, 2.34\text{e-6}, 2.08\text{e-7}$. 

For $t = 0.1:$ relative $l^2$ error of density, momentum and energy are $6.41\text{e-3}, 8.72\text{e-3}, 1.62\text{e-2}$.

</div></div>

<div v-click>


</div>

---


<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0$

<img src="/ex4_sol_f_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>

<img src="/ex4_sol_macro_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>



</div><div v-click>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0.1$

<img src="/ex4_sol_f_t1.png" width="400" height="300" class="h-45 float-right ml-5"/> -->

<img src="/ex4_sol_macro_t1.png" width="400" height="300" class="h-45 float-right ml-5"/>


</div></div>

<div v-click>

</div>

---

# Training process

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 

Case I

<img src="/case1_sol_err.png" width="400" height="300" class="h-40 float-left ml-5"/> 

Case II

<img src="/case2_sol_err.png" width="400" height="300" class="h-40 float-left ml-5"/> 



</div><div v-click>

###### 

Case III

<img src="/case3_sol_err.png" width="400" height="300" class="h-40 float-left ml-5"/> 

Case IV

<img src="/case4_sol_err.png" width="400" height="300" class="h-40 float-left ml-5"/> 


</div></div>

<div v-click>

</div>

---

# Test examples - Case V

<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
Case V: Initial conditions ($\varepsilon=10^{-2}$)

$$
\begin{equation*}
\begin{aligned}
&     \begin{equation*}
    \rho_0(x) =
    \left\{
    \begin{aligned}
         & 1.5, \; x < 0, \\
         & 0.625 \; x \ge 0,
    \end{aligned}
    \right.
    \end{equation*} \\
& u_0(x) =0, \\
& \begin{equation*}
    T_0(x) =
    \left\{
    \begin{aligned}
         & 1.5, \; x < 0, \\
         & 0.75 \; x \ge 0.
    \end{aligned}
    \right.
    \end{equation*}.
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

</div><div v-click>

###### 
Setting: ResNet with units $[3, 128, 128, 128, 128, 128, 128, 1]$ for $f$ and $[2, 96, 96, 96, 96, 96, 96, 1]$ both for $\rho, u$ and $T$. Batch size is $512$ in domain, and $512$ on initial, the number of quadrature points is $64$. 

$\lambda_{5} = \lambda_{6} = 10$, $\lambda_{11} = \lambda_{13} = \lambda_{14} = 100$ 
and others are set to be $1$. 

For $t = 0:$ mean square error of density, momentum and energy are $1.72\text{e-4}, 1.59\text{e-4}, 7.08\text{e-5}$. 

For $t = 0.05:$ relative $l^2$ error of density, momentum and energy are $4.03\text{e-2}, 2.31\text{e-2}, 5.15\text{e-2}$.

</div></div>

<div v-click>


</div>

---


<div>

</div>

<div class="grid grid-cols-2 gap-x-4 mt-4">

<div v-click>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0$

<img src="/ex5_sol_f_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>

<img src="/ex5_sol_macro_t0.png" width="400" height="300" class="h-45 float-left ml-5"/>



</div><div v-click>

###### 
The integrals of approximate $f$ and approximate density, momentum and energy at time $t = 0.1$

<img src="/ex5_sol_f_t1.png" width="400" height="300" class="h-45 float-right ml-5"/> -->

<img src="/ex5_sol_macro_t1.png" width="400" height="300" class="h-45 float-right ml-5"/>


</div></div>

<div v-click>

</div>

---

# Conclusions

<br>

We propose several Asymptotic-Preserving Neural Networks for solving the multiscale time-dependent kinetic problems:

- Linear transport
  - APNN based on micro-macro decomposition
  - APNN based on odd-even parity method
  
- Boltzmann-BGK equation
  - APNN based on local conservation laws 
