---
layout: post
title: Notes from Yarin Gal's Thesis
---

## Techniques to estimate the expected log likelihood 

$$
\int q_{\theta} \log p(y_i | f^w(x_i)) dw
$$

### Monte Carlo Estimator in variational inference

We wish to estimate the derivatives of the expected
log likelihood with respect to $\theta$. This allows
us to optimize the objective for the variational inference. 

Consider in general

$$
I(\theta) = \frac{\partial}{\partial\theta}
\int f(x) p_{\theta}(x) dx
$$


#### The score function estimator
Assume we can do differentiating outside integral side

$$
\begin{align}
\frac{\partial}{\partial\theta}
\int f(x) p_{\theta}(x) dx & = 
\int f(x) \frac{\partial}{\partial\theta} p_{\theta}(x) dx \\
& \int f(x) \frac{\partial \log p_{\theta}(x)}{\partial\theta} 
p_{\theta}(x) dx
\end{align}
$$

This leads to an unbiased stochastic estimator 

$$
\hat{I}_1(\theta) = f(x)\frac{\partial \log p_{\theta}(x)}{\partial\theta}
$$

with $x \sim p_{\theta}(x)$, i.e. sample a few $x$ from 
$p_{\theta}(x)$, we can use it to estimate $I(\theta)$



