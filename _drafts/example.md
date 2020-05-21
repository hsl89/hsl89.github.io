---
layout: post
title: Notes from Yarin Gal's Thesis
header-includes:
    - \usepackage{algorithm2e}
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


## Stochastic Regularizer

### Dropout and Approximate Inference

To use the pathwise derivative estimator, we need to reparametrize each 
$q_{\theta_{l, i}}(w_{l,i})$ (The family of distributions on $w_{l, i}$
paramterized by $\theta_{l, i}$) as $w_{l,i} = g(\theta_{l,i}, \epsilon_{l,i})$
and specify some $p(\epsilon_{l,i})$. 

The loss objective of variational inference is 

$$
\begin{align}
\hat{L}_{VI}(\theta) & = - C \sum_{i\in S} 
\int q_{\theta}(w) \log p(y_i | f^w(x_i)) dw + KL(q_{\theta}(w) || p(w))  \\
& = -C \sum_{i\in S} 
\int p(\epsilon)\log p(y_i|f^{g(\theta, \epsilon)}(x_i))d\epsilon 
+ KL(q_{\theta}(w) || p(w))
\end{align}
$$

Then we can replace the expected log likelihood with its stochastic 
estimator

$$
\hat{L}_{MC}(\theta) = -C \sum_{i \in S} \log p(y_i|f^{g(\theta,\epsilon)}(x_i)) 
+ KL(q_{\theta}(w) || p(w))
$$

such that $\mathbb{E}_{S, \epsilon}(\hat{L}_{MC}(\theta)) = \hat{L}_{VI}(\theta)$

Therefore the SGD algorithm for minimizing $q_{\theta}(w)$ and $p(w|X, Y)$
is 



If weights are trained with probability $p$, i.e. each weight has a 
probability $p$ to be turned on. Then, in the test time, we want
expected output of all those "thined network" therefore, we need to 
multiply each weights by $p$. 


Optimising any neural network with dropout is equivalent to a form
of approximate inference in a probabilistic interpretation of the 
model.




## Reference
[Grave, 2011, Practical Variational Inference for Neural Network](
https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks.pdf)

