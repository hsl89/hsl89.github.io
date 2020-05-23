---
layout: post
title: Notes from Yarin Gal's Thesis
---

Central theme:
Optimizing any deterministic NN with dropout is equivalent to 
a form of approximate inference in a probabilistic interpretation
of the model. This means the optimal weights found through the optimization
of a dropout NN are the same as the optimal variational parameters
in a Bayesian NN with the same structure. 

Why do we call the paramters in a Bayesian NN a variational parameter?

want to find $q(w)$ that is close to the model's posterior $p(w|X, Y)$ 
what is $\theta$ in relation to the weight parameters to Bayesian NN?


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

such that $$\mathbb{E}_{S, \epsilon}(\hat{L}_{MC}(\theta)) = \hat{L}_{VI}(\theta)$$

Therefore the SGD algorithm for minimizing $$q_{\theta}(w)$$ and $$p(w| X, Y)$$

>Given dataset X, Y<br>
Define learning rate $\eta$<br>
Initialize parameters $\theta$ randomly<br>

While $\theta$ has not converged
>>Sample $M$ random variables $\hat{\epsilon}\sim p(\epsilon)$, $S$ a random 
subset of $\{1, \dots, N\}$ of size $M$<br>
Calculate Stochastic derivative estimator w.r.t. $\theta$:

$$
\hat{\Delta\theta} \leftarrow -\frac{N}{M} \sum_{i\in S} 
\frac{\partial}{\partial\theta} \log p(y_i | f^{g(\theta, \epsilon)}) 
+ \frac{\partial}{\partial\theta} KL(q_{\theta}(w) || p(w))
$$

>>Update $\theta$:

$$
    \theta \leftarrow \theta + \eta \hat{\Delta\theta}
$$

Let's take a look at its relation with some stochastic regularization techniques.
The most popular SRT is *dropout* Suppose each input feature is used with 
probability $p$. Suppose the model has $N$ parameters, then dropout produces
$2^N$ 'thinned' networks. At inference time, we want the expected prediction
from those $2^N$ models, so we scale each weights by $p$ [Hinton et al](
http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

SRT injects noise on the feature space. In Bayesian NNs, the uncertainty 
comes from model parameters. It is easy to transform noise from features
to the model paramters. Suppose masks for features are $\epsilon_1, \epsilon_2$
and parameters are
$\theta = \{M_1, M_2, b\}$ 

Then, the model is equivalent to stochastic paramters
$\hat{\theta} = \{ diag(\epsilon_1)M_1, diag(\epsilon_2)M_2 \} $

Then we can write the optimization objective as

$$
\hat{L}_{dropout}(M_1, M_2, b) := 
\frac{1}{M} \sum_{i\in S} E^{\hat{W}^i_1, \hat{W}^i_2, b}(x_i, y_i) 
    + \lambda_1 ||M_1||^2 + \lambda_2 ||M_2||^2 + \lambda_3||b||^2
$$

with $\hat{W}^i_1$ and $\hat{W}^i_2$ corresponding to new masks $\hat{\epsilon}^i_1$
and $\hat{\epsilon}^i_2$ sampled for data point $i$. 

For regression problem, minimizing MSL is equivalent to maximizing log 
likelihood 

$$
\log p(y|f^{M_1, M2, b}(x)) + \text{const}
$$
where $p(y| f^{M_1, M_2, b}(x)) = N(y; f^{M_1, M_2, b}(x), \tau^{-1}I) $

$\hat{w} = \{\hat{W}^i_1, \hat{W}^i_2, b \} =: g(\theta, \hat{\epsilon}_i)$ 
$p(\epsilon)$ is the product of Bernoulli distributions with probability
$p_i$ (the probability the neuron is not turned off). 

So the loss objective is 

$$
\hat{L}_{dropout}(M_1, M_2, b) = -\frac{1}{M_{\tau}} 
\sum_{i \in S} \log p(y_i | f^g(x)) + \lambda_1||M_1||^2 
+ \lambda_2 ||M_2||^2 + \lambda_3||b||^2
$$

This optimization objective is same to that of approximate inference
if in $\hat{L}_{MC}$ we choose the prior $p(w)$ in the way so that

$$
KL(q_{\theta}(w) || p(w)) \propto \lambda_1||M_1||^2 + \lambda_2||M_2||^2 
+ \lambda_3||b||^2
$$




If weights are trained with probability $p$, i.e. each weight has a 
probability $p$ to be turned on. Then, in the test time, we want
expected output of all those "thined network" therefore, we need to 
multiply each weights by $p$. 


Optimising any neural network with dropout is equivalent to a form
of approximate inference in a probabilistic interpretation of the 
model.


## Model uncertainty in Bayesian neural networks

variance ratio
predictive entropy
information gain

### Some difficulties with measuring the uncertainty this way
Model's uncertainty is not calibrated. 

# Reference
[Grave, 2011, Practical Variational Inference for Neural Network](
https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks.pdf)


