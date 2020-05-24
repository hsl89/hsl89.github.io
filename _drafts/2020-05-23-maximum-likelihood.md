
---
layout: post
comments: true
title: Maximum Likelihood Estimate
date: 2020-05-23
tags: Statistics
---

> I want to review MLE. 

MLE: $$\hat{\mu} = \text{arg max}_{\mu \in \Gamma} \{I_x(\mu)\}$$

Can be extened to provide maximum likelihood estimate for a function
$$\hat{\theta} = T(\hat{\mu})$$ 

$$I_x(\theta) = \log f_{\theta}(x)$$ 

$$\hat{\theta}$$ is always the MLE of $\theta$ 

Fisher information is the variance of $\dot{l}_x(\theta)$ 

Note $\dot{l}_x(\theta)$ is a family of functions parameterized
by $\theta$. 


$$
\italic{I}_{\theta} = \int_X \dot{l}_x(theta)^2 f_{\theta}(x) dx 
$$

The reason to consider such derivative is that at $\hat{\theta}$
$\italic{I}_{\theta}$ should be 0. 


Fisher's fundamental theomre for the MLE, in large samples

$$
\hat{\theta} \dot{\sim} \mathcal{N}(\theta, \frac{1}{n\italic{I}_{\theta}}
$$

what does it even mean? $\theta$ is the parameters that varies in the 
familiy $\Omega$, why put it as mean?


