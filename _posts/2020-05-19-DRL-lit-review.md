---
layout: post
comments: true
title: "Not-So-formal Lit Review of Recent Progress in Deep Reinforcement Learning" 
date: 2020-05-19
tags: Deep-learning Reinforcement-learning
---

> I want to know what's going on in the DRL community. That's why I am doing this lit
review.

<!--more-->

## Rainbow: Combining Improvements in Deep Reinforcement Learning
[paper is here](https://arxiv.org/pdf/1710.02298.pdf)
This is a good entry-point, because the authors in this paper combines various 
improvements of deep Q-learning into one coherent system. They argued that the 
independent improvements (as extensions of DQL) are indeed complementary. 
Here are a list of improvements they combined:

### Double Q-learning
[van Hasslt 2010](link) [van Hasselt, Guez, and Silver 2016](
https://arxiv.org/pdf/1509.06461.pdf)

In this algo, the authors addresses the *overestimation* problem when 
updating the value function acccording to 

$$
\tag{1}
(R_{t+1} + \gamma_{t+1} \text{max}  q_{\bar{\theta}} 
    (S_{t+1}, a^{\prime}) - q_{\theta}(S_t, A_t))^2
$$

$\theta$ is the parameters of the *online net*, the one that evaluates
the states and actions. 
Action at each time step is selected based on the estimate
of $\theta$. $\bar{\theta}$ is the paramters of the *target net*. It is
the one that is periodically updated by the parameters of $\theta$. 
Using a target net makes training more stable. The optimization algorithm
is RMSprop.

*Question: why RMSprop is a reasonable optimizer here?*
No clue, will investigate later.

What is an overestimation bias?
It is the error that comes from taking the best action in Eq (1)

A theorem that estabilishes the theoretical fundation of the overestimation
bias is the following

Theorem 1 in [Double Q-Learning paper](https://arxiv.org/pdf/1509.06461.pdf)

*Consider a state in which all the true optimal action values are equal
at $Q_\*(s, a) = V_\*(s)$ for some $V_\*(s)$. Let $Q_t$ be arbitrary value 
estimates that are on the whole unbiased, i.e. 
$ \sum_a (Q_t(s, a)  - V_\*(s)) = 0$, that that are not all correct, such
that $ \frac{1}{m} \sum_a (Q_t(s, a) - V_\*(s))^2 = C$ for some $C > 0$,
where $m >= 2$ is the number of actions in $s$. Under these conditions,
$$
\text{max}_a Q_t(s, a) \ge V_*(s) + \sqrt{\frac{C}{m-1}}
$$
The lower bound is tight. Under the same condition, the lower bound on 
the aboslute error of Double Q-Learning esitimate is 0*

Ok, what this theorem say? It says, the traditional Q-learning incurs 
a bias of at least $\sqrt{\frac{C}{m-1}}$, whereas the Double Q-learning
does not incur any bias.

Ok, so what is Double Q-Learning and why it does not incur overestimation
bias?

The target of Q-Learning is 

$$
Y^Q_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t)
$$

which can be rewritten as 

$$
Y^Q_t = R_{t+1} + \gamma Q(S_{t+1}, \text{argmax}_a Q(S_{t+1},a;\theta_t); 
    \theta_t)
$$

The target of Double Q-Learning is

$$
Y^{DoubleQ}_t = R_{t+1} + \gamma 
Q(S_{t+1}, \text{argmax}_a Q(S_{t+1}, a; \theta_t); \theta^{\prime}_t)
$$

The loss of objective is 

$$
(R_{t+1} + \gamma q_{\bar{\theta}}(S_{t+1}, \text{argmax}_{a^{\prime}} 
    q_{\theta}(S_{t+1}, a^{\prime})) - q_{\theta}(S_t, A_t))^2
$$

Why Double Q-learning removes overestimation bias,
No clue, will investigate latter.



