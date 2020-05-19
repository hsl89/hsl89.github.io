---
layout: post
comments: true
title: Overview of Bayesian deep learning
date: 2020-05-14
tags: Bayesian-deep-learning
---

> As we have seen from my [previous post](
https://hsl89.github.io/uncertainty-of-deep-neural-network/).
The probability vector of
a deterministic network cannot consistently capture the uncertainty of its
prediction. And we have also seen that if we use the entropy of the probablity 
vector as a proxy to uncertainty, the performance of active learning is 
pretty bad. In this post, I want to discuss some basics of Bayesian 
statistics and using it to study the model uncertainty. Then we will use 
this uncertainty to design an active learning query strategy.

<!--more-->

## Warm up
In supervised machine learning, the objective is to find a function
$f$ that best describes the relation between
the  features $X = (x_1, x_2, \cdots, x_n)$
and the labels $Y = (y_1, y_2, \cdots, y_n)$. Suppose the data 
$D = (X, Y)$ is observed from the system $S$. Then, $f$ is a proxy
to the real data generation process of $S$ which is difficult to know.
The criterion for a good model $f$ is that when new observations 
$D^{\prime} = (X^{\prime}, Y^{\prime})$ arises, $f$ can still relate
$X^{\prime}$ to $Y^{\prime}$ with good precision. 

Let's take one step back and think about what are we doing when building
and training an ML model?
There are infinitely many ways you can relate $X$ to $Y$ up to certain extent, 
the process of building and training an ML model is amount to find an $f$
such that *after* seeing the data $D$, we belief $f$ is the most likely 
candidate that describes the unknown data generation process of $S$.

Formally, let $\Theta$ be the function space that describes the process of 
$S$ that relates $X$ to $Y$, i.e. $\Theta$ is a set of all *possible* candidates
$\theta$ that relates $X$ to $Y$. The process of machine learning is to 
find a conditional distribution $p(\theta | D)$ over $\Theta$. This distribution
reflects our belief on how likely the candidate $\theta$ is the one that relates
$X$ to $Y$. This distribution $p(\theta | D)$ has an interesting name, it is called
the *posterior* distribution on $\Theta$, because it is something we derive *after*
seeing $D$.

This brings us to the playground of Bayesian statistics, because it offers a 
framework for us to think about $p(\theta | D)$. 


## Bayes' Theorem
There are two things you can do when you are handed with the data $D$:

1) Based on your understanding of the $S$, you make a hypothesis $H$ about the 
how $X$ and $Y$ are related, then you update your hypothesis $H$ after
inspecting $D$

2) You can compute frequency statistics on $D$ and make claims on $S$ 
directly from those frequency statistics. For example, you might have
heard people talking about making *Null Hypothesis* and calculating
$p$-value that reflects the probability that **given** Null Hypothesis
is true, it would be rejected $p*100$ times out of 100 trials 
due to randomness in data collection. 


If you prefer 1, then you are labeled as a *Bayesian*
If you prefer 2, then you are labeled as a *frequentist*

The difference between Bayesians and frequentists is that 
Bayesians want $P(H | D)$ whereas frequentists want $P(D | H)$, i.e.
Bayesians want to know how much belief to put in the hypothesis 
$H$ after seeing the data $D$; frequentists want to know given the 
hypothesis is true, what's the odd for the observation $D$?

One thing that relates these two schools of thought is the Bayes' Theorem:

$$
P(H | D) = \frac{P(D | H) P(H)}{P(D)}
$$

In Bayesian statistics, each term above has a name:

$P(D | H)$ is the *likelyhood* of the dataset $D$,i.e.
how likely can we see $D$ given hypothesis $H$

$P(H)$ is the *prior* on the hypothesis $H$,i.e. how
deep we believe the data generation process is $H$. 

$P(D)$ is called the *evidence*. It is the probablity of the 
observation $D$. One way to think about it is "the average" 
probability of seeing the observation $D$ over all hypothesis
of data generation process:

$$
P(D) = \int P(D | H) P(H) dH
$$

If you do machine learning, then you are baptized by Bayesian
school of thought, even if you have not heard of Bayesian statistics
until 10 mins ago. When you do the following things

```
from sklearn.ensembles import RandomForestClassifier
```
or 
```
import torch.nn as nn
```
you are conjuring up a hypothesis $H$ about the relation between $X$ and $Y$,
a function space $\Theta$ that describes the relation between $X$ and $Y$ and
a *prior* probability distribution $p(\theta)$ on $\Theta$. Except intead of
looking at all possible functions, you are looking at functions that look like
a random forest in case 1, a neural network in case 2.

When you train your model, you are updating your posterior $P(\theta | D)$ on 
$\Theta$. Each "fit" you do gives you the most likely candidate $\theta$ given $D$.


## Uncertainty
Before we jump into uncertainty in machine learning, let's ask ourselves what are
we even uncertain about? 

I think one reasonable thing to be uncertain about is the posterior distribution
$p(\theta | D)$ on $\Theta$. We find a posterior through machine learning 
(aka maximum likelyhood),
how do we know how far it is from the true posterior?

You can "compute" this uncertainty via Shannon entropy:

$$
H[\theta | D] = -\int_{\Theta} p(\theta | D) \log p(\theta | D) d\theta
$$

The uncertainty on the posterior $p(\theta | D)$ naturally carries over 
to the uncertainty for making predictions. Let $x^*$ be a new sample
observed from the system $S$.

Given our function space $\Theta$ and 
posterior $p(\theta | D)$, the probability of $x^\*$ mapped to $y^\*=c$
under the unknown data generation process of $S$ is given by

$$
p(y^* | x^*, D) = \int_{\Theta} p(y^* | x^*, \theta)p(\theta | D) d\theta
$$

If the output $y^*$ is a discrete random variable that takes on value
$c_1, c_2, \cdots, c_n$, then we can again use Shannon entropy to 
"compute" this predictive uncertainty

$$
H[y^*|x^*, D] = - \sum_{i=1}^n p(y^*=c_i | x^*, D)\log p(y^*=c_i | x^*, D)
$$

## Information-theoretic active learning
The goal of active learning is to selectively add data to the training 
set $D$ that "boost the model's performance to the largest extent". 
In light of the Bayesian philosophy, how do we materialize the objective of 
active learning? 

One reasonable thing we can do is to enlarge the data set $D$ in the way
so that it reduces the uncertainty of posterior $p(\theta | D)$ fastest.
We assumed the existence of a function space $\Theta$ that gathers all potential
candidates for describing the data generation process of $S$, we have computed
the posterior $p(\theta | D)$ on $\Theta$. Now, we are given more data $D^{\prime}$
we can say the most valuable sample $(x', y')$ is the one that reduces the 
uncertainty of the posterior $p(\theta | D \cup \{(x', y')\})$ fastest, i.e.

$$
\tag{1}
\text{arg max}_{(x',y')} H[\theta | D] - \mathbb{E}_{y'\sim p(y'|x',D)}
    (H[\theta | D'\cup {(x', y')}])
$$

Of course, in practice there is no way you can evalute the above quantity,
because it involves many intractable integrals. 
But if $y$ is a discrete random variable, i.e. classification ML problem, 
there are ways to get around it. Instead of looking at the posterior uncertainty,
we can look at the predictive uncertainty because $H[y^\*|x^\*, D]$ is computed
as a finite sum. 

In this case, the sample $(x', y')$ satisfies equation (1) is also the one
that has the biggest conditional information gain:

$$
\tag{2}
I[\theta, y'|x', D] = \text{arg max}_{(x', y')} H[y'|x',D] -
    \mathbb{E}_{\theta \sim p(\theta | D)}[H[y'|x', \theta]]
$$

We can interpret the samples $(x', y')$ with maximal conditional information
gain as the one such that the overall predictive uncertainty is high (high 
$H[y|x, D]$), but for each fixed element $\theta$ in $\Theta$, the 
predictive entropy is low (low $H[y|x, \theta]$). Those are the samples
that incurs biggest dissagreement among individual members of $\Theta$.

The query strategy defined by eq (2) is called Bayesian Active Learning by
Disagreement (BALD)

In the next blog, I will discuss practical implementation of BALD via
approximate inference.



## Reference

[Bayesians vs Frequentists](
https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading20.pdf)


[Uncertainty in Deep Learning (Yarin Gal's Thesis)](
http://mlg.eng.cam.ac.uk/yarin/thesis/3_bayesian_deep_learning.pdf
)

[Deep Bayesian Active Learning with Image Data](
https://arxiv.org/pdf/1703.02910.pdf)

[Bayesian Active Learning for Classification and Preference Learning](
https://arxiv.org/pdf/1112.5745.pdf)

