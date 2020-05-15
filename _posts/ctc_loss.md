
The basic idea is to interpret the network outputs as a probability distribution over
the all possible label seqeuences, conditioned on a given input sequence. 
Given this distribution, an objective function can be derived can be derived that
maximises the probabilities of the correct labellins. 

The task of labelling unsegmented data sequence is called *temperal classification* .
Classification done by RNN is called *connectionist temporal classification* (because
labeling depends on the previous frame)

By contrast, we refer to independent labelling of each time-step as *framewise classification*

### Mathematical Formulation of CTC loss

$S$ denotes the training samples drawn from a fixed distribution $D_{X \times Z}$.
The input space $X = \mathbb{R}^*$ is the set of all sequences of $m$ dimentional vectors.
The target space $L^*$ is the set of all sequences over finite set $L$ (alphabet). 
Each element in $S$ is a pair of sequences $(x, z)$. The target sequence $z# is at most as long as 
input sequence $x$

Let $h: X \rightarrow Z$ be the classifier that minimize 
some task specific errors. 

#### Label Error Rate


#### Connectionist Temporal Classification

Key idea is to transform the network outputs into a 
conditional probability distribution over label sequences.

For an input sequence $x$ of length $T$, define a RNN 
with $m$ inputs, $n$ outputs and weight vector $w$ as 
a continuous map

$$
    N_w: (\mathbb{R}^m)^T \rightarrow (\mathbb{R}^m)^T
$$

Let $y = N_w(x)$ be the sequence of network outputs, 
and denote $y_k^t$ the activation of unit $k$ at time 
step $t$. The $y^t_k$ is interprested as the probability
of observing label $k$ at time $t$, which defines a 
distribution over the set $L^{\prime T}$, where
$L^{\prime} = L \cup {\text{blank}}$

$$
p(\pi | x ) = \pi\limits_{t=1}^T y^t_{\pi_t} 
$$

#### Prediction
Next step is to define a many-to-one map 

$$
B: L^{\prime T} \rightarrow L^{<<T}
$$

$L^{<<T}$ is the set of labelings (sequences of length
less than $T$) 

$B$ works by removing blanks and repeated labels from 
the paths. 

$B(a-ab-) = B(-aa--abb) = abb$

Then use $B$ to define the conditional probability of a
given labeling $l \in L^{\le T}$ as the sum of the 
probabilities of all the pathes to it

$$
p(l | x) = \sum\limits_{\pi \in B^{-1}(l)} p(\pi | x)
$$


$h(x) = \text{argmax}_{l} p(l | x)$ 

$h(x)$ is hard to compute, because one needs to run an 
exhaustive search on the entire labeling space (intractable)

A good approximation

$$
    h(x) \approx B(\pi^*)
$$
where $\pi^* = \text{argmax}_{\pi\in N^t)(\pi | x)$, ie
$\pi^*$ is the concatenation of the most probably neuron
at each time step. 

#### Training the network

Objective function is derived from the principal of 
maximal likelihood, i.e. minimizing the objective function
maximises log likelihood of the target label.

For some seq of length $r$, denote by $p_{1:p}$ and
$q_{r-p: p}$ its first and last $p$ symbols respectively.
Define the forward variable $a_t(s)$ to be the total 
probability of $l_{1:s}$ at time $t$

$$
a_t(s) = \sum\limits_{\pi\in N^T, B(\pi_{1:t})=1_{1:s}} 
$$

Insert blank between each chars and add blank to the 
begining and the end of the labeling. 
$abc \rightarrow -a-b-c-$

Let $l^{\prime}$ be the augmented label. In calculating
the probabilities pf prefixed of $I^{\prime}$ we allow
all transitions between blank and non-blank labels.
We allow all prefixes to start with either a blank (b) 
or the first symbol in l $l_1$.

$C_t = \sum_{s} a_t(s), D_t = \sum_{s} \beta_t(s)$




