---
title: "Weight Initialization: Why Zero Fails, and Why Xavier and He Work"
date: 2026-09-04 10:00:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [weight-initialization, xavier-initialization, he-initialization, neural-networks, backpropagation, activation-functions]
math: true
toc: true
comments: true
published: true
permalink: /posts/weight-initialization-zero-xavier-he/
description: "Why neural networks need random but carefully scaled weights, how zero initialization creates symmetry, and how Xavier and He initialization preserve signals through deep networks."
---

Why do neural networks need a special initialization scheme? Why not set every weight to zero? And if zero is a bad choice, why not simply sample every weight from a random distribution?

The short answer is:

1. **All-zero weights fail to break symmetry.** Neurons in the same layer begin identically and cannot learn different features.
2. **Arbitrarily scaled random weights can still fail.** Signals and gradients may shrink toward zero or grow without bound as they pass through many layers.
3. **Xavier and He initialization are random initializations with carefully chosen scales.** Their job is to keep information flowing at a reasonable magnitude.

So the real comparison is not *random initialization versus Xavier or He*. Xavier and He are particular forms of random initialization.

This post develops that idea from first principles.

## What initialization actually chooses

Consider layer $\ell$ of a feed-forward network:

$$
z^{(\ell)}=W^{(\ell)}a^{(\ell-1)}+b^{(\ell)},
\qquad
a^{(\ell)}=\phi\!\left(z^{(\ell)}\right).
$$

Before training begins, we must assign initial values to every weight matrix $W^{(\ell)}$ and bias vector $b^{(\ell)}$. These values determine:

- whether different neurons can learn different features;
- the scale of activations during the forward pass;
- the scale of gradients during backpropagation;
- whether sigmoid or tanh units begin in their responsive region or in saturation;
- how easily the optimizer can make useful progress in its first updates.

Initialization does not need to find a good solution by itself. It needs to place the network in a state from which learning is possible and numerically stable.

## If training changes the weights, why does the starting point matter?

It is tempting to think that initialization becomes irrelevant because training eventually changes every weight. But an optimizer does not overwrite the weights independently of their current values. Its first update is

$$
W_1=W_0-\eta\left.\frac{\partial L}{\partial W}\right|_{W=W_0}.
$$

The gradient is evaluated using the activations produced by $W_0$. The next gradient depends on $W_1$, which depends on the previous gradient, and so on. Initialization therefore affects the entire optimization trajectory:

```text
initial weights
      ↓
initial activations
      ↓
initial gradients
      ↓
first update
      ↓
all later states
```

If the initial state saturates every sigmoid, kills every ReLU, or produces enormous intermediate values, the optimizer receives poor information for making its very first move. It cannot simply promise to repair the weights later, because the route to those later weights is built from gradients computed now.

In a deep network the problem compounds. If every layer multiplies a signal's variance by $0.5$, then after 100 layers the multiplier is $0.5^{100}$. If every layer multiplies it by $1.5$, the multiplier becomes $1.5^{100}$. A modest per-layer distortion can therefore produce an unusable network before training has a chance to correct it.

## Why not initialize every weight to zero?

Suppose a hidden layer has two neurons:

$$
h_1=\phi(w_1^Tx+b_1),
\qquad
h_2=\phi(w_2^Tx+b_2).
$$

If we initialize

$$
w_1=w_2=0,
\qquad
b_1=b_2=0,
$$

then the two neurons produce exactly the same output for every input. Because the network treats them identically, they also receive identical gradient updates:

$$
\frac{\partial L}{\partial w_1}
=
\frac{\partial L}{\partial w_2}.
$$

After one update, their weights are still equal. The same remains true after every later update. A layer with 100 identical neurons therefore behaves like repeated copies of one neuron instead of learning 100 different features.

This is the **symmetry problem**.

```text
same initialization
       ↓
same activation
       ↓
same gradient
       ↓
same update
       ↓
neurons remain identical
```

Random weights break this permutation symmetry. One neuron may start slightly more responsive to one direction in the input, another neuron to a different direction, and gradient descent can develop those differences into distinct features.

### With ReLU, zero is even more troublesome

For ReLU,

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

Deep-learning libraries conventionally use a derivative of zero at $z=0$. If a ReLU layer begins with zero weights and biases, every pre-activation is zero and its local gradient is zero. The hidden units have no useful route away from that state.

With sigmoid, a zero pre-activation produces $\sigma(0)=0.5$ rather than zero, but all hidden units still behave identically. The exact failure mode changes; the symmetry problem does not.

### A useful caveat: zero initialization is not always wrong

Zero initialization is safe in some narrower settings:

- A single logistic-regression model can start at zero because it does not contain a hidden layer of interchangeable neurons.
- Biases in a neural network are commonly initialized to zero because random weights have already broken the symmetry between neurons.
- Some residual architectures intentionally initialize the *last operation of a residual branch* to zero. The skip connection still carries a nonzero signal, so this is not equivalent to setting the whole network to zero.

The rule is therefore not “no parameter may equal zero.” The rule is that trainable units that should learn different features must not begin in a symmetry that gradient descent cannot break.

## Why not use arbitrary random weights?

Randomness solves symmetry, but not signal scale.

Imagine sampling every weight independently from a standard normal distribution, $w_{ij}\sim\mathcal{N}(0,1)$. If a neuron has 1,000 inputs, it adds 1,000 randomly weighted terms. Even when the individual inputs are moderate, the sum can have a very large variance. Repeating this across layers can make activations explode.

Making the random values extremely small creates the opposite problem. Each layer reduces the signal, and a deep network can drive both activations and gradients toward zero.

The three regimes look like this:

| Weight scale | Forward activations | Backward gradients | Likely result |
| --- | --- | --- | --- |
| Too small | Shrink with depth | Vanish | Early layers learn very slowly |
| Too large | Grow with depth | Explode or become unstable | Saturation, overflow, unstable updates |
| Well scaled | Remain comparable | Remain usable | Optimization starts in a trainable regime |

To choose the useful scale, we need to see how variance travels through a layer.

## Variance propagation through a linear layer

Take one pre-activation with $n_{\text{in}}$ inputs:

$$
z_j=\sum_{i=1}^{n_{\text{in}}}w_{ji}x_i.
$$

Assume, approximately, that:

- the weights and inputs are independent;
- the weights have mean zero;
- the terms are identically distributed;
- the bias is initially zero.

Then

$$
\operatorname{Var}(z_j)
=n_{\text{in}}\operatorname{Var}(w)\operatorname{Var}(x).
$$

Define

$$
\text{fan-in}=n_{\text{in}},
\qquad
\text{fan-out}=n_{\text{out}}.
$$

If we want the output of the linear operation to have roughly the same variance as its input, then

$$
n_{\text{in}}\operatorname{Var}(w)\approx1,
$$

so a natural forward-pass choice is

$$
\operatorname{Var}(w)\approx\frac{1}{n_{\text{in}}}.
$$

Equivalently, the standard deviation should scale like

$$
\operatorname{Std}(w)\approx\frac{1}{\sqrt{n_{\text{in}}}}.
$$

This inverse-square-root scaling is the central idea. A neuron with more inputs receives smaller individual weights so that their sum does not automatically become larger.

### A numerical example

For a width of $1{,}000$:

- standard-normal weights have variance $1$, so a linear layer multiplies activation variance by roughly $1{,}000$;
- weights with standard deviation $0.01$ have variance $10^{-4}$, so the layer multiplies variance by roughly $0.1$;
- weights with standard deviation $1/\sqrt{1000}\approx0.0316$ keep the multiplier near $1$.

The difference appears small at one layer and enormous after many layers. A variance multiplier of $0.1$ repeated ten times becomes $10^{-10}$.

## Do sigmoid and tanh solve the scaling problem?

Sigmoid and tanh bound their outputs, so they prevent the *activation values* from becoming arbitrarily large:

$$
0<\sigma(z)<1,
\qquad
-1<\tanh(z)<1.
$$

But bounding the output does not make a poorly scaled input harmless. When $|z|$ is large, both functions enter saturation and their derivatives approach zero:

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))\rightarrow0,
\qquad
\tanh'(z)=1-\tanh^2(z)\rightarrow0.
$$

The forward value stays finite while the backward signal vanishes. Sigmoid's derivative is at most $1/4$, so repeated sigmoid layers can shrink gradients even before extreme saturation.

ReLU has the opposite character on its positive side:

$$
\operatorname{ReLU}(z)=z,
\qquad
\operatorname{ReLU}'(z)=1
\quad\text{for }z>0.
$$

It does not cap a large positive activation, and its positive-side derivative does not automatically damp a large gradient. This avoids one source of vanishing gradients but makes sensible weight scaling especially important. Activation functions change the failure mode; they do not eliminate the need for initialization.

## The backward pass imposes another constraint

For a layer's input gradient, backpropagation contains a weighted sum over the layer's outputs:

$$
\delta_i^{(\ell-1)}
=
\phi'\!\left(z_i^{(\ell-1)}\right)
\sum_{j=1}^{n_{\text{out}}}w_{ji}^{(\ell)}\delta_j^{(\ell)}.
$$

Ignoring the activation derivative for a moment, preserving gradient variance suggests

$$
\operatorname{Var}(w)\approx\frac{1}{n_{\text{out}}}.
$$

Under assumptions similar to the forward-pass derivation, the more complete relation is

$$
\operatorname{Var}\!\left(\delta^{(\ell-1)}\right)
\approx
n_{\text{out}}\operatorname{Var}\!\left(W^{(\ell)}\right)
\mathbb{E}\!\left[\phi'\!\left(z^{(\ell-1)}\right)^2\right]
\operatorname{Var}\!\left(\delta^{(\ell)}\right).
$$

This equation shows exactly where the initial weight variance enters gradient stability. If the multiplicative factor is consistently greater than $1$, gradient variance can grow exponentially with depth. If it is consistently below $1$, gradient variance can vanish.

The precise claim is not “any large initial variance always causes unstable gradients.” What counts as large depends on fan-out, the activation derivative, width, depth, normalization, and the rest of the architecture. The problem is a variance that is too large *relative to those quantities*.

The forward pass points to fan-in; the backward pass points to fan-out. When the input and output widths differ, one weight variance cannot satisfy both equations exactly. Xavier initialization uses a compromise between them.

## Xavier initialization

Xavier initialization, also called **Glorot initialization**, is designed for activations that are approximately linear around zero, especially tanh and linear units. It uses

$$
\boxed{
\operatorname{Var}(w)=\frac{2}{n_{\text{in}}+n_{\text{out}}}
}
$$

to balance forward and backward signal propagation.

Two common implementations are:

### Xavier normal

$$
w_{ij}\sim
\mathcal{N}\!\left(
0,
\frac{2}{n_{\text{in}}+n_{\text{out}}}
\right).
$$

Its standard deviation is

$$
\sqrt{\frac{2}{n_{\text{in}}+n_{\text{out}}}}.
$$

### Xavier uniform

$$
w_{ij}\sim
\mathcal{U}\!\left[
-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},
+\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}
\right].
$$

The constant $6$ is not a different principle. A uniform random variable on $[-a,a]$ has variance $a^2/3$. Setting that variance to $2/(n_{\text{in}}+n_{\text{out}})$ gives the stated bounds.

### Why Xavier helps tanh and sigmoid

Tanh and sigmoid saturate when their inputs have large magnitudes. In those regions their derivatives are small, so gradients vanish. Xavier initialization tries to keep initial pre-activations within a moderate range rather than immediately pushing most units into saturation.

Tanh is zero-centered and fits the assumptions better than sigmoid. Xavier is still commonly associated with sigmoid networks, but initialization cannot remove sigmoid's nonzero mean or prevent saturation throughout a very deep unnormalized network.

## Why Xavier is not ideal for ReLU

ReLU discards every negative pre-activation:

$$
a=\max(0,z).
$$

If $z$ has a symmetric distribution around zero, approximately half its values become zero. More precisely, the second moment satisfies

$$
\mathbb{E}[a^2]\approx\frac{1}{2}\mathbb{E}[z^2].
$$

If we used the linear forward-pass scale $\operatorname{Var}(w)=1/n_{\text{in}}$, the second moment would be roughly halved at every ReLU. Signals would progressively shrink.

This derivation tracks second moments rather than exact centered variances because ReLU outputs have a positive mean. The distinction matters mathematically, but the practical conclusion is the same: ReLU requires a larger initial weight scale.

## He initialization

He initialization, also called **Kaiming initialization**, compensates for the fraction of the signal removed by ReLU. Its fan-in form uses

$$
\boxed{
\operatorname{Var}(w)=\frac{2}{n_{\text{in}}}
}
$$

for ReLU networks.

The factor of $2$ cancels the approximate factor of $1/2$ introduced by the activation:

$$
\underbrace{n_{\text{in}}}_{\text{summed inputs}}
\underbrace{\frac{2}{n_{\text{in}}}}_{\text{weight variance}}
\underbrace{\frac12}_{\text{ReLU}}
\approx1.
$$

### He normal

$$
w_{ij}\sim
\mathcal{N}\!\left(0,\frac{2}{n_{\text{in}}}\right),
$$

with standard deviation

$$
\sqrt{\frac{2}{n_{\text{in}}}}.
$$

### He uniform

$$
w_{ij}\sim
\mathcal{U}\!\left[
-\sqrt{\frac{6}{n_{\text{in}}}},
+\sqrt{\frac{6}{n_{\text{in}}}}
\right].
$$

Some libraries allow `fan_out` mode instead. Fan-in mode prioritizes forward activations; fan-out mode prioritizes backward gradients. In a constant-width network they are the same.

### Leaky ReLU

For

$$
\phi(z)=
\begin{cases}
z,&z\ge0,\\
\alpha z,&z<0,
\end{cases}
$$

the negative half is multiplied by $\alpha$ rather than removed. The corresponding scale is

$$
\operatorname{Var}(w)
=
\frac{2}{(1+\alpha^2)n_{\text{in}}}.
$$

At $\alpha=0$, this reduces to the ReLU formula.

## Xavier versus He at a glance

| Initialization | Typical activation | Weight variance | Main intuition |
| --- | --- | ---: | --- |
| Xavier/Glorot | Linear, tanh; sometimes sigmoid | $2/(n_{\text{in}}+n_{\text{out}})$ | Balance forward and backward scales |
| He/Kaiming | ReLU | $2/n_{\text{in}}$ | Compensate for ReLU removing the negative half |
| Generalized He | Leaky ReLU | $2/((1+\alpha^2)n_{\text{in}})$ | Account for the negative slope |

A useful default rule is:

```text
tanh or linear hidden layer  -> Xavier
ReLU or leaky ReLU layer     -> He/Kaiming
specialized architecture     -> follow its initialization recipe
```

The distribution may be normal or uniform. Matching its variance to the layer and activation is usually more important than choosing between those two shapes.

## What are fan-in and fan-out for a convolution?

For a dense layer with a weight matrix of shape

$$
n_{\text{out}}\times n_{\text{in}},
$$

the definitions are direct. For a two-dimensional convolution with kernel height $k_h$, kernel width $k_w$, $c_{\text{in}}$ input channels, and $c_{\text{out}}$ output channels,

$$
n_{\text{in}}=c_{\text{in}}k_hk_w,
\qquad
n_{\text{out}}=c_{\text{out}}k_hk_w.
$$

Grouped and transposed convolutions require more care because the effective connectivity and weight layout differ. Framework initialization utilities calculate fans from the tensor shape, but their documentation should be checked when weights are stored or used in a nonstandard orientation.

## PyTorch examples

PyTorch exposes these schemes through `torch.nn.init`:

```python
import torch.nn as nn


def init_relu_module(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)


model.apply(init_relu_module)
```

For a tanh layer:

```python
gain = nn.init.calculate_gain("tanh")
nn.init.xavier_uniform_(layer.weight, gain=gain)
nn.init.zeros_(layer.bias)
```

For leaky ReLU, the initializer needs the same negative slope as the activation:

```python
negative_slope = 0.1

nn.init.kaiming_normal_(
    layer.weight,
    a=negative_slope,
    mode="fan_in",
    nonlinearity="leaky_relu",
)
```

One subtle point: a framework's default layer initialization is not necessarily the textbook scheme you would guess from the activation used later. If initialization is important to an experiment, set it explicitly and record it in the configuration.

## Initialization and activation must be chosen together

The activation changes both the forward signal and the derivative used during backpropagation. That is why there cannot be one universally correct weight variance.

- **Sigmoid and tanh** can saturate, so keeping pre-activations near their responsive region matters.
- **ReLU** removes negative values, motivating the factor of $2$ in He initialization.
- **Leaky ReLU** retains part of the negative signal, so its slope changes the required gain.
- **GELU and SiLU** have smooth, input-dependent slopes. Xavier or Kaiming-like schemes are often used, but modern architectures usually rely on a tested model-level recipe rather than a one-line activation rule.
- **SELU** was designed with a particular initialization and architecture to produce self-normalizing behavior. Replacing pieces independently can invalidate that argument.

Initialization is therefore part of the architecture, not merely a preprocessing detail.

## Does normalization make initialization irrelevant?

Batch normalization, layer normalization, and residual connections make deep networks much easier to optimize, but they do not make initialization meaningless.

Normalization can control activation statistics at selected points. Residual paths give gradients shorter routes through the model. Neither guarantees that every branch, projection, attention score, or gradient begins at a healthy scale.

Modern Transformers illustrate this clearly. Their initialization recipes may account for:

- embedding and projection scales;
- the growth of variance across residual additions;
- pre-normalization versus post-normalization;
- the number of residual branches and total depth;
- special scaling of output projections.

For an established architecture, reproducing its complete initialization recipe is safer than applying “Xavier everywhere” or “He everywhere.”

## Common mistakes

### 1. Thinking Xavier and He are alternatives to randomness

They are random initializations. What they specify is the scale of the distribution.

### 2. Randomizing biases to break symmetry

Random weights already break symmetry. Zero biases are normally fine and are often preferable.

### 3. Looking only at the forward pass

A network can have reasonable activations and still have poorly scaled gradients. Fan-out and activation derivatives enter the backward calculation.

### 4. Applying one initializer to every parameter tensor

Normalization scales, normalization offsets, embeddings, recurrent matrices, and residual output projections may require different treatment. A blanket recursive rule can overwrite deliberate defaults.

### 5. Confusing initialization with a guarantee

The derivations rely on approximate independence, identical distributions, and simple activation statistics. Training immediately creates correlations and changes distributions. Initialization gives optimization a good starting regime; it does not guarantee convergence or good generalization.

### 6. Forgetting reproducibility

Random initialization means different seeds can produce different training trajectories. Setting and recording the random seed makes an experiment repeatable; it does not turn all weights into the same value.

## A practical debugging checklist

When a new model fails to train, inspect more than the loss:

1. Log activation means and standard deviations at several depths after the first forward pass.
2. Log gradient norms after the first backward pass.
3. Check the fraction of zero ReLU activations and watch for dead layers.
4. Verify the actual fan-in, fan-out, activation, and gain used by the initializer.
5. Confirm that a custom initialization pass did not overwrite normalization or special residual parameters.
6. Compare a few random seeds before concluding that an architecture is broken.

If activations shrink monotonically with depth, the initial scale may be too small. If they grow rapidly, saturate, or become non-finite, it may be too large. Similar trends in gradient norms reveal trouble in the backward pass.

## The central idea

Weight initialization solves two different problems:

1. **Randomness breaks symmetry**, allowing neurons in the same layer to learn different features.
2. **Careful scaling preserves signal**, keeping activations and gradients usable across depth.

Xavier initialization chooses a variance that balances fan-in and fan-out for approximately linear, tanh-like propagation:

$$
\operatorname{Var}(w)=\frac{2}{n_{\text{in}}+n_{\text{out}}}.
$$

He initialization increases the fan-in scale to compensate for ReLU:

$$
\operatorname{Var}(w)=\frac{2}{n_{\text{in}}}.
$$

The broader principle is more important than either formula:

> Initialize different neurons differently, but scale those differences so that information can travel through the network in both directions.

## References

- Xavier Glorot and Yoshua Bengio, [Understanding the Difficulty of Training Deep Feedforward Neural Networks](https://proceedings.mlr.press/v9/glorot10a.html), AISTATS 2010.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html), ICCV 2015.
- Andrew M. Saxe, James L. McClelland, and Surya Ganguli, [Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks](https://arxiv.org/abs/1312.6120), ICLR 2014.
- PyTorch, [`torch.nn.init` documentation](https://docs.pytorch.org/docs/stable/nn.init.html).
