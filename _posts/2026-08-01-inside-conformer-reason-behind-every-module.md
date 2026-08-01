---
title: "Inside Conformer: Why Its Modules Fit Together"
date: 2026-08-01 11:00:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [conformer, glu, pointwise-convolution, depthwise-convolution, self-attention, speech-recognition]
math: true
toc: true
comments: true
published: true
permalink: /posts/inside-conformer-reason-behind-every-module/
description: "A tensor-by-tensor explanation of the Conformer block, including its Macaron feed-forward layers, attention, pointwise convolution, GLU, depthwise convolution, and Swish activation."
---

The Conformer block can initially look like an architecture assembled from unrelated parts:

```text
Feed-forward
Self-attention
Pointwise convolution
GLU
Depthwise convolution
Batch normalization
Swish
Another pointwise convolution
Another feed-forward module
```

The block becomes easier to understand when we focus on its core modeling decisions:

- How should information at one time step be transformed? Use a feed-forward network.
- How should distant positions communicate? Use self-attention.
- How should nearby acoustic patterns be modeled efficiently? Use temporal convolution.
- How should channels communicate before and after that local operation? Use pointwise convolutions.
- How should the model decide which features to pass into the local operator? Use a GLU.

[Part 1](/posts/cnn-vs-transformer-why-conformer-needs-both/) explained why speech benefits from both convolution and self-attention. This article follows one representation through the complete [original Conformer block](https://arxiv.org/abs/2005.08100). Where the paper provides ablation evidence, I distinguish it from architectural intuition.

## The complete block

Let $x_i$ be the input to Conformer block $i$. The paper writes the block as

$$
\begin{aligned}
\tilde{x}_i
&=x_i+\frac{1}{2}\operatorname{FFN}(x_i),\\
x_i'
&=\tilde{x}_i+\operatorname{MHSA}(\tilde{x}_i),\\
x_i''
&=x_i'+\operatorname{Conv}(x_i'),\\
y_i
&=\operatorname{LayerNorm}\left(
x_i''+\frac{1}{2}\operatorname{FFN}(x_i'')
\right).
\end{aligned}
$$

In words:

```text
Input x
  │
  ├── ½ × Feed-Forward ── Add
  │
  ├── Self-Attention ──── Add
  │
  ├── Convolution ─────── Add
  │
  ├── ½ × Feed-Forward ── Add
  │
  └── Final LayerNorm
  ↓
Output y
```

Every major module maps the representation back to the model dimension. If the input has shape

$$
X\in\mathbb{R}^{T\times D},
$$

then the output of the entire block also has shape $T\times D$.

We will use $T=500$ and $D=512$ as a running example.

## Why begin and end with feed-forward modules?

Ignoring normalization and dropout, the core feed-forward transformation applies the same multilayer perceptron independently at every time step:

$$
\operatorname{FFN}(x_t)
=W_2\,\operatorname{Swish}(W_1x_t+b_1)+b_2.
$$

In the original Conformer, the first linear transformation expands the channel dimension by a factor of four and the second projects it back:

```text
one time step

512 features
    ↓ linear
2048 features
    ↓ Swish and dropout
    ↓ linear
512 features
```

The FFN does not mix different time positions. It asks a different question:

> Given everything currently stored at this time step, which new combinations of features should be constructed?

Attention mixes information across positions, while the FFN transforms the channel representation at each position. They perform complementary jobs.

### Why two FFNs?

Conformer uses a Macaron-style structure: one FFN before attention and convolution, and another afterward. This design was inspired by [Macaron-Net](https://arxiv.org/abs/1906.02762), which interprets the Transformer-like block through a numerical integration perspective.

A useful non-technical intuition is that the first FFN prepares each position's features before contextual mixing, while the second processes the contextualized result. That is an interpretation, not the main evidence for the choice.

The stronger evidence is empirical. The Conformer authors compared the paired Macaron FFNs with a single FFN having the same parameter count and found that the paired design performed better.

### Why multiply each FFN by one half?

Following the Macaron design, the two FFNs use half-step residual weights:

$$
x\leftarrow x+\frac12\operatorname{FFN}(x).
$$

Following Macaron-Net, each FFN contributes half a residual step. The Conformer paper also compared full-step and half-step variants.

## Why self-attention comes next

Multi-head self-attention lets every time step construct a content-dependent combination of other time steps. For one head,

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Multiple heads use different learned projections, allowing different interaction patterns to be represented simultaneously.

The attention module answers:

> Which parts of the utterance are relevant to the current position, based on their content?

The original Conformer also uses relative positional encoding from Transformer-XL. Content alone does not tell the model whether a related frame occurred immediately before, several steps after, or far away. Relative position information gives attention access to distance and direction. The paper motivates this choice as improving generalization across input lengths and robustness to variation in utterance length.

The attention submodule uses the pre-norm residual form:

$$
x' = x + \operatorname{MHSA}(\operatorname{LayerNorm}(x)).
$$

## Why convolution follows attention

Self-attention can learn local relationships, so the convolution module is not present because attention is mathematically incapable of seeing nearby frames. It is present because convolution supplies a direct and efficient local temporal bias.

The sequence can be understood as

```text
Self-attention
    ↓
Make each position aware of relevant wider context
    ↓
Convolution
    ↓
Refine how those context-rich features evolve locally
```

This story is helpful, but the experimental result matters more: the authors evaluated multiple ways to combine the two modules and reported that stacking convolution after self-attention worked best among the tested arrangements.

Now we can open the convolution module itself.

## The convolution module at a glance

For an input with shape $T\times D$, the module is

```text
Input                         500 × 512
  ↓ LayerNorm                 500 × 512
  ↓ Pointwise Conv            500 × 1024
  ↓ GLU                       500 × 512
  ↓ 1-D Depthwise Conv        500 × 512
  ↓ BatchNorm                 500 × 512
  ↓ Swish                     500 × 512
  ↓ Pointwise Conv            500 × 512
  ↓ Dropout                   500 × 512
  ↓ Add original input        500 × 512
Output                        500 × 512
```

For the normalization layers in this diagram, see [BatchNorm](/posts/deep-learning-field-notes/#batch-normalization) and [LayerNorm](/posts/deep-learning-field-notes/#layer-normalization).

Many implementations pass sequences to `Conv1d` as `batch × channels × time` rather than `batch × time × channels`. That transpose changes the axis order expected by the API, not the conceptual roles of $T$ and $D$.

## First pointwise convolution: mix channels and create gates

For a temporal sequence, a pointwise convolution is a one-dimensional convolution with kernel size one. It looks at one time position at a time but spans all input channels.

At time $t$,

$$
z_t=Wx_t+b.
$$

If $x_t\in\mathbb{R}^{512}$ and the layer produces 1024 output channels, then

$$
W\in\mathbb{R}^{1024\times512}.
$$

Each output channel can depend on every input channel. The operation therefore mixes features such as energy, pitch-related evidence, voicing, noise, and higher-level learned features—but it does not yet look at neighboring time steps.

```text
Pointwise convolution

across time:      does not mix neighbors
across channels:  mixes all channels
```

Why expand from $D$ to $2D$? Because the next GLU splits the result into two equal parts. One part carries candidate features; the other produces gates.

## GLU: learn which features should pass

The pointwise output $z_t\in\mathbb{R}^{2D}$ is split along its channel dimension:

$$
z_t=[A_t,B_t],
$$

where $A_t,B_t\in\mathbb{R}^{D}$. The Gated Linear Unit computes

$$
\operatorname{GLU}(z_t)
=A_t\odot\sigma(B_t).
$$

The sigmoid maps every element of $B_t$ into $(0,1)$:

```text
gate near 1  -> pass most of the candidate feature
gate near 0  -> suppress most of the candidate feature
```

Values in between pass a scaled amount. Both $A_t$ and $B_t$ are learned transformations of the current input, so the model learns the candidate features and their gates together.

### Why use a GLU rather than a standard activation?

ReLU applies a fixed rule:

$$
\operatorname{ReLU}(x)=\max(0,x).
$$

GLU creates an explicit multiplicative interaction between a candidate feature and an input-dependent gate. It can attenuate or preserve a feature without using zero as a universal threshold.

It also retains a comparatively direct gradient path through the candidate branch:

$$
\frac{\partial\left(A\odot\sigma(B)\right)}{\partial A}
=\sigma(B).
$$

When a gate is open, gradients can travel through $A$ without first passing through a saturating candidate activation such as `tanh`.

GLU was introduced by Dauphin et al. in [*Language Modeling with Gated Convolutional Networks*](https://arxiv.org/abs/1612.08083). That work sought a parallel convolutional alternative to recurrent language models and reported that GLUs improved optimization and accuracy compared with the alternative nonlinearities they evaluated. Conformer reused the gating mechanism inside its temporal convolution module; it did not invent GLU.

## Depthwise 1-D convolution: learn local evolution per channel

After GLU, the representation is back to $T\times D$. A depthwise temporal convolution now slides a separate kernel along time for every channel.

```text
Channel 1:  temporal kernel 1 slides over time
Channel 2:  temporal kernel 2 slides over time
...
Channel D:  temporal kernel D slides over time
```

For channel $c$,

$$
y_{t,c}
=\sum_{j=0}^{k-1}w_{j,c}\,x_{t+j,c}.
$$

Notice what is absent: there is no sum over other input channels. Each channel is processed independently.

This is almost the opposite of the pointwise operation:

| Operation | Mixes time neighbors? | Mixes channels? |
| --- | --- | --- |
| Pointwise convolution | No | Yes |
| Depthwise 1-D convolution | Yes | No |

The first pointwise convolution constructs and gates useful channel combinations. The depthwise convolution then studies how each resulting channel evolves through a local time window.

### Why depthwise rather than a full convolution?

Assume the input and output both have $D$ channels and the temporal kernel has width $k$. Ignoring bias terms, a full convolution needs

$$
kD^2
$$

parameters. A depthwise convolution needs only

$$
kD.
$$

With $D=512$ and $k=32$, one of the kernel sizes evaluated in the original paper:

$$
\begin{aligned}
\text{full temporal convolution}
&=32\times512\times512
\approx8.39\text{ million},\\
\text{depthwise temporal convolution}
&=32\times512
=16{,}384.
\end{aligned}
$$

The pointwise layers still contribute parameters: the first uses approximately $2D^2$ weights and the final one uses $D^2$. For $D=512$, those are about 524 thousand and 262 thousand weights respectively. The complete module is therefore not “almost free,” but its local temporal operator is far cheaper than a full $k\times D\times D$ convolution.

This factorization also separates two jobs cleanly:

```text
pointwise  -> channel mixing
depthwise  -> local temporal mixing
pointwise  -> channel mixing again
```

## Why Swish?

Conformer uses the Swish activation, also called SiLU:

$$
\operatorname{Swish}(x)=x\sigma(x).
$$

Unlike ReLU, it is smooth and permits small negative outputs. In the reported experiments, replacing Swish with ReLU slowed convergence; this is evidence for the published setup, not a universal claim that Swish is always better.

## Final pointwise convolution: let channels communicate again

Depthwise convolution deliberately isolates channels. Before the module returns its result, a second pointwise convolution mixes them:

$$
y_t=W_{\mathrm{out}}x_t+b_{\mathrm{out}},
\qquad
W_{\mathrm{out}}\in\mathbb{R}^{D\times D}.
$$

Its role is not to scan time again. It recombines the locally processed channels into the representation expected by the rest of the encoder.

The complete information flow is therefore

```text
Mix channels and create gates
              ↓
Select useful candidate features
              ↓
Model local time patterns per channel
              ↓
Mix the resulting channels
```

## What the ablations support

The paper provides several empirical anchors for the design:

| Choice | Reported result |
| --- | --- |
| Convolution sub-block | Removing it caused the largest degradation among the tested architectural removals |
| Module order | Convolution after self-attention performed best among the evaluated arrangements |
| Macaron FFNs | Two half-step FFNs outperformed one equally parameterized FFN |
| Depthwise kernel size | Performance improved through the intermediate larger kernels, then worsened at $65$ |
| Number of attention heads | Increasing the number helped only up to a point |

These results support the published configuration; they are not universal architectural laws. The complete block can be summarized as

```text
per-position transformation
          ↓
global content-based interaction
          ↓
gated local temporal modeling
          ↓
per-position transformation
```

Attention handles global content-based interaction, depthwise convolution handles local temporal evolution, and the pointwise projections plus GLU organize and gate the channel representation between them.

## References

- Gulati et al., [*Conformer: Convolution-augmented Transformer for Speech Recognition*](https://arxiv.org/abs/2005.08100), 2020.
- Dauphin et al., [*Language Modeling with Gated Convolutional Networks*](https://arxiv.org/abs/1612.08083), 2017.
- Lu et al., [*Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View*](https://arxiv.org/abs/1906.02762), 2019.
- Dai et al., [*Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*](https://arxiv.org/abs/1901.02860), 2019.
- Ramachandran et al., [*Searching for Activation Functions*](https://arxiv.org/abs/1710.05941), 2017.
