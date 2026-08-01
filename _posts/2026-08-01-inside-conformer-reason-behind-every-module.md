---
title: "Inside Conformer: The Reason Behind Every Module"
date: 2026-08-01 11:00:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [conformer, glu, pointwise-convolution, depthwise-convolution, self-attention, speech-recognition, residual-connections]
math: true
toc: true
comments: true
published: true
permalink: /posts/inside-conformer-reason-behind-every-module/
description: "A tensor-by-tensor explanation of the Conformer block, including its Macaron feed-forward layers, attention, pointwise convolution, GLU, depthwise convolution, normalization, and residual paths."
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

The list becomes much easier to remember when every component is treated as an answer to a specific problem.

- How should information at one time step be transformed? Use a feed-forward network.
- How should distant positions communicate? Use self-attention.
- How should nearby acoustic patterns be modeled efficiently? Use temporal convolution.
- How should channels communicate before and after that local operation? Use pointwise convolutions.
- How should the model decide which features to pass into the local operator? Use a GLU.
- How can a deep stack modify representations without repeatedly destroying them? Use residual connections and normalization.

[Part 1](/posts/cnn-vs-transformer-why-conformer-needs-both/) explained why speech benefits from both convolution and self-attention. This article follows one representation through the complete [original Conformer block](https://arxiv.org/abs/2005.08100) and separates three kinds of explanation:

1. **What the paper says:** the documented design or motivation.
2. **What the experiment supports:** the result of an ablation or comparison.
3. **What helps us understand it:** an intuition that should not be mistaken for proof.

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

Every major module preserves the model dimension, which makes residual addition possible. If the input has shape

$$
X\in\mathbb{R}^{T\times D},
$$

then the output of the entire block also has shape $T\times D$.

We will use $T=500$ and $D=512$ as a running example.

## Why begin and end with feed-forward modules?

A Transformer feed-forward network applies the same small multilayer perceptron independently at every time step:

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

Attention mixes information across positions. The FFN transforms the channel representation at each position. Both are required because moving information and processing information are different jobs.

### Why two FFNs?

Conformer uses a Macaron-style structure: one FFN before attention and convolution, and another afterward. This design was inspired by [Macaron-Net](https://arxiv.org/abs/1906.02762), which interprets the Transformer-like block through a numerical integration perspective.

A useful non-technical intuition is that the first FFN prepares each position's features before contextual mixing, while the second processes the contextualized result. That is an interpretation, not the main evidence for the choice.

The stronger evidence is empirical. The Conformer authors compared the paired Macaron FFNs with a single FFN having the same parameter count and found that the paired design performed better.

### Why multiply each FFN by one half?

The two FFNs together play the role occupied by one full residual FFN update in a standard Transformer block. Conformer therefore uses half-step residual weights:

$$
x\leftarrow x+\frac12\operatorname{FFN}(x).
$$

This scaling comes from the Macaron design. It prevents the pair from being treated as two unrelated full-strength replacements. The Conformer paper also compared full-step and half-step variants rather than choosing the coefficient only by intuition.

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

The attention path uses pre-normalization and a residual connection:

$$
x' = x + \operatorname{MHSA}(\operatorname{LayerNorm}(x)).
$$

This lets the attention module propose a contextual update without being responsible for reconstructing the complete representation from scratch.

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

Many implementations store a sequence for `Conv1d` as `batch × channels × time` rather than `batch × time × channels`. That transpose changes the memory layout expected by the API, not the conceptual roles of $T$ and $D$.

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

## GLU: a learned gate without recurrence

The pointwise output $z_t\in\mathbb{R}^{2D}$ is split along its channel dimension:

$$
z_t=[A_t,B_t],
$$

where $A_t,B_t\in\mathbb{R}^{D}$. The Gated Linear Unit computes

$$
\operatorname{GLU}(z_t)
=A_t\odot\sigma(B_t).
$$

The sigmoid maps every element of $B_t$ into $(0,1)$. Element by element:

```text
gate near 1  -> pass most of the candidate feature
gate near 0  -> suppress most of the candidate feature
middle value -> pass a scaled amount
```

For one scalar feature, if $A=5$:

$$
\begin{aligned}
\sigma(B)=0.95 &\quad\Rightarrow\quad y=4.75,\\
\sigma(B)=0.05 &\quad\Rightarrow\quad y=0.25.
\end{aligned}
$$

The gate is not a hand-written confidence score and does not come from a separate labeling process. Both $A$ and $B$ are learned transformations of the current input.

### Why use a GLU instead of only ReLU?

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

### GLU is not GRU

The similar names hide two different mechanisms.

| | GLU | GRU |
| --- | --- | --- |
| Full name | Gated Linear Unit | Gated Recurrent Unit |
| Has recurrent hidden state? | No | Yes |
| Depends on the previous time step's state? | No | Yes |
| Core idea | Gate candidate features | Update and reset recurrent memory |
| Parallel across a known sequence? | Yes | Recurrence imposes sequential dependence |
| Used in the original Conformer convolution block? | Yes | No |

A GRU computes reset and update gates involving both the current input $x_t$ and previous state $h_{t-1}$. A GLU simply gates one feed-forward branch with another:

$$
A\odot\sigma(B).
$$

The gate idea is related, but GLU removes recurrence.

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

## Why BatchNorm after the depthwise convolution?

The Conformer paper places BatchNorm immediately after the depthwise convolution and says it helps train deep models.

For temporal convolution, BatchNorm normalizes each channel using statistics accumulated across the batch and temporal positions. This differs from LayerNorm, which normalizes the feature vector within each individual time step.

That distinction matters:

- LayerNorm is convenient around attention and feed-forward modules and does not depend on batch-level statistics during inference.
- BatchNorm is used inside Conformer's convolutional path, following established CNN practice.

The safest historical statement is that this was the normalization choice validated in the original architecture. It should not be turned into a universal rule that every convolutional speech model must use BatchNorm. Streaming constraints, small batches, and later Conformer variants can motivate alternatives.

## Why Swish follows normalization

Conformer uses the Swish activation, also called SiLU:

$$
\operatorname{Swish}(x)=x\sigma(x).
$$

Unlike ReLU, it is smooth and permits small negative outputs. The paper reports that replacing Swish with ReLU made convergence slower in its experiments.

This is a good example of the evidence hierarchy:

- **Fact:** the block uses Swish.
- **Experimental observation:** Swish converged faster than ReLU in the reported setup.
- **Possible intuition:** smooth gating may make optimization easier.
- **Overclaim to avoid:** Swish is always better than ReLU in every architecture.

## Final pointwise convolution: let channels communicate again

Depthwise convolution deliberately isolates channels. Before the module returns its result, a second pointwise convolution mixes them:

$$
y_t=W_{mathrm{out}}x_t+b_{mathrm{out}},
\qquad
W_{mathrm{out}}\in\mathbb{R}^{D\times D}.
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

## The residual path uses addition, not concatenation

After dropout, the convolutional update is added to the module input:

$$
y=x+\operatorname{ConvModule}(x).
$$

This is elementwise addition. It is not concatenation.

If $x$ has shape $500\times512$, both branches must return $500\times512$, and their sum remains $500\times512$:

```text
x                     500 × 512
ConvModule(x)         500 × 512
-------------------------------- +
y                     500 × 512
```

Concatenation would instead produce $500\times1024$ along the channel dimension and would require an additional projection to return to the model width.

### Does adding the original input undo the convolution?

No. The convolutional branch is not forced to replace the representation. It learns a correction:

$$
y=x+f(x).
$$

If the current representation is already useful, $f(x)$ can be small. If local evidence demands a substantial revision, the learned update can be larger. Addition also does not imply a 50-50 average; there is no division by two in this residual connection.

Residual paths improve gradient flow and make deep stacks easier to optimize. Conceptually, each module can focus on what it should change rather than reconstructing everything that should remain unchanged.

## Why normalization appears in several places

It is easy to see “LayerNorm, BatchNorm, final LayerNorm” and assume they are redundant. They act at different boundaries.

- **Pre-LayerNorm around submodules:** prepares each time step's feature vector before a residual transformation.
- **BatchNorm inside the convolution module:** normalizes channel activations after temporal convolution using convolution-style statistics.
- **Final LayerNorm:** normalizes the result of the complete Conformer block.

Normalization is part of the optimization design, not the mechanism that creates local or global context. It helps keep the scale and distribution of intermediate representations manageable as many residual blocks are stacked.

## What the authors actually tested

A believable architecture explanation should not invent a perfect intention for every line of code. The original paper provides several useful empirical anchors:

### Convolution mattered most

In the paper's comparison against a Transformer-style block, removing the convolution sub-block caused the largest degradation among the tested architectural removals. This supports the central claim that explicit local modeling adds something important to attention for speech recognition.

### Module order mattered

The authors tested combinations of convolution and Transformer modules. Convolution placed after self-attention performed best among their evaluated arrangements. That is stronger evidence than the attractive but retrospective phrase “attention first gives convolution context-rich features.”

### Macaron FFNs helped

Two half-step feed-forward modules surrounding attention and convolution performed better than one equally parameterized FFN in the reported ablation.

### Kernel size had a useful middle range

The paper swept depthwise kernel sizes $3$, $7$, $17$, $32$, and $65$. Performance improved toward the intermediate larger kernels, then worsened at $65$. More local context helped, but an indefinitely wider kernel was not automatically better.

### Attention heads had a task-dependent optimum

Increasing the number of heads helped only up to a point in the reported configuration. More heads divide a fixed model dimension into smaller subspaces, so “more” is not a free improvement.

These ablations explain the published design under one experimental setting. They are evidence, not eternal architectural laws.

## A compact mental model

When reading the block, attach one question to every operation:

| Module | Question it answers |
| --- | --- |
| First half-step FFN | How should each position's features be prepared? |
| Self-attention | Which distant or nearby positions matter for this content? |
| First pointwise convolution | Which channel combinations should become candidates and gates? |
| GLU | Which candidate features should pass? |
| Depthwise convolution | How does each selected feature evolve locally through time? |
| BatchNorm and Swish | How should the local path be normalized and transformed? |
| Final pointwise convolution | How should the locally processed channels be recombined? |
| Second half-step FFN | How should each contextualized position be transformed before leaving? |
| Residual paths | What should each module change while preserving useful information? |

The architecture then stops looking like a collection of fashionable layers:

```text
per-position transformation
          ↓
global content-based interaction
          ↓
gated local temporal modeling
          ↓
per-position transformation
```

That is the central design logic of Conformer. Attention and convolution handle different structures, while GLU, pointwise projections, normalization, and residual paths make their collaboration expressive and trainable.

## References

- Gulati et al., [*Conformer: Convolution-augmented Transformer for Speech Recognition*](https://arxiv.org/abs/2005.08100), 2020.
- Dauphin et al., [*Language Modeling with Gated Convolutional Networks*](https://arxiv.org/abs/1612.08083), 2016/2017.
- Lu et al., [*Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View*](https://arxiv.org/abs/1906.02762), 2019.
- Dai et al., [*Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*](https://arxiv.org/abs/1901.02860), 2019.
- Ramachandran et al., [*Searching for Activation Functions*](https://arxiv.org/abs/1710.05941), 2017.
