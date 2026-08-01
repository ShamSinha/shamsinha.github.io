---
title: "CNN vs. Transformer: Why Conformer Needs Both"
date: 2026-08-01 10:00:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [cnn, transformer, conformer, self-attention, speech-recognition, inductive-bias, deep-learning]
math: true
toc: true
comments: true
published: true
permalink: /posts/cnn-vs-transformer-why-conformer-needs-both/
description: "A careful comparison of convolution and self-attention, and why Conformer combines local temporal modeling with global content-based interactions for speech recognition."
---

If self-attention can connect every position to every other position, why would a modern speech model still need a convolution?

That question leads directly to the idea behind [Conformer](https://arxiv.org/abs/2005.08100). Speech has structure at more than one scale. A few nearby frames contain the acoustic evidence for a sound, while words and sentences create dependencies across much longer spans. Convolution and self-attention approach those relationships differently:

```text
Convolution     -> local structure is built into the operation
Self-attention  -> relationships are chosen dynamically from the input
```

The point is not that one mechanism is universally superior. Each gives the model a different way to move and combine information. Conformer works because it makes those mechanisms cooperate.

This first article develops that argument. [Part 2](/posts/inside-conformer-reason-behind-every-module/) will open the Conformer block and ask why every component is there.

## Begin with the shape of speech

An automatic speech-recognition system rarely sends raw pressure measurements directly into its main encoder. A common front end converts an utterance into acoustic features such as log-Mel filterbanks. After subsampling and projection, we can describe the encoder input as

$$
X\in\mathbb{R}^{T\times D},
$$

where:

- $T$ is the number of time steps;
- $D$ is the feature or channel dimension.

For example, $X$ might have shape $500\times512$: five hundred time steps, each represented by a 512-dimensional vector.

This representation contains at least two kinds of dependencies.

### Local acoustic structure

Neighboring speech frames are highly correlated. Phonetic cues, formant movement, onsets, offsets, and transitions unfold over short intervals. If one frame contains part of a vowel, nearby frames are unusually informative about the rest of it.

### Longer-range context

The identity of a sound can also depend on a distant word, the broader utterance, or a speaker-level pattern. The useful context is not always confined to a fixed window.

A good speech encoder therefore needs both a microscope and a wide-angle lens.

## What convolution assumes

A one-dimensional convolution moves a kernel along time. With kernel width $k$, the output at position $t$ is computed from a local window around that position:

$$
y_t=\sum_{i\in\mathcal{N}(t)}W_i x_{t+i}+b.
$$

The same weights are reused at every time position. The kernel does not need to relearn what a short transition looks like at the beginning, middle, and end of an utterance.

This gives convolution two important inductive biases:

1. **Locality:** nearby positions deserve special attention.
2. **Translation equivariance:** the same pattern should be recognizable wherever it occurs.

An inductive bias is not merely a limitation. It is prior knowledge encoded in the architecture. When the prior matches the data, the model does not need to rediscover that structure entirely from examples.

### How a convolution obtains wider context

A single small kernel sees only a neighborhood. Wider context emerges by stacking layers, enlarging kernels, using dilation, or downsampling:

```text
Layer 1: short local pattern
   ↓
Layer 2: pattern of local patterns
   ↓
Layer 3: wider temporal structure
```

This hierarchy is useful, but information between far-apart positions must travel through multiple operations. The path becomes longer as the distance grows unless the architecture deliberately expands the receptive field.

## What self-attention assumes

Self-attention takes a different approach. From the same input sequence it constructs queries, keys, and values:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

Scaled dot-product attention is

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

For each position, the model computes how strongly it should combine information from other positions. In unrestricted self-attention, a time step can interact directly with every other time step in a single layer.

The important difference is that the mixing weights depend on the input.

```text
Utterance A: frame 80 may rely heavily on frame 73
Utterance B: frame 80 may rely heavily on frame 25
```

A convolution kernel is learned, but once learned its weights are applied in the same pattern at every location. Attention is also learned, but its actual routing pattern changes from example to example.

### Why the short path matters

The original [Transformer paper](https://arxiv.org/abs/1706.03762) compared architectures using three considerations:

- per-layer computational complexity;
- how much computation can be parallelized;
- the path length between dependent positions.

Self-attention connects any pair of positions through a constant number of sequential operations. A local convolution needs multiple stacked layers to connect positions beyond one receptive field. A recurrent model must also process its state sequentially through time.

That short path is a powerful property when a task contains distant dependencies.

## CNN and Transformer are different kinds of mixers

The comparison becomes clearer if we describe both operations as information mixers.

| Question | Convolution | Self-attention |
| --- | --- | --- |
| Which positions interact directly? | A fixed local neighborhood | Potentially every position |
| Are mixing weights input-dependent? | No; the learned kernel is reused | Yes; attention changes with the input |
| Is locality built in? | Yes | Not in vanilla global attention |
| How does distant information meet? | Through stacked or widened receptive fields | Directly within one layer |
| Main sequence-length concern | Grows roughly linearly for a fixed kernel | Vanilla attention forms a $T\times T$ matrix |
| Useful prior | Nearby patterns repeat across time | Relevant positions may be anywhere |

This table describes architectural tendencies, not a contest with one permanent winner. CNNs can obtain global context. Transformers can learn local attention patterns. Modern systems also modify both families with dilation, global pooling, local attention, sparse attention, and hierarchical representations.

## If attention can learn locality, why add convolution?

It can. A self-attention head is free to place most of its weight on neighboring frames.

But the ability to learn a behavior is different from having that behavior built into the operation. Global self-attention begins with a broad search space: every position may interact with every other position. Convolution starts with the prior that a compact neighborhood matters and applies the same detector throughout the sequence.

For speech, that prior is sensible. Local acoustic patterns occur everywhere, and their relative timing matters. A convolution gives the model an efficient specialist for that job while attention remains free to model content-dependent relationships over wider ranges.

The division is not absolute:

```text
Attention may learn local and global relationships.
Convolution may contribute to increasingly broad context when layers are stacked.
```

The useful claim is narrower: **each operation makes one kind of relationship especially natural.**

## Fixed computation versus dynamic routing

Suppose the same learned convolution processes two utterances. At each time step it applies the same kernel coefficients to the same relative offsets. The activations change, but the local routing template does not.

Self-attention constructs its weights from the current sequence. A head may behave differently when it encounters silence, a repeated phrase, or an ambiguous acoustic segment. This content-dependent routing gives attention considerable flexibility.

That flexibility has a cost. For a sequence of length $T$, unrestricted attention materializes pairwise scores with quadratic size $O(T^2)$. A fixed-width temporal convolution grows roughly linearly with $T$. The practical trade-off therefore includes representation, data, memory, hardware, and latency—not only theoretical expressiveness.

## Strong and weak inductive bias

CNNs are often described as having a stronger inductive bias than Transformers. This statement is useful if interpreted carefully.

A strong matching bias can improve data efficiency because the model begins with helpful assumptions. A weaker bias gives the model more freedom, but the desired structure may need to be learned from more data or supplied through other design decisions.

The original Vision Transformer work provides a useful example. A pure Transformer performed very well on image classification when pretrained on large datasets, while the paper also noted that its weaker image-specific inductive biases affected performance in smaller-data regimes. That does not prove a universal law that Transformers scale forever while CNNs plateau. It shows that architecture, data scale, pretraining, and optimization interact.

The safer conclusion is:

> Convolution spends part of its capacity budget on assumptions about locality and repeated patterns. Self-attention spends more of its capacity learning which positions should interact for the current input.

## Claims that should not be mixed with the architecture comparison

Several ideas are frequently attributed to Transformers even though they are not exclusive to them.

### Self-supervision is a training objective

Masked prediction, contrastive learning, next-token prediction, and reconstruction are training objectives. Transformers have benefited enormously from them, but CNNs can also be trained with self-supervision. The success of foundation models comes from an interaction among architecture, objectives, data, compute, and optimization.

### Pooling is not mandatory in every CNN

Some CNNs downsample aggressively; others preserve high-resolution features or use carefully designed multiscale paths. Similarly, Transformers may reduce information through tokenization, patch embedding, subsampling, or token merging. “CNNs discard information while Transformers preserve it” is too broad.

### Tokenization is an interface, not a unique advantage

A Transformer commonly receives word pieces, image patches, or acoustic embeddings rather than raw sensor values. CNN pipelines also learn feature hierarchies before later processing. The important question is what information the input representation exposes and discards, not which architecture owns the idea of preprocessing.

### Scaling is empirical, not an architectural guarantee

Transformers have demonstrated remarkable scaling in several domains. That observation should not become a theorem that any Transformer will beat any CNN when enlarged. Dataset, parameterization, training recipe, compute allocation, and task structure all matter.

## Conformer: assign each mechanism a suitable job

The Conformer authors framed their goal as combining the strengths of Transformers and CNNs in a parameter-efficient speech encoder. Their summary was direct:

- Transformers capture content-based global interactions well.
- CNNs exploit local features effectively.

A simplified Conformer block can be viewed as

```text
Input
  │
  ├─ Feed-forward transformation
  │
  ├─ Self-attention: content-dependent context
  │
  ├─ Convolution: local temporal refinement
  │
  └─ Feed-forward transformation
  ↓
Output
```

The resulting representation does not choose between local and global modeling. It is repeatedly updated by both.

The order also matters. The paper compared several ways of combining convolution and attention and found an advantage in placing the convolution module after self-attention. A useful intuition is that attention first makes each time step context-rich, after which convolution examines how those enriched features evolve locally. That intuition is helpful, while the stronger evidence is the authors' ablation: they actually compared arrangements rather than relying only on a story.

## What the evidence says

On LibriSpeech, the original Conformer reported strong word-error rates across 10M, 30M, and 118M parameter configurations. More useful for understanding the architecture, however, are its ablation studies:

- The convolution sub-block was the most important of the tested changes from a Transformer block.
- Macaron-style paired feed-forward modules performed better than a single feed-forward module with the same parameter count.
- Swish produced faster convergence in their experiments.
- Placing convolution after self-attention performed better than the alternative combinations they tested.
- Increasing the depthwise-convolution kernel helped up to an intermediate range, while the largest tested kernel degraded performance.

These results do not prove that every modern speech system must use the same block. They explain why the published Conformer was more than an arbitrary pile of familiar modules.

## The main idea

CNNs and Transformers are not separated by a simple “old versus new” boundary.

Convolution says:

> Nearby patterns repeat, so let us encode that knowledge directly.

Self-attention says:

> The important relationship depends on the current input, so let us compute it dynamically.

Speech benefits from both statements. Conformer turns that observation into an architecture: attention handles flexible global interaction, while convolution provides an efficient local temporal bias.

The next question is more detailed: why does the convolution module begin with a pointwise convolution, why does it double the channels, what exactly does a GLU gate, and why use a depthwise convolution afterward? [Part 2 opens the entire block and follows every tensor through it.](/posts/inside-conformer-reason-behind-every-module/)

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
- Gulati et al., [*Conformer: Convolution-augmented Transformer for Speech Recognition*](https://arxiv.org/abs/2005.08100), 2020.
- Dosovitskiy et al., [*An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929), 2020.

