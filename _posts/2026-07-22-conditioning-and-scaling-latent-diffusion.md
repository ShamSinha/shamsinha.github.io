---
title: "Conditioning and Scaling Latent Diffusion Models"
date: 2026-07-22 10:00:00 +0530
categories: [Machine Learning, Generative AI]
tags: [diffusion-models, latent-diffusion, unet, conditioning, cross-attention, classifier-free-guidance, training-guide]
math: true
toc: true
comments: true
published: true
permalink: /posts/conditioning-and-scaling-latent-diffusion/
description: "A practical guide to matching U-Net capacity to latent geometry, injecting global and spatial conditions, and scaling latent diffusion training deliberately."
---

Once an unconditional latent-diffusion baseline works, two questions usually follow:

1. How large should the denoiser be?
2. Where should conditioning enter the network?

These choices interact. Latent resolution determines compute and memory; condition type determines which injection mechanism preserves the information the model needs. Adding depth, attention, and guidance at the same time makes failures difficult to attribute.

This article develops those changes in a controlled order. It assumes the VAE, normalization, prediction target, and noise schedule already satisfy the contract established in Part 1.

{% include diffusion-training-series.md %}

## Match capacity to latent geometry

Spatial latent diffusion usually uses a U-Net. Its capacity should be chosen from the **latent shape**, not the original image shape.

### What `hidden_dims` usually means

For a spatial U-Net:

```yaml
hidden_dims: [128, 256, 512]
```

usually describes channel width at successive resolutions. Wider levels increase feature capacity and memory use. More levels increase the receptive field and create a smaller bottleneck.

The length of the list matters as much as its values. A deeper U-Net can downsample too aggressively for a small latent, while a large latent can make every additional level expensive.

### Latent resolution changes the cost

Doubling latent height and width creates four times as many spatial positions. With channel widths held fixed, convolution cost grows roughly with that spatial increase. Full attention forms pairwise interactions between positions, so its attention matrix can grow by roughly sixteen times.

A larger latent may preserve more detail through the VAE, but the denoiser may require:

- Smaller batches.
- Fewer attention blocks.
- Gradient accumulation.
- Mixed precision.
- Memory-efficient attention.
- More training time.

Do not conclude that a larger latent is worse merely because its early loss is higher. It is a harder, higher-dimensional modeling problem. Compare quality at similar compute budgets.

### Does a deeper U-Net make samples sharper?

Depth can improve receptive field, global consistency, and condition use. It does not directly guarantee sharpness.

Smooth samples may instead come from:

- A lossy VAE decoder.
- Insufficient latent spatial resolution.
- Incorrect latent normalization.
- Undertraining.
- A prediction or sampler mismatch.
- Weak conditioning.
- Excessive regularization.

Increase depth when the baseline shows insufficient capacity, not as the first response to every soft sample.

## Match the condition path to the condition

Conditioning can be global, spatial, or both. Start with the simplest path that preserves the structure of the input.

| Condition type | Simple starting path | Why |
| --- | --- | --- |
| Class or category | Embedding added to time or residual features | The condition describes the whole sample |
| Mask, depth map, or layout | Input concatenation | Direct spatial correspondence is preserved |
| Dense spatial condition needing strong control | Multiscale feature injection | Each U-Net level receives aligned local features |
| Text or variable-length representation | Cross-attention | Image features can query condition tokens |

### Global labels

Class or category labels can be embedded and added to the time embedding or residual blocks. This is efficient when one condition describes the entire sample.

The simplest test is intervention: generate from the same initial noise while changing only the label. The output should change in the expected direction.

### Input concatenation

A spatial condition can be resized to latent resolution and concatenated with the noisy latent at the U-Net input.

This approach is easy to implement and retains direct spatial alignment. Its limitation is that deeper layers must remember the condition after repeated transformations.

### Multiscale feature injection

A condition encoder can create a feature pyramid whose resolutions match the U-Net. Injecting those features at multiple levels gives each stage direct access to spatial structure.

This is often stronger than input-only concatenation for masks, maps, and dense controls because local boundaries do not have to survive the entire encoder path unaided.

### Cross-attention

Cross-attention lets image features query condition tokens. The original [latent-diffusion work](https://arxiv.org/abs/2112.10752) uses it as a flexible interface for conditions such as text and bounding boxes.

For spatial masks or maps, tokenization may weaken exact pixel correspondence. Cross-attention is therefore often complementary to a convolutional feature pyramid rather than a replacement for it:

```text
multiscale features -> local spatial alignment
cross-attention     -> global condition context
```

Attention adds memory and optimization complexity. Establish a convolutional baseline before enabling it.

## Classifier-free guidance requires training support

[Classifier-free guidance](https://arxiv.org/abs/2207.12598) uses one network for both conditional and unconditional prediction. During training, the condition is randomly dropped so the unconditional branch is learned.

At generation time:

$$
\hat y_{\text{guided}}
=
\hat y_{\text{uncond}}
+w\left(\hat y_{\text{cond}}-\hat y_{\text{uncond}}\right).
$$

Here $w$ is the guidance scale under this convention. Larger values can improve condition adherence, but they can also reduce diversity, exaggerate artifacts, or move samples outside the training distribution.

A conditional model does not automatically support classifier-free guidance. The unconditional branch—usually created through condition dropout—must be trained explicitly.

## Keep spatial augmentation aligned

For spatially conditioned diffusion, apply exactly the same geometric transform to the image and its condition.

Plausible transforms may include:

- Horizontal or vertical flips.
- Small rotations.
- Mild translation and scaling.
- Small shear consistent with the data geometry.

Use interpolation appropriate to each tensor:

```text
continuous image -> bilinear or bicubic
categorical mask -> nearest neighbor
```

Apply augmentation only to the training set. Validation should remain stable. Strong but unrealistic transforms teach the model to represent examples that should not exist.

## Optimization choices that matter while scaling

### Learning rate

A larger U-Net or latent does not necessarily tolerate the learning rate used by a smaller baseline. AdamW is a common optimizer; begin conservatively, especially with limited data or complex conditioning.

If both training and validation loss rise rapidly, inspect learning rate and gradient norms before redesigning the architecture.

### Batch size and accumulation

Larger latents and attention often force smaller batches, which increases gradient variance. Gradient accumulation raises the effective batch size without increasing peak memory.

Be explicit about whether a reported step means a microbatch or an optimizer update. EMA and learning-rate schedules should usually advance after optimizer updates.

### Mixed precision

Mixed precision can reduce memory and improve throughput. Monitor NaNs, overflow, and instability in schedule calculations. Keep sensitive schedule buffers and reductions in adequate precision.

Compare against a short full-precision run before assuming an instability is architectural.

### Gradient clipping

Gradient clipping is a useful guardrail for occasional spikes. If gradients are clipped continuously, the underlying learning rate or loss scaling may be wrong.

When diagnosing instability, log gradient norms both before and after clipping.

### Dropout

Dropout can regularize a large model on limited data, but too much can produce underfit or soft samples. Start with zero or a small value and add it only when overfitting is visible.

Condition dropout used for classifier-free guidance is a separate decision from ordinary network dropout.

## Develop the model in stages

The safest development order is additive.

### Stage 1: unconditional baseline

Train the smallest useful model with a frozen VAE, fixed normalization, epsilon prediction, an established schedule, and EMA. This tests the latent pipeline without condition complexity.

### Stage 2: simple conditioning

Add one simple path:

- Label embeddings for global classes.
- Input concatenation for spatial conditions.

Hold the starting noise fixed and confirm that changing the condition changes the output.

### Stage 3: stronger conditioning

If the simple path is insufficient, add one of the following at a time:

- Multiscale features.
- Cross-attention.
- Classifier-free guidance.

Measure condition adherence separately from visual quality.

### Stage 4: objective and schedule ablations

Compare epsilon with velocity prediction, or linear with cosine scheduling, while holding the other major variables fixed.

### Stage 5: capacity and resolution

Increase U-Net width, depth, latent resolution, or training data only after baseline behavior is understood.

This order keeps failures attributable. Simultaneously changing the VAE, latent size, U-Net, prediction type, schedule, condition path, and sampler creates an experiment that teaches very little.

## A short scaling checklist

- U-Net depth matches the latent geometry.
- Peak memory is measured at the intended latent size.
- Time embeddings reach every relevant block.
- Changing the condition visibly changes fixed-noise output.
- Spatial images and conditions receive identical transforms.
- Attention or guidance is added only after a simpler baseline is measured.
- Optimizer steps, rather than microbatches, drive EMA and scheduling.
- One major variable changes per experiment.

The architecture and condition path are now defined. [Part 3](/posts/debugging-latent-diffusion-training/) explains how to evaluate the resulting model, interpret noisy validation, choose a sampler, and diagnose failures systematically.
