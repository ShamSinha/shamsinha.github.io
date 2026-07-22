---
title: "The Diffusion Model Is the Latent Prior: A Practical Training Guide"
date: 2026-07-22 09:00:00 +0530
categories: [Machine Learning, Generative AI]
tags: [diffusion-models, latent-diffusion, denoising, unet, vae, ddim, classifier-free-guidance, ema, training-guide]
math: true
toc: true
comments: true
published: true
permalink: /posts/training-latent-diffusion-models/
description: "A systems-first guide to training stable latent diffusion models, from latent normalization and prediction targets to conditioning, validation, sampling, and checkpoint resume."
---

*A systems-first guide to training stable latent diffusion models.*

A latent diffusion model learns the distribution of encoded data. The VAE defines **how an image is represented**; diffusion learns **which representations are plausible** and how to generate them from noise. This separation is the central idea behind [latent diffusion](https://arxiv.org/abs/2112.10752): move the expensive generative process into a compressed space while preserving enough information for the decoder to recover a good image.

This division of responsibility is useful:

```text
VAE       -> preserves and reconstructs information
diffusion -> learns the distribution of VAE latents
decoder   -> turns generated latents back into images
```

Calling diffusion *the latent prior* is useful, but it needs one qualification. A VAE may already be regularized toward a simple prior such as $\mathcal N(0,I)$. The diffusion model is not that analytic VAE prior. It is a learned generative model of the **aggregated encoded-data distribution**—the distribution of latents actually produced by the frozen encoder.

This division of responsibility creates several failure boundaries. Denoising loss can decrease while samples remain poor because the VAE is lossy, the latent scale is wrong, the condition is ignored, or the sampler does not match the training objective. Conversely, validation loss can fluctuate while fixed-seed samples improve because ordinary diffusion validation is stochastic.

The goal is therefore not merely to minimize denoising mean-squared error (MSE). It is to build a system whose latent representation, training target, conditioning path, sampler, and evaluation procedure all obey the same contract.

> If the VAE itself is still being designed, start with [The VAE Is the Image Codec](/posts/vae-training-for-latent-diffusion/). That companion guide covers the reconstruction ceiling, latent geometry, KL behavior, and codec selection in detail. This article begins after a usable VAE exists.

## The mental model in one minute

The complete system has three learned components with different jobs:

| Component | Learns | Cannot repair |
| --- | --- | --- |
| VAE encoder | A compact representation of an image | Information discarded by the bottleneck |
| Diffusion denoiser | The distribution and structure of those representations | A lossy or inconsistently scaled latent space |
| VAE decoder | A mapping from latent back to pixels | Missing latent detail or a poorly generated latent |

A productive debugging question is:

> **Which component owns the defect I can see?**

If real-image VAE reconstructions are already soft, fix the codec. If reconstructions are sharp but generated latents decode poorly, inspect diffusion. If fixed generated latents decode differently between scripts, inspect the integration contract.

## Start from the complete system

Before tuning diffusion, write down the full generation path. Every arrow is an interface that can fail:

```text
real image
    -> frozen encoder
    -> normalized latent z_0
    -> forward diffusion produces z_t
    -> denoiser predicts noise, velocity, or clean latent
    -> reverse sampler produces generated z_0
    -> latent normalization is inverted
    -> frozen decoder
    -> generated image
```

For conditional generation, include every condition explicitly:

```text
condition -> encoder, denoiser, decoder, or some combination
```

The exact contract—shape, scale, condition, objective, schedule, and checkpoint metadata—must remain stable across training, validation, standalone generation, and checkpoint resume.

Before a full run, verify:

- The VAE reconstructs real images at acceptable quality.
- The VAE is frozen and in evaluation mode.
- Encoder and decoder conditions are passed consistently.
- The latent shape matches the denoiser.
- Latent normalization is measured and reversible.
- A latent can be encoded, normalized, denormalized, and decoded without changing reconstruction.
- The diffusion objective and sampler use the same prediction convention.

Many apparent diffusion problems are integration problems at one of these boundaries. Treat this checklist as a preflight test, not as cleanup after an expensive run fails.

## Understand the forward process

Diffusion training constructs a noisy latent directly from a clean latent $z_0$:

$$
z_t
=
\sqrt{\bar\alpha_t}\,z_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

where:

- $t$ is a randomly sampled timestep.
- $\epsilon$ is standard Gaussian noise.
- $\bar\alpha_t$ controls how much clean signal remains.

The network receives $z_t$, $t$, and any condition. It learns a target from which the reverse process can estimate a cleaner latent. The original [DDPM formulation](https://arxiv.org/abs/2006.11239) connects this denoising objective to a variational bound and denoising score matching.

The important computational point is that training does **not** run the full noising chain one step at a time. A random $z_t$ can be sampled in one operation from $z_0$. A process with 1,000 defined timesteps still needs only one denoiser evaluation per example during an ordinary training step.

## Training timesteps and sampling steps are different

These two settings are often confused.

### Training timesteps

The training timestep count $T$ defines the discretized forward noise process. A common baseline is 1,000. During each training step, the model usually sees one randomly selected timestep for each sample.

Increasing $T$ does not mean each batch performs more network evaluations. It changes the set of noise levels and the schedule discretization.

### Sampling steps

Sampling steps are the number of denoiser evaluations used during generation.

A DDPM sampler may follow all training timesteps. A DDIM or other accelerated sampler can use a subset, such as 50, 100, or 250 steps.

| Setting | What it controls | Cost during one training batch |
| --- | --- | --- |
| Training timesteps | The noise levels in the forward process | Usually one sampled level per example |
| Sampling steps | Denoiser evaluations during generation | No effect unless the batch also logs generated samples |

More sampling steps can improve convergence up to diminishing returns. They cannot recover details discarded by the VAE or compensate for a poorly trained denoiser.

Validation image logging also performs reverse diffusion. It is not a separate visualization trick. If 16 images are logged with 400 DDIM steps, validation must run 6,400 denoiser evaluations just for those samples. Slow validation may therefore come from sample logging rather than validation loss computation.

## Choose a prediction objective

The three common prediction targets are noise, clean data, and velocity.

### Epsilon prediction

The network predicts the sampled noise, $\epsilon$.

This is the target used by the original DDPM experiments and remains an established baseline. It is easy to implement and reason about.

Its loss weighting across signal-to-noise ratios is not always ideal. Very noisy and nearly clean timesteps can contribute differently to optimization.

### Clean-latent prediction

The network predicts $z_0$ directly.

This makes the output intuitive, but errors can behave differently across timesteps. High-noise inputs make direct clean prediction difficult, and loss weighting becomes important.

### Velocity prediction

Velocity prediction combines noise and clean signal:

$$
v_t
=
\sqrt{\bar\alpha_t}\,\epsilon
-
\sqrt{1-\bar\alpha_t}\,z_0.
$$

This parameterization was developed as part of work on stable, low-step diffusion sampling in [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512). It can behave more evenly across noise levels in some setups, particularly when the sampler and loss weighting are designed for it.

It is not automatically better. A run that becomes unstable after switching to v-prediction may have an incorrect target conversion, a sampler mismatch, a schedule interaction, or hyperparameters inherited from an epsilon baseline.

When comparing objectives:

1. Keep the VAE and latent normalization fixed.
2. Verify target and reverse-process equations with unit tests.
3. Start a new run unless the checkpoint used the same prediction type.
4. Keep the schedule fixed for the first comparison.
5. Compare fixed-seed samples and timestep-bucket losses.

Never resume an epsilon checkpoint as a v-prediction run simply because the output shapes match. The learned meaning of the output is different.

## Choose a noise schedule carefully

The schedule determines how signal-to-noise ratio changes with timestep.

### Linear beta schedule

A linear beta schedule is a strong baseline. It is simple, widely implemented, and was used in the original DDPM setup. That history makes it a useful control experiment, not a universal optimum.

### Cosine schedule

A cosine schedule typically preserves usable signal for more of the trajectory and changes how training effort is distributed across noise levels. It was introduced as one of the practical improvements in [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). It can improve some datasets and latent scales, but should still be treated as an ablation.

Cosine plus v-prediction is a reasonable experiment, not a guaranteed upgrade. Changing both simultaneously removes the baseline needed to identify which change helped or hurt.

### Signal-to-noise weighting

Uniform timestep sampling does not imply equal learning value at every timestep. Very easy high-SNR examples or very noisy low-SNR examples may dominate gradients depending on the objective.

Methods such as SNR weighting, [Min-SNR weighting](https://arxiv.org/abs/2303.09556), or alternative timestep sampling can rebalance training. Min-SNR, for example, interprets timestep learning as a multitask problem with potentially conflicting optimization directions. Add a weighting scheme only after logging loss by timestep range; otherwise, it may solve a problem that has not been measured.

## Freeze and normalize the latent space

The VAE should normally remain frozen during diffusion training. Joint training causes the latent distribution to drift while the denoiser is trying to learn it, turning the target into a moving coordinate system.

Use:

```python
vae.eval()

for parameter in vae.parameters():
    parameter.requires_grad = False
```

Evaluation mode matters even when parameters are frozen because dropout or batch normalization can otherwise change encoding behavior.

### Decide between posterior means and samples

Encoding with the posterior mean $\mu$ creates a deterministic latent dataset. It removes encoder sampling noise and therefore gives diffusion a lower-variance target. Whether the decoded result is sharper is model-dependent and should be measured rather than assumed.

Encoding with posterior samples trains diffusion on the stochastic representation used during VAE training. It may better match the decoder's training distribution but can make the latent distribution harder to model.

Choose one and use it consistently for:

- Latent-statistics computation.
- Diffusion training.
- Validation.
- Generation.

### Choose and freeze a normalization convention

Latent channels may have different means and variances. For a custom VAE, channel-wise standardization is a reasonable default:

$$
z_{\text{norm},c}
=
\frac{z_c-\mu_c}{\sigma_c}.
$$

Invert this transform before decoding:

$$
z_c=z_{\text{norm},c}\sigma_c+\mu_c.
$$

Some established latent-diffusion systems instead use a single scalar scale factor. Either convention can work. What matters is that it is measured or deliberately chosen, inverted before decoding, and never changed silently.

Compute any statistics from the frozen VAE over the training split. Store the convention and its values in the diffusion checkpoint. Reuse them during validation, resume, and generation.

If normalization statistics change while denoiser weights remain fixed, the meaning of every latent coordinate changes. This can produce severe generation regressions without any architecture error.

## Match model capacity to latent geometry

Spatial latent diffusion usually uses a U-Net. Its capacity should be chosen from the latent shape, not the original image shape.

### Hidden dimensions

For a spatial U-Net:

```yaml
hidden_dims: [128, 256, 512]
```

usually describes channel width at successive resolutions. More levels increase receptive field and create a smaller bottleneck. Wider levels increase feature capacity and memory use.

The length of the list is as important as its values. A deeper U-Net may downsample too aggressively for a small latent or consume excessive memory for a large latent.

### Larger latent size changes diffusion cost

Doubling latent height and width creates four times as many spatial positions. Convolution cost grows roughly with that spatial increase when channel widths remain fixed. Full attention forms interactions between positions, so its attention matrix can grow by roughly sixteen times.

A larger latent may preserve more image detail through the VAE, but the denoiser may need:

- Smaller batches.
- Fewer attention blocks.
- Gradient accumulation.
- Mixed precision.
- Memory-efficient attention.
- More training time.

Do not conclude that a larger latent is worse because its early loss is higher. It may simply be a harder and higher-dimensional modeling problem. Compare sample quality at similar compute budgets.

### Does a deeper model make outputs sharper?

Depth can improve global consistency, receptive field, and conditioning use. It does not directly guarantee sharpness.

Smooth output may come from:

- A lossy VAE decoder.
- Insufficient latent spatial resolution.
- Poor latent normalization.
- Undertraining.
- A prediction or sampler mismatch.
- Weak conditioning.
- Excessive regularization.

Increase depth when the baseline lacks capacity, not as the first response to every smooth sample.

## Inject conditioning at the right places

Conditioning can be global, spatial, or both.

### Global labels

Class or category labels can be embedded and added to time embeddings or residual blocks. This is efficient when the condition describes the whole sample.

### Input concatenation

A spatial condition can be resized to latent resolution and concatenated with the noisy latent at the U-Net input.

This preserves direct spatial correspondence and is easy to implement. Its limitation is that deeper layers must remember condition details after repeated transformations.

### Multiscale feature injection

A condition encoder can create a feature pyramid that is injected at each U-Net resolution.

This is often stronger for dense spatial control because every level receives a condition at matching resolution. It preserves local boundaries better than input-only concatenation.

### Cross-attention

Cross-attention lets image features query condition tokens. It is useful when the condition contains global relationships or a variable-length representation. The original [latent-diffusion work](https://arxiv.org/abs/2112.10752) uses cross-attention as a flexible interface for inputs such as text and bounding boxes.

For spatial masks or maps, tokenization may reduce exact pixel correspondence. Cross-attention is often complementary to a convolutional condition pyramid rather than a replacement for it:

```text
multiscale features -> local spatial alignment
cross-attention     -> global condition context
```

Attention adds memory and optimization complexity. Establish a convolutional conditioning baseline before enabling it.

### Classifier-free guidance

[Classifier-free guidance](https://arxiv.org/abs/2207.12598) requires randomly dropping the condition during training so the same network learns conditional and unconditional predictions.

At generation time:

$$
\hat y_{\text{guided}}
=
\hat y_{\text{uncond}}
+w\left(\hat y_{\text{cond}}-\hat y_{\text{uncond}}\right).
$$

Here $w$ is the guidance scale under this convention. Higher guidance can improve condition adherence, but it can also reduce diversity, exaggerate artifacts, or push samples outside the training distribution.

Guidance is not available merely because a model is conditional. The unconditional branch—usually learned through condition dropout—must exist during training.

## Keep spatial augmentation aligned

For spatially conditioned diffusion, geometric transforms must be identical for image and condition.

Valid examples may include:

- Horizontal or vertical flipping.
- Small rotations.
- Mild translation and scaling.
- Small shear consistent with natural geometry.

Use interpolation appropriate to each tensor:

```text
continuous image -> bilinear or bicubic
categorical mask -> nearest neighbor
```

Apply augmentation only to the training set. Validation must remain stable.

An augmentation is useful only if it represents plausible variation. Strong unrealistic transforms teach the denoiser to model examples that should not exist.

## Optimization settings that matter

### Learning rate

Diffusion models are often sensitive to learning rate. A larger U-Net or larger latent does not necessarily tolerate the learning rate used by a smaller baseline.

AdamW is a common optimizer. Start conservatively, especially when training data is limited or conditioning is complex. If both training and validation losses rise rapidly, inspect learning rate and gradient norms before redesigning the architecture.

### Batch size and gradient accumulation

Small batches increase gradient variance. Gradient accumulation can increase effective batch size without increasing peak memory.

Be explicit about whether reported steps refer to optimizer updates or microbatches. EMA updates and learning-rate schedules should usually follow optimizer steps.

### Mixed precision

Mixed precision can significantly reduce memory and improve throughput. Monitor for NaNs, overflow, and instability in schedule calculations. Keep sensitive schedule buffers and reductions in adequate precision.

### Gradient clipping

Gradient clipping is a useful guardrail for occasional spikes, especially in deep conditional models. Continuous clipping at every step may indicate a learning-rate or loss-scaling problem.

Log gradient norms before and after clipping if instability is being diagnosed.

### Dropout

Dropout can regularize a large model on limited data, but too much may produce soft or underfit samples. Start with zero or a small value and add it only when overfitting is visible.

## Use EMA for generation

An exponential moving average (EMA) maintains a smoothed copy of denoiser parameters:

$$
\theta_{\text{EMA}}
\leftarrow
d\,\theta_{\text{EMA}}+(1-d)\,\theta.
$$

EMA samples are often more stable and visually coherent than samples from raw training weights. Values such as `0.999` or `0.9999` are common, but the number alone is incomplete: the effective smoothing depends on how often EMA is updated and how long the run lasts.

Important details:

- Update EMA after optimizer updates.
- Save EMA weights in checkpoints.
- Restore them during resume.
- State clearly whether validation samples use EMA.
- Do not compare raw-model samples from one run with EMA samples from another.

Training and validation denoising loss are usually computed with raw model weights, while logged images may use EMA. This is one reason sample quality and scalar loss do not always move together.

## Why validation loss fluctuates

Diffusion validation is stochastic unless noise and timesteps are fixed. The same clean latent can produce a different validation problem on every epoch.

Sources of variance include:

- Random Gaussian noise.
- Random timesteps.
- Different proportions of easy and hard noise levels.
- Small validation sets.
- Diverse conditions.
- Raw-model metrics combined with EMA samples.

Single-epoch spikes are therefore normal. Judge the moving trend rather than one point.

### Make validation more informative

Use two complementary protocols.

First, retain stochastic validation to estimate average denoising performance.

Second, build a deterministic diagnostic panel:

- Fixed validation examples.
- Fixed Gaussian noise.
- Fixed timestep set.
- Fixed generation seeds.
- Fixed conditions.

Also log loss by timestep bucket, for example:

```text
low noise:    t in [0, 0.25 T)
medium noise: t in [0.25 T, 0.75 T)
high noise:   t in [0.75 T, T)
```

The exact boundaries are not sacred. Their purpose is to reveal whether a schedule or objective change helps one part of the trajectory while hurting another.

## Loss and sample quality are related but not identical

Lower denoising MSE is useful, but it does not perfectly rank generated image quality—especially across runs with different parameterizations or loss weightings.

Reasons include:

- Different objectives produce losses on different scales.
- A small improvement at easy timesteps may dominate the average.
- EMA weights may generate better samples than raw weights used for loss.
- The VAE decoder can amplify or hide latent errors.
- Pixel sharpness and perceptual quality are not directly represented by noise MSE.

Track both scalar and visual diagnostics:

- Train and validation denoising loss.
- Loss by timestep bucket.
- Prediction and target standard deviation.
- Latent mean and standard deviation.
- Fixed-seed generated samples.
- Condition adherence.
- Diversity across seeds.
- Decoded-image metrics when ground truth pairing is meaningful.

## Should diffusion use perceptual loss?

The standard diffusion objective operates in noisy latent space. Adding an image-space perceptual loss requires estimating a clean latent, decoding it, and comparing image features. This is computationally expensive and can create unstable gradients across timesteps.

Before adding perceptual loss to diffusion, verify:

1. The VAE round-trip reconstruction is sharp.
2. Latent normalization is correct.
3. The denoising objective is implemented correctly.
4. Fixed samples improve with ordinary training.
5. The feature extractor is appropriate for the task.

At high-noise timesteps, converting a prediction into an estimate of $z_0$ can amplify error before the decoder even sees it. If an auxiliary image-space loss is explored, apply it carefully—often at low-noise timesteps or during late fine-tuning—and keep its weight small. A perceptual objective should refine a working model, not rescue a broken baseline.

## Understand the sampler

### DDPM

[DDPM](https://arxiv.org/abs/2006.11239) sampling follows a stochastic reverse process. It is faithful to the original formulation but often requires many denoiser evaluations.

### DDIM

[DDIM](https://arxiv.org/abs/2010.02502) defines a non-Markovian process with the same training objective as DDPM and can sample using a subset of timesteps. With `eta = 0`, sampling is deterministic for a fixed starting noise; increasing `eta` adds stochasticity.

Fewer DDIM steps improve speed but generally increase discretization error. More steps can help until the model, VAE, or numerical path becomes the limiting factor; improvement is not guaranteed to be monotonic for every schedule and checkpoint.

Use a small number of steps for frequent training previews and a larger, fixed setting for formal comparison. Otherwise, validation becomes unnecessarily slow and samples across epochs are not comparable.

### Sampling is part of generation and logging

When a training callback generates images for TensorBoard or another logger, it runs the same reverse process used by standalone generation. The logger is not training the model. It is periodically evaluating the current denoiser by sampling from noise.

## Overfit a tiny subset before a long run

Before spending days on full training, overfit one batch or a tiny dataset.

The goal is not good generalization or a publishable sample. The goal is to verify that:

- Loss decreases substantially.
- The model can fit the denoising problems generated from the small training subset.
- Conditioning affects output in the expected direction.
- Sampling and decoding work.
- EMA updates correctly.
- Checkpoints save and reload without changing fixed-seed results.

If a model cannot overfit a tiny subset, a full run is unlikely to solve the problem. This test catches objective, conditioning, normalization, and sampler bugs quickly.

## Use staged model development

### Stage 1: unconditional baseline

Train the smallest useful unconditional model with a frozen VAE, a fixed normalization convention, epsilon prediction, an established schedule, and EMA.

This tests the latent pipeline without conditioning complexity.

### Stage 2: simple conditioning

Add the simplest appropriate condition path:

- Label embedding for global classes.
- Input concatenation for spatial conditions.

Confirm that changing the condition changes generated output.

### Stage 3: stronger conditioning

If simple conditioning is insufficient, add multiscale injection, cross-attention, or classifier-free guidance one at a time.

Measure condition adherence separately from visual quality.

### Stage 4: objective and schedule ablations

Compare epsilon and velocity prediction, or linear and cosine schedules, with all other major variables fixed.

### Stage 5: capacity and resolution scaling

Increase U-Net depth, width, latent resolution, or training data only after the baseline behavior is understood.

This order keeps failures attributable. Simultaneously changing the VAE, latent size, U-Net, prediction type, schedule, and sampler creates an experiment whose result is difficult to learn from.

## Checkpointing and resume

A true resume should restore:

- Denoiser weights.
- Optimizer state.
- Learning-rate scheduler state.
- Epoch and global step.
- EMA weights.
- Latent normalization statistics.
- Diffusion objective and schedule metadata.

Use `last.ckpt` for continuing an interrupted run. A best checkpoint may come from an earlier epoch and is better suited to evaluation or intentional rollback.

Before resuming, verify that current configuration matches the checkpoint:

- VAE checkpoint and latent shape.
- Mean versus sampled encoding.
- Latent normalization mode and statistics.
- Prediction type.
- Noise schedule and timestep count.
- U-Net architecture.
- Conditioning paths and channel counts.

Some logging frameworks create a new version directory during resume even though training state continues correctly. That is a logging decision, not evidence that optimization restarted. Check restored epoch and global step.

## A general starting configuration

This is a conservative starting template, not a universal optimum. Field names are illustrative and should be mapped to the framework in use:

```yaml
vae:
  checkpoint: path/to/frozen_vae.ckpt
  latent_type: spatial
  use_mu: true
  latent_normalization: channel
  # Compute automatically, then save in the diffusion checkpoint.
  latent_channel_mean:
  latent_channel_std:

diffusion:
  timesteps: 1000
  beta_schedule: linear
  prediction_type: epsilon

model:
  architecture: unet
  hidden_dims: [128, 256, 512]
  time_embed_dim: 128
  dropout: 0.0
  use_cross_attention: false

optimizer:
  lr: 0.00005
  ema_decay: 0.999

trainer:
  precision: 32-true
  gradient_clip_val: 1.0
  save_last: true

generation:
  sampler: ddim
  ddim_steps: 100
  ddim_eta: 0.0
  use_ema: true
```

After stability is established, mixed precision, larger models, stronger conditioning, alternative prediction targets, or advanced loss weighting can be evaluated independently.

## Debug from the outside in

When samples look wrong, inspect the pipeline in this order:

1. **Decode real encodings.** If a real image does not survive the VAE round trip, diffusion does not own the defect.
2. **Test normalization round-trip parity.** Encoding, normalizing, inverting, and decoding should match an encode–decode pass without normalization.
3. **Inspect the training target.** Check shapes, ranges, target conversions, and loss by timestep before changing architecture.
4. **Overfit a tiny subset.** This exposes broken objectives, conditioning paths, and samplers much faster than a full run.
5. **Generate with fixed seeds.** Compare raw and EMA weights under identical sampler settings.
6. **Test the condition independently.** Swap, shuffle, or zero the condition and verify that the output responds.
7. **Only then scale the model.** Increase capacity, latent resolution, data, or training time after the baseline behaves coherently.

This order moves from the cheapest and most deterministic checks toward the most expensive explanations. It also prevents a larger U-Net from temporarily hiding an interface bug.

## Common failure modes

| Symptom | Likely causes | First checks |
| --- | --- | --- |
| Validation loss jumps between epochs | Random noise and timestep composition, small validation set | Moving average, fixed validation noise, timestep buckets |
| Validation loss rises rapidly from the start | Learning rate, normalization, objective implementation, schedule mismatch | Latent mean/std, gradient norms, tiny-subset overfit |
| Loss decreases but images remain waxy | VAE reconstruction ceiling, undertraining, sampler mismatch | Real-image VAE reconstruction, fixed EMA samples |
| Output ignores the condition | Weak injection, condition-data mismatch, overly aggressive condition dropout | Condition tensors, shuffled-condition ablation, dropout rate, multiscale features |
| Spatial layout follows condition but texture is poor | VAE detail loss, insufficient denoiser capacity, undertraining | VAE reconstruction, latent resolution, deeper feature blocks |
| v-prediction run is unstable | Target conversion or sampler mismatch, inherited hyperparameters | Unit tests, prediction metadata, fixed schedule comparison |
| Some generated channels dominate after decoding | Incorrect latent channel scaling | Saved means/stds, normalization inversion |
| Training resumes but samples suddenly change | VAE path or normalization stats changed, EMA not restored | Checkpoint metadata, EMA state, global step |
| Validation is extremely slow | Too many logged samples or sampling steps | Sample count, logging interval, DDIM steps |
| Samples are identical across seeds | Initial noise is accidentally reused, collapsed model, condition or guidance is too strong | Seeded noise tensors, diversity metrics, guidance scale |
| Samples are diverse but condition adherence is weak | Conditioning path too weak or guidance absent | Condition injection, classifier-free dropout, guidance |
| NaNs appear | Mixed precision overflow, extreme gradients, invalid schedule math | Full precision test, gradient norms, schedule buffers |
| Training loss is much lower than validation | Overfitting, split mismatch, augmentation gap | Fixed panels, data distributions, dropout or capacity |

## A diffusion readiness checklist

### VAE interface

- Real images reconstruct with acceptable fidelity.
- The VAE is frozen and in evaluation mode.
- Latent shape and channel count are verified.
- Encoder and decoder conditions are passed consistently.
- Mean versus sampled encoding is fixed.
- Latent normalization is reversible and checkpointed.

### Objective and schedule

- Forward noising is unit-tested.
- Prediction target is unit-tested.
- Reverse conversion matches prediction type.
- Training and sampling use compatible schedules.
- Loss is logged by timestep range.

### Architecture and conditioning

- U-Net depth matches latent geometry.
- Memory cost is measured at the intended latent size.
- Time embeddings reach every relevant block.
- Conditions visibly influence output.
- Image and spatial conditions receive identical augmentations.
- Cross-attention or guidance is added only when needed.

### Optimization

- A tiny subset can be overfit.
- Learning rate and gradient norms are stable.
- EMA updates after optimizer steps.
- Mixed precision has been compared against a full-precision sanity run.
- Training throughput and validation sampling cost are known.

### Validation and generation

- Stochastic validation loss is tracked with a moving average.
- Fixed noise, timesteps, seeds, and conditions are logged.
- Raw and EMA sample sources are identified.
- Sample quality, diversity, and condition adherence are evaluated separately.
- Sampler settings remain fixed when comparing checkpoints.

### Resume

- `last.ckpt` exists.
- Optimizer, scheduler, EMA, epoch, and global step restore correctly.
- VAE and diffusion metadata match current configuration.
- Latent normalization statistics are reused, not silently recomputed.

## Final takeaways

The diffusion model is a learned prior over the encoded-data distribution. Training succeeds when every part of that relationship remains explicit and consistent.

The most reusable lessons are:

1. Verify the VAE reconstruction ceiling before tuning diffusion.
2. Choose a latent normalization convention and save its statistics with checkpoints.
3. Keep prediction target, schedule, and sampler mathematically consistent.
4. Do not confuse training timesteps with sampling steps.
5. Expect stochastic validation loss to fluctuate.
6. Use fixed-seed EMA samples and timestep-bucket losses alongside average MSE.
7. Match U-Net depth and attention cost to latent geometry.
8. Use multiscale features for local spatial conditioning and cross-attention for global context.
9. Overfit a tiny subset before launching a long run.
10. Change one major variable at a time.
11. Resume from `last.ckpt` with the same VAE, objective, architecture, and normalization.
12. Remember that more sampling steps cannot repair a lossy image codec.

Once the latent contract is stable, diffusion training becomes much less mysterious. Loss fluctuations become interpretable, architecture changes become measurable, and sample quality can be traced to the component that actually controls it.

## Further reading

These are the most useful next stops; the article links each one where its idea first appears.

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — the DDPM forward process, reverse process, and noise-prediction foundation.
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — cosine schedules, learned reverse variances, and faster sampling.
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — deterministic and accelerated sampling from a DDPM-trained model.
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — diffusion in a pretrained autoencoder space and cross-attention conditioning.
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — joint conditional and unconditional training for guidance without a separate classifier.
- [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) — velocity parameterization and progressive reduction of sampling steps.
- [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556) — timestep loss weighting as a multitask optimization problem.
- [The VAE Is the Image Codec](/posts/vae-training-for-latent-diffusion/) — the companion guide to reconstruction quality, latent capacity, KL behavior, and normalization.
