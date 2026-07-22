---
title: "Training Latent Diffusion Without Guesswork: The Core Pipeline"
date: 2026-07-22 09:00:00 +0530
categories: [Machine Learning, Generative AI]
tags: [diffusion-models, latent-diffusion, denoising, vae, noise-schedule, training-guide]
math: true
toc: true
comments: true
published: true
permalink: /posts/training-latent-diffusion-models/
description: "An approachable systems-first guide to the latent diffusion training pipeline, prediction targets, noise schedules, latent normalization, and a stable baseline."
---

Your denoising loss is decreasing, but the generated images still look poor. Is the U-Net too small? Does the sampler need more steps? Should epsilon prediction be replaced with velocity prediction?

Possibly—but those are rarely the first questions to answer.

A latent diffusion system crosses several boundaries: image to latent, clean latent to noisy latent, network output to a reverse-process estimate, and generated latent back to image. A mismatch at any boundary can produce bad samples even while the loss appears healthy.

This article develops the smallest complete mental model needed to train that system reliably. It deliberately postpones advanced conditioning and debugging so the core pipeline remains visible.

{% include diffusion-training-series.md %}

## What does the diffusion model learn?

A latent diffusion model learns the distribution of encoded data. The VAE defines **how an image is represented**; diffusion learns **which representations are plausible** and how to generate them from noise.

```text
VAE encoder -> represents images as latents
diffusion   -> learns the distribution of those latents
VAE decoder -> turns generated latents back into images
```

This is the central efficiency idea behind [latent diffusion](https://arxiv.org/abs/2112.10752): run the expensive generative process in a compressed space while preserving enough information for the decoder to reconstruct a useful image.

Calling diffusion *the latent prior* is useful, but it needs one qualification. A VAE may already be regularized toward an analytic prior such as $\mathcal N(0,I)$. The diffusion model is not that fixed Gaussian. It is a learned generative model of the **aggregated encoded-data distribution**—the latents actually produced by the frozen encoder.

The three learned components therefore own different failures:

| Component | Learns | Cannot repair |
| --- | --- | --- |
| VAE encoder | A compact image representation | Information discarded by its bottleneck |
| Diffusion denoiser | The structure of the encoded-data distribution | A lossy or inconsistently scaled latent space |
| VAE decoder | A mapping from latent to pixels | Missing latent detail or a poorly generated latent |

If real-image VAE reconstructions are already soft, diffusion does not own that defect. Start with [The VAE Is the Image Codec](/posts/vae-training-for-latent-diffusion/) before tuning the denoiser.

## Write down the complete contract

Before training, write the entire generation path:

```text
real image
    -> frozen encoder
    -> normalized clean latent z_0
    -> forward diffusion produces z_t
    -> denoiser predicts noise, velocity, or z_0
    -> reverse sampler produces generated z_0
    -> latent normalization is inverted
    -> frozen decoder
    -> generated image
```

Every arrow is an interface. Shape, scale, prediction convention, schedule, and optional condition must agree across training, validation, standalone generation, and checkpoint resume.

Before a full run, verify:

- Real images survive the VAE encode–decode round trip.
- The VAE is frozen and in evaluation mode.
- The latent shape matches the denoiser input and output.
- Latent normalization is measured, reversible, and saved.
- Normalizing and then inverting a latent does not change its reconstruction.
- The training target and sampler use the same prediction convention.

Treat this as a preflight test. A larger model will not repair a broken contract.

## How one training example is created

Diffusion constructs a noisy latent directly from a clean latent $z_0$:

$$
z_t
=
\sqrt{\bar\alpha_t}\,z_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

Here:

- $t$ is a randomly sampled timestep.
- $\epsilon$ is standard Gaussian noise.
- $\bar\alpha_t$ controls how much clean signal remains.

The denoiser receives $z_t$, $t$, and any condition. It predicts a target from which the reverse process can estimate a cleaner latent. The original [DDPM formulation](https://arxiv.org/abs/2006.11239) connects this denoising objective to a variational bound and denoising score matching.

The important computational point is that training does **not** run the full noising chain. Any $z_t$ can be sampled directly from $z_0$ in one operation. A process with 1,000 defined timesteps still uses only one denoiser evaluation per example during an ordinary training step.

## Training timesteps are not sampling steps

These two settings are commonly confused:

| Setting | What it controls | Training-batch cost |
| --- | --- | --- |
| Training timesteps $T$ | The discretized forward noise process | Usually one randomly sampled level per example |
| Sampling steps | Denoiser evaluations during generation | None, unless the batch also logs generated images |

A common baseline uses $T=1000$. Increasing $T$ changes the available noise levels and schedule discretization; it does not make every batch call the network 1,000 times.

Generation is different. A DDPM sampler may traverse every training timestep, while DDIM and other accelerated samplers can use a subset. More sampling steps may reduce discretization error, but they cannot restore information discarded by the VAE or compensate for an untrained denoiser.

Sampling also explains unexpectedly slow validation. Logging 16 images with 400 reverse steps requires 6,400 denoiser evaluations, independent of the cost of validation-loss computation.

## Choose one prediction target

The network output can represent noise, clean data, or velocity. The tensor shapes may be identical, but the learned meanings are not.

### Epsilon prediction

The network predicts the sampled noise $\epsilon$. This is the target used by the original DDPM experiments and remains the clearest baseline.

Its effective weighting across signal-to-noise ratios is not always ideal. Easy high-SNR and difficult low-SNR examples can influence optimization differently, but that is a reason to measure timestep behavior—not a reason to abandon the baseline immediately.

### Clean-latent prediction

The network predicts $z_0$ directly. The output is intuitive, but direct clean prediction becomes difficult at high-noise timesteps, and the loss weighting deserves careful attention.

### Velocity prediction

Velocity combines noise and clean signal:

$$
v_t
=
\sqrt{\bar\alpha_t}\,\epsilon
-
\sqrt{1-\bar\alpha_t}\,z_0.
$$

This parameterization appears in [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) and can behave more evenly across noise levels in some setups.

It is not an automatic upgrade. Instability after switching to velocity prediction often indicates an incorrect target conversion, sampler mismatch, schedule interaction, or hyperparameters inherited from an epsilon run.

When comparing targets:

1. Keep the VAE, normalization, and schedule fixed.
2. Unit-test both the target and reverse conversion.
3. Start a new run unless the checkpoint used the same target.
4. Compare fixed-seed samples and loss by timestep range.

Never resume an epsilon checkpoint as a velocity-prediction run merely because the output shapes match.

## Choose a noise schedule deliberately

The schedule determines how signal-to-noise ratio changes with $t$.

### Linear beta schedule

A linear beta schedule is simple, widely implemented, and historically tied to the original DDPM baseline. That makes it a useful control experiment, not a universal optimum.

### Cosine schedule

A cosine schedule preserves usable signal for more of the trajectory and changes how training effort is distributed across noise levels. It was introduced as a practical improvement in [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

Cosine plus velocity prediction is a reasonable experiment, but changing both at once destroys the comparison needed to learn which change helped.

### Signal-to-noise weighting

Uniform timestep sampling does not imply equal learning value at every timestep. Techniques such as [Min-SNR weighting](https://arxiv.org/abs/2303.09556) rebalance conflicting timestep objectives.

Add weighting only after logging loss by timestep range. Otherwise, it solves a problem that has not been observed.

## Freeze and normalize the latent space

The VAE should normally remain frozen during diffusion training:

```python
vae.eval()

for parameter in vae.parameters():
    parameter.requires_grad = False
```

Freezing prevents the latent coordinate system from drifting while the denoiser learns it. Evaluation mode also disables any training-dependent dropout or batch-normalization behavior.

### Posterior means or posterior samples?

Encoding with the posterior mean $\mu$ produces a deterministic, lower-variance latent dataset. Encoding with posterior samples better matches stochastic VAE training but can make the distribution harder to model.

Neither convention is universally correct. Choose one and use it consistently for:

- Latent-statistics computation.
- Diffusion training and validation.
- Standalone generation tests.

### Normalization is part of the checkpoint

For a custom VAE, channel-wise standardization is a reasonable default:

$$
z_{\text{norm},c}
=
\frac{z_c-\mu_c}{\sigma_c},
\qquad
z_c=z_{\text{norm},c}\sigma_c+\mu_c.
$$

Some established systems use a single scalar scale factor instead. Either convention can work. What matters is that it is measured or deliberately chosen, inverted before decoding, and never changed silently.

Compute statistics from the frozen VAE over the training split. Save the convention and values with the diffusion checkpoint. Recomputing them while keeping denoiser weights fixed changes the meaning of every latent coordinate.

## A conservative baseline

The following is a starting experiment, not a claim of optimality. Field names are illustrative:

```yaml
vae:
  checkpoint: path/to/frozen_vae.ckpt
  use_mu: true
  latent_normalization: channel
  latent_channel_mean:
  latent_channel_std:

diffusion:
  timesteps: 1000
  beta_schedule: linear
  prediction_type: epsilon

model:
  architecture: unet
  hidden_dims: [128, 256, 512]
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

The baseline is intentionally boring: frozen codec, fixed normalization, epsilon prediction, established schedule, small U-Net, and EMA. Its purpose is to reveal whether the pipeline works before advanced features create more places to look.

## Five takeaways

1. Diffusion models the encoded-data distribution; it cannot repair a lossy VAE.
2. Training samples one noise level directly, while generation performs many reverse steps.
3. Prediction target, schedule, and sampler must share the same mathematical convention.
4. Latent normalization is part of the model contract and must travel with the checkpoint.
5. Establish one simple baseline before changing objectives, schedules, capacity, or conditioning.

The core pipeline is now complete. [Part 2](/posts/conditioning-and-scaling-latent-diffusion/) builds on it by choosing U-Net capacity, injecting conditions, and scaling training without losing spatial alignment.
