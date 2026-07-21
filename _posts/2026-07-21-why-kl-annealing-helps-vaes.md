---
title: "Why KL Annealing Helps VAEs: A Mathematical Toy Experiment"
date: 2026-07-21 15:00:00 +0530
categories: [Machine Learning, Generative AI]
tags: [vae, kl-divergence, kl-annealing, kl-warmup, posterior-collapse, variational-inference, rate-distortion, python]
math: true
toc: true
comments: true
published: true
permalink: /posts/why-kl-annealing-helps-vaes/
description: "A small, exactly solvable VAE experiment that visualizes posterior collapse, KL weights, and KL warmup."
---

A variational autoencoder has two jobs that can conflict:

1. Reconstruct the input well.
2. Keep the encoder distribution close to a simple prior.

The standard objective is

$$
\mathcal{L}
=
\mathcal{L}_{\text{recon}}
+
\beta D_{\mathrm{KL}}\left(q_\phi(z\mid x)\|p(z)\right).
$$

The reconstruction term asks the latent variable $z$ to preserve information about $x$. The KL term asks the encoder distribution to resemble the same prior, usually $\mathcal{N}(0,I)$, for every input.

If the KL pressure becomes strong before the decoder has learned to use $z$, the easiest solution is often

$$
q_\phi(z\mid x) \approx p(z),
$$

which means that $z$ contains little information about $x$. This is posterior collapse.

KL annealing changes the order in which the two objectives take effect. It starts with a small KL weight and gradually increases it:

$$
\beta_t: 0 \longrightarrow \beta_{\text{target}}.
$$

This article builds a tiny VAE whose expected loss and gradients can be written exactly. We can then watch collapse and annealing happen without relying on a large neural network or noisy minibatches.

This is the mathematical companion to [The VAE Is the Image Codec](/posts/vae-training-for-latent-diffusion/), which covers the broader practical workflow for architecture selection, reconstruction losses, latent diagnostics, normalization, and downstream diffusion training.

> The point is not that this scalar model reproduces every behavior of a real VAE. The point is that it isolates the optimization conflict in a form we can derive and plot.

---

## 1. Why the KL term likes zero means and unit variances

For one latent dimension, let the encoder distribution be

$$
q(z\mid x)=\mathcal{N}(\mu,\sigma^2),
$$

and let the prior be

$$
p(z)=\mathcal{N}(0,1).
$$

Their KL divergence is

$$
D_{\mathrm{KL}}\left(q(z\mid x)\|p(z)\right)
=
\frac{1}{2}
\left(
\mu^2+\sigma^2-\log\sigma^2-1
\right).
$$

Most implementations predict the log variance

$$
r=\log\sigma^2,
$$

so that $\sigma^2=e^r$. The KL becomes

$$
K(\mu,r)
=
\frac{1}{2}
\left(
\mu^2+e^r-r-1
\right).
$$

Its derivatives are

$$
\frac{\partial K}{\partial \mu}=\mu,
$$

and

$$
\frac{\partial K}{\partial r}
=
\frac{1}{2}(e^r-1).
$$

Both derivatives vanish at

$$
\mu=0,
\qquad
r=0.
$$

Since $r=0$ implies $\sigma^2=1$, the global KL minimum is

$$
q(z\mid x)=\mathcal{N}(0,1)=p(z).
$$

![Contour plot of KL divergence over encoder mean and log variance, with its minimum at zero mean and zero log variance.](/assets/images/kl-annealing/kl_geometry.png)

The center of the plot is an extremely simple target for the optimizer. The encoder can reduce KL by making its output heads almost independent of the input:

$$
\mu_\phi(x)\approx 0,
\qquad
\log\sigma_\phi^2(x)\approx 0.
$$

If this happens for every $x$, then sampling becomes

$$
z
=
\mu_\phi(x)+\sigma_\phi(x)\epsilon
\approx
\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,1).
$$

The latent sample is now mostly input-independent noise.

---

## 2. An exactly solvable scalar VAE

Let the data be Gaussian:

$$
x\sim\mathcal{N}(0,V),
\qquad V=4.
$$

Use a scalar linear encoder:

$$
q(z\mid x)=\mathcal{N}(ax,\sigma^2).
$$

The parameter $a$ is the encoder gain. If $a=0$, the posterior mean does not depend on $x$.

Using the reparameterization trick,

$$
z=ax+\sigma\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,1).
$$

Use a scalar linear decoder:

$$
\hat{x}=bz.
$$

The parameter $b$ measures how strongly the decoder relies on the latent variable. If $b=0$, the decoder ignores $z$.

A collapsed point therefore looks like

$$
a\approx 0,
\qquad
b\approx 0,
\qquad
\sigma^2\approx 1.
$$

### Exact expected reconstruction error

The reconstruction error is

$$
x-\hat{x}
=
x-b(ax+\sigma\epsilon)
=
(1-ab)x-b\sigma\epsilon.
$$

Because $x$ and $\epsilon$ are independent and have zero mean,

$$
D
=
\mathbb{E}\left[(x-\hat{x})^2\right]
=
V(1-ab)^2+b^2\sigma^2.
$$

This equation contains two reconstruction costs:

- $V(1-ab)^2$: signal mismatch;
- $b^2\sigma^2$: noise amplified by the decoder.

### Exact expected KL

For a fixed $x$,

$$
D_{\mathrm{KL}}
=
\frac{1}{2}
\left(
(ax)^2+\sigma^2-\log\sigma^2-1
\right).
$$

Since $\mathbb{E}[x^2]=V$,

$$
K
=
\mathbb{E}_x[D_{\mathrm{KL}}]
=
\frac{1}{2}
\left(
Va^2+\sigma^2-\log\sigma^2-1
\right).
$$

### Training objective

The experiment minimizes

$$
\mathcal{J}_t
=
\lambda D+\beta_tK,
$$

with

$$
\lambda=0.05.
$$

The coefficient $\lambda$ is written explicitly because the meaningful strength of $\beta$ depends on the numerical scale of the reconstruction term.

---

## 3. Why encoder and decoder learning must coordinate

Write $r=\log\sigma^2$. The gradients for $a$ and $b$ are

$$
\frac{\partial\mathcal{J}}{\partial a}
=
-2\lambda Vb(1-ab)+\beta Va,
$$

and

$$
\frac{\partial\mathcal{J}}{\partial b}
=
\lambda\left[-2Va(1-ab)+2b\sigma^2\right].
$$

Near initialization, suppose $a$ and $b$ are both close to zero. Then

$$
\frac{\partial\mathcal{J}}{\partial a}
\approx
-2\lambda Vb+\beta Va,
$$

and

$$
\frac{\partial\mathcal{J}}{\partial b}
\approx
-2\lambda Va+2\lambda b.
$$

This exposes the optimization problem:

- The reconstruction gradient that helps $a$ depends on $b$.
- The reconstruction gradient that helps $b$ depends on $a$.
- The KL gradient $\beta Va$ acts directly on $a$ from the start.

So the encoder and decoder need to grow together, but the KL term can suppress the encoder before this coordination begins.

When $a$ is driven toward zero, the decoder receives almost no useful signal to increase $b$. When $b$ remains near zero, reconstruction provides almost no useful signal to increase $a$. The collapsed region becomes sticky.

![Encoder gain versus decoder reliance for constant, warmup, weak-KL, and strong-KL training schedules.](/assets/images/kl-annealing/encoder_decoder_trajectory.png)

In the plot, every run starts at the same near-zero point. Constant target KL and strong KL remain near the collapsed region. Warmup first permits both gains to grow, establishing a useful reconstruction path before regularization becomes strong.

---

## 4. KL weight as an information price

There is another way to interpret the experiment. Let $R$ denote the information rate carried by the latent variable. For a Gaussian source under mean-squared distortion, the ideal rate-distortion relationship is

$$
D(R)=Ve^{-2R}.
$$

Suppose we optimize

$$
J(R)=\lambda D(R)+\beta R.
$$

Substituting the Gaussian rate-distortion curve gives

$$
J(R)
=
\lambda Ve^{-2R}+\beta R.
$$

Differentiate:

$$
\frac{dJ}{dR}
=
-2\lambda Ve^{-2R}+\beta.
$$

For an interior optimum,

$$
-2\lambda Ve^{-2R}+\beta=0,
$$

which gives

$$
R^*(\beta)
=
\frac{1}{2}
\log\left(\frac{2\lambda V}{\beta}\right).
$$

Information rate cannot be negative, so the full solution is

$$
R^*(\beta)
=
\max\left(
0,
\frac{1}{2}
\log\left(\frac{2\lambda V}{\beta}\right)
\right).
$$

The collapse threshold is therefore

$$
\beta_c=2\lambda V.
$$

For this experiment,

$$
\beta_c
=
2(0.05)(4)
=
0.4.
$$

If $\beta\ge 0.4$, zero information is optimal in this idealized rate-distortion problem.

![The optimal information rate decreases to zero as the KL weight reaches the analytic collapse threshold.](/assets/images/kl-annealing/information_vs_beta.png)

This graph makes an important practical point:

> A KL weight is meaningful only relative to the scale of the reconstruction loss.

There is no universal statement that $\beta=1$ is weak, strong, or correct. A change in image size, likelihood normalization, reduction method, or reconstruction coefficient can radically change the effective trade-off.

---

## 5. The four schedules in the experiment

The simulation compares:

$$
\text{Constant target:}
\qquad
\beta_t=0.38,
$$

$$
\text{Linear warmup:}
\qquad
\beta_t
=
0.38\min\left(1,\frac{t}{1500}\right),
$$

$$
\text{Weak KL:}
\qquad
\beta_t=0.10,
$$

and

$$
\text{Strong KL:}
\qquad
\beta_t=0.60.
$$

![The constant, linear-warmup, weak-KL, and strong-KL beta schedules used in the toy experiment.](/assets/images/kl-annealing/beta_schedules.png)

The most important comparison is between the first two runs. They use the same final value, $\beta=0.38$. The only difference is whether the full penalty is active immediately or introduced gradually.

---

## 6. Measuring how much information the latent carries

In this Gaussian channel,

$$
z=ax+\sigma\epsilon.
$$

The signal variance is $Va^2$, while the noise variance is $\sigma^2$. Therefore the mutual information is

$$
I(X;Z)
=
\frac{1}{2}
\log\left(
1+\frac{Va^2}{\sigma^2}
\right).
$$

If $a=0$, then

$$
I(X;Z)=0.
$$

If $a$ grows or $\sigma^2$ shrinks, the latent carries more information.

![Mutual information during optimization, showing that warmup lets the channel become informative before compression.](/assets/images/kl-annealing/information_during_training.png)

The behavior is revealing:

- Constant $\beta=0.38$ remains almost completely collapsed.
- Strong $\beta=0.60$ collapses even more decisively.
- Weak $\beta=0.10$ learns a highly informative latent channel.
- Warmup first increases information, then compresses it as $\beta_t$ rises.

That rise-and-fall pattern is the intended logic of annealing:

1. Learn to use the channel.
2. Compress and regularize the channel.
3. Continue training at the desired final trade-off.

---

## 7. Reconstruction behavior

The reconstruction distortion is

$$
D=V(1-ab)^2+b^2\sigma^2.
$$

![Expected reconstruction distortion during optimization for the four KL-weighting schedules.](/assets/images/kl-annealing/reconstruction_during_training.png)

The collapsed runs remain close to

$$
D\approx V=4,
$$

which is what happens when the model does not use the latent signal.

The weak-KL run reconstructs best, but it also pays the largest KL cost. This is not automatically the best generative model: weak regularization can produce a latent distribution that is difficult to sample from using the prior.

The warmup run ends between the two extremes. It preserves some latent information while using the same final $\beta$ as the collapsed constant-target run.

### Final values

| Experiment | Final $\beta$ | Distortion | KL | $I(X;Z)$ in nats |
|---|---:|---:|---:|---:|
| Constant target | 0.38 | 3.9996 | 0.00005 | 0.00005 |
| Linear warmup | 0.38 | 3.6088 | 0.05159 | 0.05157 |
| Weak KL | 0.10 | 1.1539 | 0.63039 | 0.62218 |
| Strong KL | 0.60 | 4.0000 | approximately 0 | approximately 0 |

The experiment illustrates two distinct roles:

- **KL weighting** chooses the reconstruction-versus-regularization trade-off.
- **KL annealing** changes the optimization path used to reach that trade-off.

They are related, but they are not the same thing.

---

## 8. Minimal runnable code

The following code performs deterministic gradient descent on the exact expected loss. It does not sample a dataset or train a neural network, so the curves directly reflect the mathematics above.

```python
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("kl_annealing_outputs")
OUT.mkdir(exist_ok=True)

# Data and objective parameters
V = 4.0
RECON_WEIGHT = 0.05
BETA_TARGET = 0.38
STEPS = 3000
LR = 0.03

# Parameters are [encoder gain a, decoder gain b, log variance r].
INITIAL = np.array([0.01, 0.01, 0.0], dtype=float)


def metrics(parameters, beta):
    a, b, log_variance = parameters
    variance = np.exp(log_variance)

    distortion = V * (1.0 - a * b) ** 2 + b**2 * variance
    kl = 0.5 * (
        V * a**2 + variance - log_variance - 1.0
    )
    mutual_information = 0.5 * np.log1p(
        V * a**2 / variance
    )
    objective = RECON_WEIGHT * distortion + beta * kl

    return distortion, kl, mutual_information, objective, variance


def gradient(parameters, beta):
    a, b, log_variance = parameters
    variance = np.exp(log_variance)

    grad_a = (
        RECON_WEIGHT * (-2.0 * V * b * (1.0 - a * b))
        + beta * V * a
    )

    grad_b = RECON_WEIGHT * (
        -2.0 * V * a * (1.0 - a * b)
        + 2.0 * b * variance
    )

    grad_log_variance = (
        RECON_WEIGHT * b**2 * variance
        + 0.5 * beta * (variance - 1.0)
    )

    return np.array([grad_a, grad_b, grad_log_variance])


def simulate(name, beta_schedule):
    parameters = INITIAL.copy()
    rows = []

    for step in range(STEPS):
        beta = float(beta_schedule(step))
        distortion, kl, info, objective, variance = metrics(
            parameters, beta
        )
        a, b, log_variance = parameters

        rows.append(
            {
                "experiment": name,
                "step": step,
                "beta": beta,
                "a": a,
                "b": b,
                "log_variance": log_variance,
                "variance": variance,
                "distortion": distortion,
                "kl": kl,
                "mutual_information": info,
                "objective": objective,
            }
        )

        parameters -= LR * gradient(parameters, beta)

    return pd.DataFrame(rows)


schedules = {
    "Constant beta = 0.38": lambda step: BETA_TARGET,
    "Warmup to beta = 0.38": lambda step: (
        BETA_TARGET * min(1.0, step / 1500.0)
    ),
    "Weak beta = 0.10": lambda step: 0.10,
    "Strong beta = 0.60": lambda step: 0.60,
}

runs = {
    name: simulate(name, schedule)
    for name, schedule in schedules.items()
}

history = pd.concat(runs.values(), ignore_index=True)
history.to_csv(OUT / "history.csv", index=False)

# Plot mutual information.
fig = plt.figure(figsize=(8, 5))
for name, frame in runs.items():
    plt.plot(
        frame["step"],
        frame["mutual_information"],
        label=name,
    )
plt.xlabel("Optimization step")
plt.ylabel("I(X; Z), nats")
plt.title("Information carried by the latent variable")
plt.legend()
plt.tight_layout()
fig.savefig(OUT / "information_during_training.png", dpi=180)
plt.close(fig)

# Plot reconstruction distortion.
fig = plt.figure(figsize=(8, 5))
for name, frame in runs.items():
    plt.plot(frame["step"], frame["distortion"], label=name)
plt.xlabel("Optimization step")
plt.ylabel("Expected squared reconstruction error")
plt.title("Reconstruction during training")
plt.legend()
plt.tight_layout()
fig.savefig(OUT / "reconstruction_during_training.png", dpi=180)
plt.close(fig)

# Plot the encoder-decoder trajectory.
fig = plt.figure(figsize=(8, 6))
for name, frame in runs.items():
    plt.plot(frame["a"], frame["b"], label=name)
plt.scatter([INITIAL[0]], [INITIAL[1]], label="Initialization")
plt.xlabel("Encoder gain a")
plt.ylabel("Decoder reliance b")
plt.title("Encoder-decoder coordination")
plt.legend()
plt.tight_layout()
fig.savefig(OUT / "encoder_decoder_trajectory.png", dpi=180)
plt.close(fig)

print(
    history.groupby("experiment").tail(1)[
        [
            "experiment",
            "beta",
            "distortion",
            "kl",
            "mutual_information",
            "a",
            "b",
            "variance",
        ]
    ].to_string(index=False)
)
```

The code is included in full so the experiment can be inspected directly. You can also download the [standalone Python script](/assets/code/kl-annealing/simulate_kl_annealing.py), the tested [requirements file](/assets/code/kl-annealing/requirements.txt), and the supplied [final metrics CSV](/assets/data/kl-annealing/final_metrics.csv).

Install the dependencies with

```bash
pip install numpy pandas matplotlib
```

and run the script with

```bash
python simulate_kl_annealing.py
```

---

## 9. How to interpret real VAE training curves

The toy model suggests several practical diagnostics.

### Case A: KL quickly falls to nearly zero

If reconstruction still improves while KL approaches zero, the decoder may be learning to model the data without using the latent variable.

Check:

- KL per latent dimension;
- active units—latent dimensions whose posterior means vary across inputs;
- mutual-information estimates;
- decoder sensitivity to changes in $z$;
- reconstructions after replacing $z$ with an independent prior sample.

### Case B: KL is large but reconstruction is poor

If the **weighted KL contribution** dominates the total objective while reconstruction remains poor, the KL weight may be too strong relative to the reconstruction scale, or warmup may be too short. If the raw, unweighted KL itself remains unexpectedly large, inspect its sum-versus-mean reduction, posterior statistics, and numerical stability before changing $\beta$.

### Case C: Reconstruction is excellent but prior samples are poor

The KL weight may be too weak. The encoder can build informative posteriors that do not aggregate into a distribution close enough to the prior.

### Case D: Warmup helps early but collapse happens later

Annealing does not alter the final objective. If the final $\beta$ makes zero information optimal, sufficiently long training can still collapse.

Possible responses include:

- reducing the final KL weight;
- using a free-bits or minimum-rate objective that protects a small information allowance from immediate penalization;
- using cyclical annealing, which repeatedly lowers and raises the KL weight;
- weakening or restructuring the decoder;
- changing reconstruction/KL normalization;
- ensuring that conditioning or skip paths do not let the decoder bypass $z$;
- monitoring KL per dimension rather than only the total.

---

## 10. What KL annealing can and cannot do

KL annealing can help when the desired final objective admits a useful latent solution, but optimization reaches the collapsed solution too early.

It changes the path:

$$
\text{learn a useful code}
\longrightarrow
\text{regularize the code}
\longrightarrow
\text{train at the target trade-off}.
$$

It does not magically repair an unsuitable final objective. If the final KL pressure prices every bit of information out of the model, warmup can delay collapse but cannot remove that incentive.

The central lesson is therefore:

> Choose the final KL weight to define the information trade-off, and use annealing to make that trade-off easier to optimize.

---

## Summary

This toy experiment makes the roles of KL weighting and KL annealing visible:

1. The KL term has a simple minimum at $\mu=0$ and $\sigma^2=1$.
2. Near initialization, KL can suppress the encoder before the decoder learns to use the latent variable.
3. Encoder and decoder learning must coordinate; collapse prevents that coordination.
4. A large enough KL weight can make zero information optimal.
5. Warmup allows an informative reconstruction path to form before regularization becomes strong.
6. Warmup changes the optimization path, not the final objective.

That is why KL annealing is useful: not because KL is bad, but because applying its full pressure too early can make the easiest short-term solution the wrong long-term representation.

---

## Further reading

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), Diederik P. Kingma and Max Welling.
- [Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing](https://arxiv.org/abs/1903.10145), Hao Fu et al.
