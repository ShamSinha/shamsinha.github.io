---
title: "Generative AI from First Principles: Variational Inference, VAEs, and Diffusion"
date: 2026-07-20 12:00:00 +0530
categories: [Machine Learning, Generative AI]
tags: [generative-ai, variational-inference, vae, elbo, reparameterization-trick, diffusion-models, ddpm, latent-variables, bayesian-inference, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/generative-ai-variational-inference-vaes-diffusion/
description: "A first-principles path from Bayesian posterior approximation and the ELBO to variational autoencoders and denoising diffusion probabilistic models."
---

Generative modeling asks for more than a prediction. It asks us to learn enough of a data distribution to create new samples from it.

Two influential approaches appear very different:

- a variational autoencoder compresses data into a probabilistic latent space and decodes samples from it;
- a diffusion model destroys data with noise and learns how to reverse that destruction.

Their common language is probability: latent variables, approximate inference, Gaussian reparameterization, likelihood, and variational bounds.

This post collects the generative-model material from my original ML/AI mathematics study notes. The basic probability concepts are developed separately in [Probability and Statistical Inference for Machine Learning](/posts/probability-statistical-inference-for-machine-learning/).

## Topic map

- [Latent-variable models](#latent-variables)
- [Why exact posterior inference is hard](#intractable-posterior)
- [Variational inference and the ELBO](#variational-inference)
- [Variational autoencoders](#vae)
- [The reparameterization trick](#reparameterization)
- [DDPM forward diffusion](#forward-diffusion)
- [The closed-form noising equation](#closed-form)
- [Reverse diffusion and training](#reverse-diffusion)
- [How VAEs and diffusion models connect](#connection)

---

## 1. Latent-variable models {#latent-variables}

A latent variable is not directly observed but helps explain observed data. Let $x$ be an observation and $z$ a latent variable. A generative model describes their joint distribution as

$$
p_\theta(x,z)=p(z)p_\theta(x\mid z).
$$

The process reads from right to left:

1. sample a latent value $z$ from a prior $p(z)$;
2. generate $x$ from the conditional distribution $p_\theta(x\mid z)$.

For images, $z$ may encode factors such as pose, lighting, style, or object identity—even though the model is not explicitly told those names. For a mixture model, the latent variable might be a cluster assignment. In a hidden Markov model, it is an unobserved state sequence.

The likelihood of an observed example must marginalize the hidden variable:

$$
p_\theta(x)=\int p_\theta(x,z)\,dz
=\int p(z)p_\theta(x\mid z)\,dz.
$$

The integral is where the mathematical difficulty begins.

---

## 2. Why exact posterior inference is hard {#intractable-posterior}

After observing $x$, we would like the posterior distribution of its latent explanation:

$$
p_\theta(z\mid x)
=\frac{p_\theta(x,z)}{p_\theta(x)}
=\frac{p(z)p_\theta(x\mid z)}
{\int p(z)p_\theta(x\mid z)\,dz}.
$$

For a neural decoder, the evidence integral in the denominator is usually high-dimensional and has no useful closed form. “Intractable” does not mean logically impossible. It means exact evaluation is unavailable or computationally prohibitive at the required scale.

Sampling methods such as Markov chain Monte Carlo approximate the posterior with samples. Variational inference takes a different route: turn inference into optimization.

---

## 3. Variational inference: approximate a distribution by optimizing {#variational-inference}

Choose a tractable family $q_\phi(z\mid x)$ to approximate the true posterior $p_\theta(z\mid x)$. We would like to minimize

$$
D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\|p_\theta(z\mid x)\right).
$$

Substitute Bayes' rule:

$$
\begin{aligned}
D_{\mathrm{KL}}(q_\phi\|p_\theta)
&=\mathbb E_{q_\phi}
\left[
\log q_\phi(z\mid x)
-\log p_\theta(x,z)
+\log p_\theta(x)
\right]\\
&=\log p_\theta(x)
-\mathbb E_{q_\phi}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
\end{aligned}
$$

Define the evidence lower bound,

$$
\mathcal L(\theta,\phi;x)
=\mathbb E_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
$$

Then

$$
\log p_\theta(x)
=\mathcal L(\theta,\phi;x)
+D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\|p_\theta(z\mid x)\right).
$$

KL divergence is nonnegative, so

$$
\mathcal L(\theta,\phi;x)\le\log p_\theta(x).
$$

This is why it is a lower bound. Maximizing the ELBO simultaneously raises a lower bound on data likelihood and reduces the gap between the approximate and true posterior.

Variational inference is especially useful in three settings: limited data where informative priors matter, problems that require explicit uncertainty, and generative modeling with hidden variables.

### Choosing the variational family

The approximation can only express distributions inside its chosen family. A common **mean-field** assumption factorizes the latent dimensions:

$$
q_\phi(z\mid x)=\prod_{j=1}^d q_{\phi_j}(z_j\mid x).
$$

This makes expectations and gradients cheap, but it cannot directly represent posterior correlations between latent dimensions. Full-covariance Gaussians are richer and more expensive. Normalizing flows, mixtures, and structured variational families add flexibility at additional computational cost.

The KL direction also matters. Minimizing $D_{\mathrm{KL}}(q\|p)$ penalizes placing mass where $p$ has little mass very strongly, but it does not penalize every region that $q$ fails to cover equally. With a multimodal true posterior and a unimodal $q$, the approximation may concentrate on one mode rather than spread across low-density space between modes. This behavior is often called **mode seeking** or **zero forcing**.

### Variational inference versus MCMC

| Property | Variational inference | MCMC |
|---|---|---|
| Basic mechanism | Optimize parameters of $q_\phi$ | Construct correlated posterior samples |
| Typical speed | Fast after choosing a tractable family | Often slower and harder to scale |
| Main approximation error | Restricted family and optimization error | Finite-chain and convergence error |
| Uncertainty behavior | Can be too narrow | Asymptotically exact under suitable conditions |
| Diagnostics | ELBO, held-out prediction, sensitivity to initialization | Trace plots, effective sample size, chain convergence |

VI is attractive when repeated inference or large datasets make sampling too expensive. MCMC is attractive when posterior fidelity matters more than speed and reliable convergence is achievable. Neither method removes the need to check the model itself.

The ELBO is generally non-convex for neural models, so different initializations can find different local optima. A high ELBO also does not prove that $q_\phi$ captures every scientifically relevant uncertainty. Posterior predictive checks, multiple runs, and comparison with a richer family or a small MCMC benchmark can reveal approximation failure.

---

## 4. Variational autoencoders {#vae}

A VAE amortizes variational inference with two neural networks:

- an **encoder** $q_\phi(z\mid x)$ predicts a distribution over latent codes;
- a **decoder** $p_\theta(x\mid z)$ predicts a distribution over reconstructions.

Usually,

$$
q_\phi(z\mid x)
=\mathcal N\!\left(z;
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right)
$$

and the prior is

$$
p(z)=\mathcal N(0,I).
$$

Using $p_\theta(x,z)=p_\theta(x\mid z)p(z)$, the ELBO becomes

$$
\mathcal L
=\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\|p(z)\right).
$$

The two terms have distinct jobs:

1. **Reconstruction term:** make $z$ retain enough information for the decoder to explain $x$.
2. **KL term:** keep each encoded distribution near the shared prior so the latent space remains continuous and sampleable.

If only reconstruction mattered, the model could place examples in isolated latent islands. Sampling a random $z\sim\mathcal N(0,I)$ might then land in an unmapped region. The KL term organizes the latent space at the price of some reconstruction precision.

For diagonal Gaussians, the KL term has a closed form:

$$
D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\|\mathcal N(0,I)\right)
=\frac12\sum_j
\left(
\mu_j^2+\sigma_j^2-\log\sigma_j^2-1
\right).
$$

The reconstruction term depends on the observation model. A Bernoulli decoder leads to binary cross-entropy; a fixed-variance Gaussian decoder leads to a scaled squared-error term. This is a likelihood choice, not merely a convenient loss-function swap, and its scale determines the balance against KL regularization.

---

## 5. The reparameterization trick {#reparameterization}

The encoder outputs $\mu_\phi(x)$ and $\sigma_\phi(x)$, but a direct draw

$$
z\sim\mathcal N(\mu_\phi(x),\sigma_\phi^2(x))
$$

places a stochastic sampling operation between the loss and the encoder parameters. The problem is not that probability distributions can never be differentiated. The problem is that a raw sample is not a deterministic function of $\mu$ and $\sigma$ in the computation graph.

Move the randomness into a parameter-free variable:

$$
\varepsilon\sim\mathcal N(0,I),
\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon.
$$

Conditional on the sampled $\varepsilon$, $z$ is differentiable with respect to $\mu_\phi$ and $\sigma_\phi$:

$$
\frac{\partial z}{\partial\mu}=1,
\qquad
\frac{\partial z}{\partial\sigma}=\varepsilon.
$$

The distribution is unchanged. Scaling and shifting a standard normal gives

$$
z\sim\mathcal N(\mu,\sigma^2).
$$

Indeed,

$$
\mathbb E[z]=\mu,
\qquad
\operatorname{Var}(z)=\sigma^2\operatorname{Var}(\varepsilon)=\sigma^2.
$$

So the trick does not replace genuine sampling with a fake deterministic point. It expresses the same random variable in a form through which low-variance pathwise gradients can flow.

### One complete VAE training step

Implementations usually predict log-variance $\ell=\log\sigma^2$ rather than $\sigma$ directly. This permits any real network output while keeping the reconstructed standard deviation positive:

$$
\sigma=\exp(\ell/2),
\qquad
z=\mu+\exp(\ell/2)\odot\varepsilon.
$$

For one mini-batch, training follows this path:

1. The encoder maps $x$ to $\mu$ and $\ell$.
2. Sample parameter-free noise $\varepsilon\sim\mathcal N(0,I)$ and construct $z$.
3. The decoder maps $z$ to the parameters of $p_\theta(x\mid z)$.
4. Compute the negative ELBO,

   $$
   L
   =-\mathbb E_q[\log p_\theta(x\mid z)]
   +\frac12\sum_j
   \left(\mu_j^2+e^{\ell_j}-\ell_j-1\right).
   $$

5. Backpropagate once through the complete graph and update encoder and decoder parameters.

The reconstruction gradient reaches the encoder through $z$:

$$
\frac{\partial L_{\mathrm{rec}}}{\partial\mu}
=\frac{\partial L_{\mathrm{rec}}}{\partial z},
$$

$$
\frac{\partial L_{\mathrm{rec}}}{\partial\ell}
=\frac{\partial L_{\mathrm{rec}}}{\partial z}
\odot\frac12\exp(\ell/2)\odot\varepsilon.
$$

The KL term contributes analytical gradients

$$
\frac{\partial L_{\mathrm{KL}}}{\partial\mu_j}=\mu_j,
\qquad
\frac{\partial L_{\mathrm{KL}}}{\partial\ell_j}
=\frac12(e^{\ell_j}-1).
$$

The chain rule then carries both contributions through the encoder. The decoder receives gradients only from the reconstruction term, while the encoder receives reconstruction and KL gradients.

A compact PyTorch-style step is:

```python
mu, logvar = encoder(x)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
x_params = decoder(z)

reconstruction = negative_log_likelihood(x, x_params).mean()
kl = -0.5 * (1 + logvar - mu.square() - logvar.exp())
loss = reconstruction + kl.sum(dim=1).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Failure modes and objective variants

With a very powerful decoder, the model may reconstruct well without using $z$. Then $q_\phi(z\mid x)$ collapses toward the prior and the KL term approaches zero—**posterior collapse**. KL warm-up, free bits, weakening the decoder, or changing the training schedule can encourage latent usage.

A $\beta$-VAE weights the KL term by a coefficient $\beta$. Values above $1$ emphasize a structured, prior-like latent space at the cost of reconstruction detail; values below $1$ favor reconstruction. This is a modeling trade-off, not a guarantee of interpretable or disentangled factors.

---

## 6. DDPM forward diffusion: a fixed destruction process {#forward-diffusion}

A denoising diffusion probabilistic model introduces latent variables $x_1,\ldots,x_T$ with the same dimensionality as data $x_0$. The forward process is a fixed Markov chain that gradually adds Gaussian noise:

$$
q(x_t\mid x_{t-1})
=\mathcal N
\left(
x_t;
\sqrt{1-\beta_t}\,x_{t-1},
\beta_tI
\right).
$$

Define

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^t\alpha_s.
$$

A sample at one step can be written with the same reparameterization idea used by VAEs:

$$
x_t=\sqrt{\alpha_t}x_{t-1}
+\sqrt{1-\alpha_t}\,\varepsilon_t,
\qquad
\varepsilon_t\sim\mathcal N(0,I).
$$

For a schedule of small positive $\beta_t$, signal gradually decays and noise grows. The schedule determines how quickly information is destroyed: early DDPMs commonly used a linear $\beta_t$ schedule, while cosine schedules shape the cumulative signal level $\bar\alpha_t$ more smoothly. At a sufficiently large $T$, $x_T$ is approximately standard Gaussian.

---

## 7. Why we can jump directly to any noise level {#closed-form}

Training would be expensive if creating $x_t$ required simulating all previous steps. Fortunately, the Gaussian chain has a closed form:

$$
q(x_t\mid x_0)
=\mathcal N
\left(
x_t;
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right),
$$

or

$$
x_t
=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\,\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I).
$$

To see why, expand two steps:

$$
\begin{aligned}
x_t
&=\sqrt{\alpha_t}
\left(
\sqrt{\alpha_{t-1}}x_{t-2}
+\sqrt{1-\alpha_{t-1}}\varepsilon_{t-1}
\right)
+\sqrt{1-\alpha_t}\varepsilon_t\\
&=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}
+\sqrt{\alpha_t(1-\alpha_{t-1})}\varepsilon_{t-1}
+\sqrt{1-\alpha_t}\varepsilon_t.
\end{aligned}
$$

The last two terms are independent zero-mean Gaussians. Their variances add:

$$
\alpha_t(1-\alpha_{t-1})+(1-\alpha_t)
=1-\alpha_t\alpha_{t-1}.
$$

Repeating the argument yields the product $\bar\alpha_t$. This is the subtle “how did two epsilon terms become one?” step emphasized in the Notion DDPM notes.

The practical consequence is important: training can sample a random $t$ and construct $x_t$ in one operation.

---

## 8. Reverse diffusion: learn to remove the noise {#reverse-diffusion}

The generative process begins from noise

$$
x_T\sim\mathcal N(0,I)
$$

and learns reverse transitions

$$
p_\theta(x_{t-1}\mid x_t)
=\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

The true reverse conditional depends on the unknown data distribution, so a neural network—commonly a time-conditioned U-Net for images—approximates it.

Rather than directly predicting the reverse mean, the common DDPM parameterization predicts the noise $\varepsilon$ used to create $x_t$. Training samples $x_0$, $t$, and $\varepsilon$, constructs

$$
x_t
=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\varepsilon,
$$

then minimizes

$$
L_{\mathrm{simple}}
=\mathbb E_{x_0,t,\varepsilon}
\left[
\|\varepsilon-\varepsilon_\theta(x_t,t)\|_2^2
\right].
$$

If the model can estimate the added noise, it can estimate the clean signal:

$$
\hat x_0
=\frac{x_t-\sqrt{1-\bar\alpha_t}\,
\varepsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}.
$$

Generation repeatedly applies the learned reverse update from $T$ down to $0$. The forward chain is fixed; the reverse chain contains the learned generative knowledge.

---

## 9. The variational view of diffusion {#diffusion-elbo}

DDPMs can also be viewed as latent-variable models trained with a variational bound. The forward process $q(x_{1:T}\mid x_0)$ plays the role of a fixed approximate posterior, while the reverse model defines

$$
p_\theta(x_{0:T})
=p(x_T)\prod_{t=1}^T
p_\theta(x_{t-1}\mid x_t).
$$

A variational upper bound on negative log-likelihood decomposes into terms comparing true forward posteriors with learned reverse transitions. Under the standard Gaussian parameterization, these terms lead to denoising objectives; the widely used noise-prediction MSE is a simplified reweighted form.

This explains why the same ideas recur in VAEs and diffusion models: both introduce latent randomness, use a tractable distribution to reason about hidden variables, and optimize a bound connected to data likelihood.

---

## 10. VAE and diffusion: two routes through latent uncertainty {#connection}

| Question | VAE | DDPM |
|---|---|---|
| What is latent? | A compact code $z$ | A sequence $x_1,\ldots,x_T$ |
| What is fixed? | Usually the prior $p(z)$ | The forward noising process $q$ |
| What is learned? | Encoder and decoder | Reverse denoising transitions |
| Key reparameterization | $z=\mu+\sigma\varepsilon$ | $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$ |
| Typical objective | Reconstruction minus KL | Noise-prediction MSE / diffusion ELBO |
| Sampling | Usually one decoder pass | Iterative denoising |

VAEs make inference fast and latent manipulation natural, but a simple observation model can produce overly smooth samples. Diffusion models often achieve excellent sample quality but require an iterative sampling process. Modern systems mix these ideas—for example, latent diffusion performs the diffusion process in an autoencoder's compressed latent space.

The deeper common thread is the reparameterized Gaussian. It turns stochastic modeling into something gradient-based learning can optimize, whether the randomness describes a semantic latent code or a controlled ladder of noise levels.

---

## A compact mental model

1. A generative model explains observations using hidden random variables.
2. Exact inference over those variables is often intractable.
3. Variational inference replaces exact integration with an optimized approximation and an ELBO.
4. A VAE learns both the approximation and the decoder, using reparameterization to train through samples.
5. A DDPM fixes the forward approximation, learns the reverse process, and uses a closed-form Gaussian to train at arbitrary time steps.

Once these five steps are clear, the equations stop looking like unrelated tricks. They become variations on one problem: how to learn a probability distribution when the useful explanation of the data is hidden.

---

## Further reading

- [Variational Inference: The Basics](https://towardsdatascience.com/variational-inference-the-basics-f70ac511bcea)
