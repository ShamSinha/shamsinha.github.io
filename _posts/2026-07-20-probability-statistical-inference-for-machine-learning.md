---
title: "Probability and Statistical Inference for Machine Learning"
date: 2026-07-20 11:00:00 +0530
categories: [Machine Learning, Statistics]
tags: [probability, bayes-theorem, likelihood, maximum-likelihood, hypothesis-testing, p-values, central-limit-theorem, correlation, multicollinearity, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/probability-statistical-inference-for-machine-learning/
description: "A first-principles guide to probability, likelihood, Bayesian updating, maximum likelihood, distributions, hypothesis testing, and correlation for machine learning."
---

Machine learning is built on uncertainty. Data is finite, labels can be noisy, parameters are estimated rather than known, and every evaluation is only a sample of future behavior.

Probability gives us a language for uncertainty. Statistical inference tells us what the observed data allows us to conclude.

This post reorganizes my original probability and statistics study notes. The model-building half continues in [Classical Machine Learning Models from First Principles](/posts/classical-machine-learning-models-from-first-principles/), while latent-variable inference is developed in [Generative AI from First Principles](/posts/generative-ai-variational-inference-vaes-diffusion/).

## Topic map

- [Probability, likelihood, and Bayes' theorem](#probability-bayes)
- [Maximum likelihood estimation](#mle)
- [Binomial distribution](#binomial)
- [Central limit theorem](#clt)
- [Jensen's inequality](#jensen)
- [Hypothesis testing and p-values](#hypothesis-testing)
- [Covariance and correlation](#correlation)
- [Multicollinearity](#multicollinearity)

---

## 1. Probability, likelihood, and Bayes' theorem {#probability-bayes}

Probability and likelihood use the same function but ask opposite questions.

- **Probability:** with parameters $\theta$ fixed, how probable is possible data $x$?
- **Likelihood:** with observed data $x$ fixed, which parameter values $\theta$ make it plausible?

We may write the same quantity as $p(x\mid\theta)$ in both cases, but the variable being treated as unknown has changed. A likelihood is a function of $\theta$; it is not generally a probability distribution over $\theta$ and need not integrate to one.

Bayes' theorem connects prior belief, likelihood, and posterior belief:

$$
p(\theta\mid x)
=\frac{p(x\mid\theta)p(\theta)}{p(x)}.
$$

- $p(\theta)$ is the **prior**, representing uncertainty before seeing the current data.
- $p(x\mid\theta)$ is the **likelihood** of the observations under a candidate parameter.
- $p(\theta\mid x)$ is the **posterior**, the updated uncertainty after seeing the data.
- $p(x)$ is the **evidence** or marginal likelihood:

$$
p(x)=\int p(x\mid\theta)p(\theta)\,d\theta.
$$

The denominator normalizes the numerator over all possible parameter values.

### A concrete Beta-Binomial update

Suppose $p$ is the unknown proportion of people who prefer tea. Start with a uniform prior,

$$
p\sim\operatorname{Beta}(1,1).
$$

After observing six tea preferences in ten independent responses, the likelihood is proportional to

$$
p^6(1-p)^4.
$$

Multiplying prior and likelihood gives

$$
p\mid x\sim\operatorname{Beta}(7,5).
$$

The posterior is not a single best number. It retains a distribution over plausible values, combining the uncertainty in the prior with the evidence supplied by the observations.

### Bayesian inference versus point estimation

A conventional model often produces one parameter estimate $\hat\theta$. Bayesian inference instead produces a distribution $p(\theta\mid x)$. That distribution can express several plausible parameter values and the uncertainty between them.

Neither viewpoint removes modeling assumptions. The posterior depends on the prior and likelihood; a point estimator depends on its objective and sampling assumptions. The benefit of probability is that those assumptions become explicit.

---

## 2. Maximum likelihood estimation {#mle}

Given independent observations $x_1,\ldots,x_n$, the likelihood is

$$
L(\theta)=\prod_{i=1}^n p(x_i\mid\theta).
$$

Maximum likelihood estimation chooses

$$
\hat\theta_{\mathrm{MLE}}
=\arg\max_\theta L(\theta).
$$

Products of many probabilities can underflow numerically and are awkward to differentiate. Because the logarithm is strictly increasing, the maximizer is unchanged if we use log-likelihood:

$$
\hat\theta_{\mathrm{MLE}}
=\arg\max_\theta\sum_{i=1}^n\log p(x_i\mid\theta).
$$

### Example: estimating a Gaussian mean

Suppose

$$
x_i\sim\mathcal N(\mu,\sigma^2)
$$

with fixed $\sigma^2$. Ignoring constants, the log-likelihood is

$$
\ell(\mu)
=-\frac{1}{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2.
$$

Maximizing it is equivalent to minimizing squared error. Differentiating gives

$$
\frac{d\ell}{d\mu}
=\frac{1}{\sigma^2}\sum_{i=1}^n(x_i-\mu)=0,
$$

so

$$
\hat\mu_{\mathrm{MLE}}=\frac1n\sum_{i=1}^n x_i.
$$

The sample mean is therefore the value that makes the observations most likely under the Gaussian model.

When a closed-form solution is unavailable, minimize the negative log-likelihood with gradient-based updates. In schematic Python, a joint update for the Gaussian parameters is

```python
# eta is the step size; dL_dmu and dL_dsigma are loss gradients.
mu_new = mu - eta * dL_dmu
sigma_new = sigma - eta * dL_dsigma
```

The following executable example estimates both $\mu$ and $\sigma$ numerically. The lower bound keeps $\sigma$ positive, and minimizing the negative log-likelihood is equivalent to maximizing the likelihood.

```python
import numpy as np
from scipy.optimize import minimize


# Generate sample data from N(5, 2^2).
np.random.seed(42)
data = np.random.normal(loc=5.0, scale=2.0, size=1000)


def neg_log_likelihood(params):
    mu, sigma = params
    if sigma <= 0:
        return np.inf

    n = len(data)
    log_likelihood = (
        -n * np.log(sigma * np.sqrt(2 * np.pi))
        - np.sum((data - mu) ** 2) / (2 * sigma**2)
    )
    return -log_likelihood


result = minimize(
    neg_log_likelihood,
    x0=[0.0, 1.0],
    method="L-BFGS-B",
    bounds=[(None, None), (1e-5, None)],
)

mu_hat, sigma_hat = result.x
print(f"Estimated mu: {mu_hat:.4f}")
print(f"Estimated sigma: {sigma_hat:.4f}")
```

### Why ML losses look familiar

Many loss functions are negative log-likelihoods:

- squared error corresponds to a Gaussian observation model;
- binary cross-entropy corresponds to a Bernoulli model;
- multiclass cross-entropy corresponds to a categorical model.

This connection explains why minimizing a loss is often statistical estimation in disguise.

---

## 3. The binomial distribution {#binomial}

If $X$ counts successes in $n$ independent Bernoulli trials with success probability $p$, then

$$
P(X=k)=\binom nkp^k(1-p)^{n-k}.
$$

Its mean and variance are

$$
\mathbb E[X]=np,
\qquad
\operatorname{Var}(X)=np(1-p).
$$

These formulas follow directly by writing the count as a sum of Bernoulli indicators. Let $X_i=1$ when trial $i$ succeeds and $0$ otherwise, so $X=\sum_{i=1}^nX_i$. Then

$$
\mathbb E[X]
=\sum_{i=1}^n\mathbb E[X_i]
=np.
$$

Because $X_i^2=X_i$,

$$
\operatorname{Var}(X_i)
=\mathbb E[X_i^2]-\mathbb E[X_i]^2
=p-p^2
=p(1-p).
$$

Independence makes the covariance terms zero, so the variances add and give $\operatorname{Var}(X)=np(1-p)$.

The binomial assumptions matter:

1. the number of trials is fixed;
2. each trial has two outcomes;
3. the success probability is constant;
4. trials are independent.

The model appears in click-through rates, conversion tests, defect counts, and classification accuracy. If examples are correlated or have different success probabilities, an ordinary binomial uncertainty calculation can be too optimistic.

---

## 4. The central limit theorem {#clt}

If $X_1,\ldots,X_n$ are independent and identically distributed with finite mean $\mu$ and variance $\sigma^2$, then

$$
\frac{\sqrt n(\bar X_n-\mu)}{\sigma}
\xrightarrow{d}\mathcal N(0,1).
$$

It is the *standardized sample mean* whose distribution approaches normality as $n$ grows. The theorem does not say:

- that the original population becomes normal;
- that the raw observations become normal;
- that every sample size is automatically large enough.

Heavy tails, strong skew, and dependent observations can slow or invalidate the usual approximation. In machine learning, this matters whenever confidence intervals or significance tests are built from cross-validation scores, user events, or other observations that may not be independent.

---

## 5. Jensen's inequality {#jensen}

For a convex function $f$,

$$
f(\mathbb E[X])\le\mathbb E[f(X)].
$$

For a concave function such as $\log$, the inequality reverses:

$$
\mathbb E[\log X]\le\log\mathbb E[X].
$$

Geometrically, a chord between two points on a convex function lies above the curve. Averaging inputs before applying the function therefore produces no larger a value than averaging the function outputs.

Jensen's inequality appears throughout machine learning. It explains why exchanging an expectation and a nonlinear function changes the result, and it is the mathematical step behind the evidence lower bound used in variational inference.

---

## 6. Hypothesis testing and p-values {#hypothesis-testing}

A hypothesis test begins with:

- a null hypothesis $H_0$;
- an alternative $H_1$;
- a test statistic $T(X)$ whose behavior under $H_0$ is known or approximated.

For an observed value $t_{\mathrm{obs}}$, a p-value is the probability, **assuming $H_0$**, of obtaining a statistic at least as incompatible with $H_0$ as the one observed:

$$
p\text{-value}
=P_{H_0}\!\left(T(X)\text{ is at least as extreme as }t_{\mathrm{obs}}\right).
$$

This is a conditional tail probability. It is not:

- the probability that $H_0$ is true;
- the probability that the result occurred “by random chance”;
- the size or practical importance of the effect;
- proof of the alternative hypothesis.

If a preselected significance level is $\alpha=0.05$ and the p-value is below it, we reject $H_0$ under that testing procedure. That rule controls long-run Type I error under its assumptions. It does not mean there is a $95\%$ probability that the scientific claim is correct.

Always report an effect size and uncertainty interval alongside a p-value. With enough data, a negligible effect can be statistically significant; with too little data, an important effect may be missed.

### Type I and Type II errors

| Decision | $H_0$ true | $H_0$ false |
|---|---|---|
| Reject $H_0$ | Type I error | Correct detection |
| Do not reject $H_0$ | Correct | Type II error |

The probability of a Type I error is controlled by $\alpha$. Power is the probability of rejecting $H_0$ when a specified alternative is true. Lowering $\alpha$ without increasing data typically reduces false positives but also reduces power.

---

## 7. Covariance and correlation {#correlation}

Covariance measures whether two variables move together:

$$
\operatorname{Cov}(X,Y)
=\mathbb E[(X-\mu_X)(Y-\mu_Y)].
$$

Its sign gives direction, but its magnitude depends on measurement units. Pearson correlation standardizes covariance:

$$
\rho_{X,Y}
=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}.
$$

This makes it dimensionless and bounds it between $-1$ and $1$.

- $\rho=1$ means a perfect positive linear relationship.
- $\rho=-1$ means a perfect negative linear relationship.
- $\rho=0$ means no linear correlation.

The last statement needs care. Zero Pearson correlation does not imply independence and does not rule out a strong nonlinear relationship. Correlation also does not establish causation.

For a concrete calculation, take

$$
X=(2,3,4,5,6),
\qquad
Y=(5,6,7,10,12).
$$

Their means are $4$ and $8$. The sum of cross-products of centered values is $18$, while the two centered sums of squares are $10$ and $34$. Therefore,

$$
r
=\frac{18}{\sqrt{10}\sqrt{34}}
\approx0.976,
$$

indicating a strong positive linear relationship. The normalization is what makes this value independent of the original measurement units.

---

## 8. Multicollinearity {#multicollinearity}

Multicollinearity means predictor columns contain overlapping linear information. In a regression model this can make coefficient estimates unstable: small changes in data can shift responsibility between correlated features while predictions remain similar.

Predictors do not need to be statistically independent for linear or logistic regression to work. The problem is exact or severe near-linear dependence, especially when the goal is coefficient interpretation.

Common responses include:

- remove or combine redundant variables;
- collect more data;
- use ridge or lasso regularization;
- replace correlated variables with principal components;
- report uncertainty rather than overinterpreting individual coefficients.

Variance inflation factors can diagnose how much a coefficient's variance increases because of the other predictors, but no universal cutoff replaces domain judgment.

---

## The inference workflow

The concepts form one sequence:

$$
\text{assumptions}
\rightarrow
\text{probability model}
\rightarrow
\text{likelihood}
\rightarrow
\text{estimate or posterior}
\rightarrow
\text{uncertainty}
\rightarrow
\text{decision}.
$$

Most statistical mistakes come from skipping one arrow: treating a likelihood as a posterior, a p-value as a hypothesis probability, correlation as causation, or an estimate as if it had no sampling uncertainty.

The goal is not merely to compute a number. It is to know what question that number answers.
