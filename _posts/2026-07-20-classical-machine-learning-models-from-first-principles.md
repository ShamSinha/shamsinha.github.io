---
title: "Classical Machine Learning Models from First Principles"
date: 2026-07-20 11:30:00 +0530
categories: [Machine Learning, Classical ML]
tags: [linear-regression, logistic-regression, regularization, naive-bayes, decision-trees, svm, k-means, pca, evaluation-metrics, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/classical-machine-learning-models-from-first-principles/
description: "A first-principles guide to regression, regularization, logistic classification, Naive Bayes, decision trees, SVMs, K-means, PCA, and model evaluation."
---

Classical machine-learning models are often presented as a list of unrelated algorithms. They become much easier to remember when organized by the assumptions they make.

- Regression assumes a structured relationship between features and a target.
- Naive Bayes assumes conditional independence.
- Decision trees assume useful predictions can be built from recursive partitions.
- Support vector machines search for a large-margin boundary.
- K-means assumes clusters can be represented by Euclidean centroids.
- PCA assumes high-variance linear directions preserve useful structure.

This post contains the model-building half of my original ML/AI mathematics study notes. Its statistical foundations are developed in [Probability and Statistical Inference for Machine Learning](/posts/probability-statistical-inference-for-machine-learning/).

## Topic map

- [Linear regression](#linear-regression)
- [Ridge and lasso regularization](#regularization)
- [Logistic regression](#logistic-regression)
- [Naive Bayes](#naive-bayes)
- [Decision trees](#decision-trees)
- [Support vector machines](#svm)
- [K-means clustering](#clustering)
- [PCA and Lagrange multipliers](#pca)
- [Choosing evaluation metrics](#metrics)

---

## 1. Linear regression {#linear-regression}

Linear regression models

$$
y=X\beta+\varepsilon,
$$

where $X\in\mathbb R^{n\times d}$ contains features, $\beta$ contains coefficients, and $\varepsilon$ represents unexplained variation.

Ordinary least squares chooses

$$
\hat\beta
=\arg\min_\beta\|y-X\beta\|_2^2.
$$

Differentiating gives the normal equations:

$$
X^TX\hat\beta=X^Ty.
$$

When $X^TX$ is invertible,

$$
\hat\beta=(X^TX)^{-1}X^Ty.
$$

In practice, QR or SVD-based solvers are numerically safer than explicitly constructing this inverse.

### What does “linear” mean?

The model is linear in its coefficients, not necessarily in the raw input. For example,

$$
y=\beta_0+\beta_1x+\beta_2x^2+\varepsilon
$$

is still a linear regression model because $\beta_0,\beta_1,\beta_2$ enter linearly. Feature transformations can therefore express curves while retaining linear estimation.

### Statistical assumptions

The familiar coefficient formulas and uncertainty estimates rely on assumptions about the errors, commonly:

- linearity of the conditional mean;
- independent observations;
- constant conditional error variance;
- no exact linear dependence between predictor columns;
- zero conditional mean error.

Normal errors are needed for small-sample exact tests, not for the least-squares coefficient calculation itself.

---

## 2. Ridge and lasso regularization {#regularization}

Regularization trades some training fit for lower variance and better behavior on unseen data.

### Ridge: L2 regularization

Ridge regression solves

$$
\arg\min_\beta
\left(\|y-X\beta\|_2^2+\lambda\|\beta\|_2^2\right).
$$

Its solution is

$$
\hat\beta_{\mathrm{ridge}}
=(X^TX+\lambda I)^{-1}X^Ty.
$$

Adding $\lambda I$ improves conditioning and smoothly shrinks coefficients. Ridge is a good fit when many predictors may carry some signal, including groups of correlated predictors.

### Lasso: L1 regularization

Lasso solves

$$
\arg\min_\beta
\left(\|y-X\beta\|_2^2+\lambda\|\beta\|_1\right).
$$

Its diamond-shaped constraint has corners on the coordinate axes. As a loss contour expands until it touches the feasible region, it often meets a corner, making one or more coefficients exactly zero. The equivalent subgradient view says the L1 penalty has a threshold around zero rather than L2's smooth pull.

Lasso is useful for sparse feature selection, but correlated predictors can make its selections unstable. Elastic net combines both penalties:

$$
\lambda_1\|\beta\|_1+\lambda_2\|\beta\|_2^2.
$$

The nested Notion discussion links to this comparison of [lasso and ridge](https://chatgpt.com/share/66efe63c-c230-8000-a808-8f5233cbd183).

> Standardize features before comparing L1 or L2 penalties. Otherwise, measurement units determine how expensive a coefficient appears.

---

## 3. Logistic regression is a classifier {#logistic-regression}

Despite its name, logistic regression models a class probability. For binary classification,

$$
P(y=1\mid x)=\sigma(w^Tx+b)
=\frac{1}{1+e^{-(w^Tx+b)}}.
$$

Taking odds and log-odds gives

$$
\log\frac{p}{1-p}=w^Tx+b.
$$

Logistic regression is therefore linear in the log-odds. With threshold $p=0.5$, its decision boundary is

$$
w^Tx+b=0,
$$

a hyperplane in the supplied feature space.

The model is usually trained by minimizing binary cross-entropy:

$$
L(w,b)
=-\sum_i
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right].
$$

This is the negative log-likelihood of independent Bernoulli observations.

### Can logistic regression model nonlinear boundaries?

It remains linear in its parameters, but its features can be nonlinear functions of the original input. A circular boundary

$$
x_1^2+x_2^2=c
$$

becomes linear after defining $z_1=x_1^2$ and $z_2=x_2^2$:

$$
z_1+z_2=c.
$$

Polynomial expansion can therefore create curved boundaries in the original space. The trade-offs are rapid feature growth, more computation, and greater overfitting risk. This was the key progression in the linked [logistic-regression limitations conversation](https://chatgpt.com/share/66ebeb44-7e90-8000-844b-66bec808fc58).

---

## 4. Naive Bayes {#naive-bayes}

Bayes' rule for class $c$ and features $x=(x_1,\ldots,x_d)$ is

$$
P(c\mid x)
=\frac{P(c)P(x\mid c)}{P(x)}.
$$

Because $P(x)$ is common to every candidate class, classification uses

$$
\hat c
=\arg\max_c P(c)P(x\mid c).
$$

Naive Bayes assumes the features are conditionally independent given the class:

$$
P(x\mid c)=\prod_{j=1}^dP(x_j\mid c).
$$

Computations use log-probabilities to turn products into sums and avoid underflow:

$$
\hat c
=\arg\max_c
\left[
\log P(c)+\sum_j\log P(x_j\mid c)
\right].
$$

Common variants are:

- **Gaussian Naive Bayes:** continuous features modeled with class-conditional Gaussians;
- **Multinomial Naive Bayes:** counts, often word counts;
- **Bernoulli Naive Bayes:** binary feature presence.

For Gaussian Naive Bayes, each feature and class has a learned mean $\mu_{jc}$ and variance $\sigma_{jc}^2$:

$$
P(x_j\mid c)
=\frac{1}{\sqrt{2\pi\sigma_{jc}^2}}
\exp\left(
-\frac{(x_j-\mu_{jc})^2}{2\sigma_{jc}^2}
\right).
$$

Training estimates the class prior and these per-class feature statistics. Prediction evaluates the density of every observed feature under each class and adds the resulting log-likelihoods to the log prior.

The independence assumption is often false, yet classification may still work well because correct class ranking does not require perfectly estimated joint probabilities. Probability calibration can nevertheless be poor.

---

## 5. Decision trees {#decision-trees}

A decision tree recursively splits the feature space. At each node it searches for a feature and threshold that make the child nodes purer than the parent.

For classification, common impurity measures are entropy

$$
H(S)=-\sum_c p_c\log p_c
$$

and Gini impurity

$$
G(S)=1-\sum_c p_c^2.
$$

A split is useful when its weighted child impurity is lower than the parent's impurity.

Trees have several appealing properties:

- they learn nonlinear relationships and feature interactions;
- they do not require feature scaling;
- their decisions can be inspected as rules;
- they work with mixed feature behavior.

Their weakness is variance. A small change in training data can produce a different deep tree. Depth limits, minimum leaf sizes, cost-complexity pruning, and ensembles manage this tendency.

- Random forests reduce variance by averaging decorrelated trees.
- Gradient boosting builds trees sequentially to correct residual errors.

---

## 6. Support vector machines {#svm}

For binary labels $y_i\in\{-1,1\}$, a hard-margin SVM seeks a separating hyperplane with maximum geometric margin:

$$
\min_{w,b}\frac12\|w\|_2^2
$$

subject to

$$
y_i(w^Tx_i+b)\ge1.
$$

Real data is rarely perfectly separable. The soft-margin objective is

$$
\min_{w,b}
\frac12\|w\|^2
+C\sum_i\max(0,1-y_i(w^Tx_i+b)).
$$

The first term favors a wide margin; hinge loss penalizes points inside the margin or on the wrong side. $C$ controls the trade-off.

Only points on or inside the margin determine the final boundary. These are the support vectors.

### Kernel methods

The dual form depends on inner products $x_i^Tx_j$. Replacing them with a kernel

$$
K(x_i,x_j)=\phi(x_i)^T\phi(x_j)
$$

allows a linear separator in an implicit feature space to represent a nonlinear boundary in the original space. RBF and polynomial kernels are common examples.

Feature scaling matters because distances and dot products determine the solution. Kernel SVMs can also become expensive as the number of training examples grows.

---

## 7. K-means clustering {#clustering}

K-means partitions points into $K$ clusters by minimizing within-cluster squared distance:

$$
\min_{\{C_k,\mu_k\}}
\sum_{k=1}^K\sum_{x_i\in C_k}\|x_i-\mu_k\|_2^2.
$$

It alternates two steps:

1. assign every point to its nearest centroid;
2. replace each centroid with the mean of its assigned points.

Each step cannot increase the objective, so the algorithm converges—but generally to a local optimum. K-means++ initialization and multiple restarts reduce sensitivity to the initial centroids.

The method prefers compact, roughly spherical, similarly sized clusters. Scaling matters because Euclidean distance gives large-scale features more influence. Outliers can pull centroids strongly.

Here is the Python example from the source notes:

```python
import numpy as np
from sklearn.cluster import KMeans


# Create sample two-dimensional data.
rng = np.random.default_rng(42)
X = rng.random((100, 2))

# Fit three clusters. Explicit n_init keeps behavior stable across versions.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
```

### Choosing $K$

The elbow method plots within-cluster sum of squares against $K$ and looks for diminishing returns.

The silhouette value for point $i$ is

$$
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))},
$$

where $a(i)$ is average distance within its cluster and $b(i)$ is the smallest average distance to another cluster.

- values near $1$ indicate a cohesive, separated assignment;
- values near $0$ indicate a boundary;
- negative values suggest a potentially poor assignment.

Neither the elbow nor silhouette method discovers an objectively “true” $K$. Clusters are useful only when their geometry and meaning fit the problem.

---

## 8. PCA and Lagrange multipliers {#pca}

Principal component analysis finds orthogonal directions of maximum variance. Center the data matrix $X$ and define covariance

$$
S=\frac1nX^TX.
$$

The first principal direction solves

$$
\max_{\|w\|_2=1}w^TSw.
$$

The norm constraint prevents the objective from growing merely by scaling $w$. Introduce a Lagrange multiplier:

$$
\mathcal L(w,\lambda)
=w^TSw-\lambda(w^Tw-1).
$$

Setting the derivative with respect to $w$ to zero gives

$$
2Sw-2\lambda w=0,
$$

so

$$
Sw=\lambda w.
$$

The principal directions are eigenvectors of the covariance matrix, ordered by eigenvalue. Equivalently, they are the right singular vectors of centered $X$.

PCA is unsupervised: it preserves high variance, not necessarily information useful for prediction. Standardize features when their units should not determine the components.

For the deeper geometry—projections, eigenvectors, spectral decomposition, and SVD—see [Linear Algebra: Important Concepts and the Doubts That Connect Them](/posts/linear-algebra-important-concepts/). That article already develops the ideas referenced by the Notion page's [spectral-decomposition conversation](https://chat.openai.com/share/2cc34c6c-cbcf-4a4b-bf6d-b75520637a82) and [SVD conversation](https://chat.openai.com/share/2a561c2e-d049-4f66-b021-c2151a5eb6e9).

---

## 9. Choosing evaluation metrics {#metrics}

For a positive class,

$$
\text{precision}=\frac{TP}{TP+FP},
\qquad
\text{recall}=\frac{TP}{TP+FN}.
$$

- Prefer **precision** when false positives are expensive—for example, sending legitimate email to spam.
- Prefer **recall** when false negatives are expensive—for example, missing a fraud case or serious disease.

Their harmonic mean is

$$
F_1
=2\frac{\text{precision}\cdot\text{recall}}
{\text{precision}+\text{recall}}.
$$

### ROC versus precision-recall

ROC curves plot true-positive rate against false-positive rate across thresholds. Precision-recall curves plot precision against recall.

ROC-AUC measures ranking across both classes, but it can look optimistic when negatives vastly outnumber positives. PR-AUC focuses more directly on positive-class performance in imbalanced settings. Neither metric chooses an operating threshold or verifies probability calibration.

### Start from consequences

Choose a metric by answering:

1. Which class matters?
2. What is the cost of each error type?
3. Is the target ranking, probability calibration, or a thresholded decision?
4. Will deployment prevalence match the evaluation set?
5. Must performance be acceptable within important subgroups?

A model is only “better” relative to an explicitly chosen consequence.

---

## Choosing a model by its assumptions

| Model | Central assumption or bias | Common strength | Common failure mode |
|---|---|---|---|
| Linear regression | Linear conditional mean | Interpretable baseline | Missed nonlinear structure |
| Logistic regression | Linear log-odds in supplied features | Probabilistic classification | Inadequate feature map |
| Naive Bayes | Conditional feature independence | Fast with little data | Poor probability calibration |
| Decision tree | Recursive axis-aligned partitions | Nonlinear interactions | High variance |
| SVM | Large-margin separation | Strong medium-sized classifier | Scaling and computational cost |
| K-means | Euclidean centroid clusters | Simple unsupervised grouping | Non-spherical clusters and outliers |
| PCA | High variance is useful | Linear compression | Discards low-variance signal |

The practical habit is to make these assumptions explicit before comparing scores. An algorithm is not merely a formula; it is a claim about the structure of the data.
