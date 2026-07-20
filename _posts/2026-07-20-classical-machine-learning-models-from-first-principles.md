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

The constrained form makes the geometry clearer:

$$
\min_\beta \|y-X\beta\|_2^2
\quad\text{subject to}\quad
\|\beta\|_2^2\le t.
$$

In two dimensions the feasible set is a circle. Its boundary is smooth, so a least-squares contour normally touches it away from an axis. Ridge therefore shrinks coefficients continuously but rarely makes them exactly zero. When two predictors carry nearly the same information, ridge can share weight between them rather than letting either coefficient become extremely large.

### Lasso: L1 regularization

Lasso solves

$$
\arg\min_\beta
\left(\|y-X\beta\|_2^2+\lambda\|\beta\|_1\right).
$$

Its diamond-shaped constraint has corners on the coordinate axes. As a loss contour expands until it touches the feasible region, it often meets a corner, making one or more coefficients exactly zero. The equivalent subgradient view says the L1 penalty has a threshold around zero rather than L2's smooth pull.

More formally, the constrained version uses $\|\beta\|_1\le t$. In two dimensions this is a diamond. At $\beta_j=0$, the subgradient of $|\beta_j|$ is the entire interval $[-1,1]$. If the data-fit gradient lies inside the interval supplied by the penalty, zero satisfies the optimality condition. This thresholding behavior is why lasso performs feature selection while ridge does not.

Lasso is useful for sparse feature selection, but correlated predictors can make its selections unstable. Elastic net combines both penalties:

$$
\lambda_1\|\beta\|_1+\lambda_2\|\beta\|_2^2.
$$

As a practical rule, prefer lasso when you expect only a small subset of features to matter and want a sparse, interpretable model. Prefer ridge when many features probably contribute, especially when predictors are correlated. Elastic net is useful when you want sparsity but also want correlated groups to be treated more stably than pure lasso often treats them.

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

Polynomial expansion can therefore create curved boundaries in the original space. The trade-offs are rapid feature growth, more computation, and greater overfitting risk.

In scikit-learn, a pipeline ensures that the same feature expansion is applied during both fitting and prediction:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LogisticRegression(),
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

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

### Worked Gaussian example

Suppose a one-dimensional problem has two classes:

$$
P(A)=0.6,
\quad X\mid A\sim\mathcal N(0,1),
$$

$$
P(B)=0.4,
\quad X\mid B\sim\mathcal N(2,1).
$$

For a new observation $x=1.5$, the two class-conditional densities are approximately

$$
P(x\mid A)=0.1295,
\qquad
P(x\mid B)=0.3521.
$$

Multiplying by the priors gives the unnormalized posterior scores

$$
s_A=0.6(0.1295)=0.0777,
\qquad
s_B=0.4(0.3521)=0.1408.
$$

Naive Bayes can classify immediately because $s_B>s_A$. If actual posterior probabilities are required, the evidence normalizes the scores:

$$
P(x)=s_A+s_B=0.2185,
$$

$$
P(A\mid x)\approx0.356,
\qquad
P(B\mid x)\approx0.644.
$$

For several features, repeat the density calculation for every feature and multiply the results under the conditional-independence assumption—or, more safely, add their logarithms.

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

This objective is the **within-cluster sum of squares (WCSS)**, also called **inertia**. Its symbols mean:

- $K$ is the number of clusters;
- $C_k$ is the set of points assigned to cluster $k$;
- $x_i$ is one point in that cluster;
- $\mu_k$ is the cluster's centroid;
- $\|x_i-\mu_k\|_2^2$ is the squared Euclidean distance from the point to its centroid.

WCSS adds these squared point-to-centroid distances over every cluster. Squaring prevents positive and negative coordinate differences from cancelling and makes distant assignments especially expensive.

It alternates two steps:

1. assign every point to its nearest centroid;
2. replace each centroid with the mean of its assigned points.

Each step cannot increase the objective, so the algorithm converges—but generally to a local optimum. K-means++ initialization and multiple restarts reduce sensitivity to the initial centroids.

### One complete iteration

Consider the one-dimensional observations

$$
X=\{1,2,8,9\},
\qquad K=2,
$$

with initial centroids $\mu_1=1$ and $\mu_2=8$.

1. **Assignment:** $1$ and $2$ are closer to $\mu_1$; $8$ and $9$ are closer to $\mu_2$.
2. **Update:** the new centroids are

   $$
   \mu_1=\frac{1+2}{2}=1.5,
   \qquad
   \mu_2=\frac{8+9}{2}=8.5.
   $$

3. **Repeat:** reassignment produces the same two clusters, so the algorithm has converged.

In practice, convergence can mean unchanged assignments, centroid movement below a tolerance, objective improvement below a tolerance, or reaching an iteration limit. Convergence does not imply the global minimum: different initial centroids can produce different partitions.

The method prefers compact, roughly spherical, similarly sized clusters. Scaling matters because Euclidean distance gives large-scale features more influence. Outliers can pull centroids strongly.

An iteration costs roughly $O(nKd)$ for $n$ observations, $K$ clusters, and $d$ features; over $I$ iterations the cost is $O(nKdI)$. Mini-batch K-means reduces the cost on very large datasets. High-dimensional distances can also become less informative, so dimensionality reduction or a different similarity model may be appropriate.

Two implementation edge cases matter:

- If a cluster receives no points, reinitialize its centroid—often to a distant point—or use the library's built-in recovery strategy.
- Because squared distance is sensitive to outliers, robust scaling, trimming, or a medoid-based method may be safer for contaminated data.

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

The elbow method compares how WCSS changes as the number of clusters increases:

1. Choose a range of candidate values for $K$.
2. Fit K-means separately for every candidate, preferably with multiple initializations.
3. Record the final WCSS for each $K$.
4. Plot WCSS against $K$.
5. Look for the elbow: the point after which another cluster produces only a modest reduction in WCSS.

WCSS can never increase merely because $K$ increases: more centroids give the model more freedom. The elbow method therefore looks for diminishing returns rather than the absolute minimum, which would trivially occur at the largest permitted $K$.

The silhouette value for point $i$ is

$$
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))},
$$

where $a(i)$ is average distance within its cluster and $b(i)$ is the smallest average distance to another cluster.

The overall silhouette score is the mean over all $n$ observations:

$$
S=\frac1n\sum_{i=1}^n s(i).
$$

- values near $1$ indicate a cohesive, separated assignment;
- values near $0$ indicate a boundary;
- negative values suggest a potentially poor assignment.

Neither the elbow nor silhouette method discovers an objectively “true” $K$. Clusters are useful only when their geometry and meaning fit the problem.

Use the two diagnostics differently. The elbow asks whether another cluster buys enough reduction in WCSS; the silhouette asks whether the resulting assignments are simultaneously cohesive and separated. If they disagree, inspect cluster stability across resampled data and judge whether the partition is meaningful for the actual application.

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

### From directions to a reduced representation

Let $W_r=[w_1,\ldots,w_r]$ contain the first $r$ eigenvectors. The reduced coordinates, often called component scores, are

$$
Z=XW_r.
$$

An approximation in the original feature space is

$$
\hat X=ZW_r^T=XW_rW_r^T.
$$

Each eigenvalue $\lambda_j$ is the variance captured by component $j$. Its explained-variance ratio is

$$
\frac{\lambda_j}{\sum_k\lambda_k}.
$$

A common choice of $r$ is the smallest number of components whose cumulative ratio reaches a target such as $90\%$ or $95\%$. That target is a compression decision, not a universal statistical rule; downstream validation may prefer a different value.

If components are whitened, each score is additionally divided by $\sqrt{\lambda_j}$. Whitening removes scale and correlation between retained components, but it can amplify low-variance noise and discards information about their relative importance.

### Small numerical example

Take the already centered observations

$$
X=
\begin{bmatrix}
-2&-1\\
0&0\\
2&1
\end{bmatrix}.
$$

Their covariance matrix is

$$
S=\frac13X^TX
=\begin{bmatrix}
8/3&4/3\\
4/3&2/3
\end{bmatrix}.
$$

Its first eigenvector is $w_1=(2,1)^T/\sqrt{5}$ with eigenvalue $10/3$; the second eigenvalue is zero. Projecting onto one component gives

$$
Z=Xw_1=
\begin{bmatrix}
-\sqrt{5}\\
0\\
\sqrt{5}
\end{bmatrix}.
$$

All observations lie on the line spanned by $(2,1)$, so the first component explains $100\%$ of the variance and $ZW_1^T$ reconstructs the data exactly. Real datasets generally leave nonzero residual variance, making the choice of $r$ consequential.

PCA is unsupervised: it preserves high variance, not necessarily information useful for prediction. Standardize features when their units should not determine the components.

For the deeper geometry—projections, eigenvectors, spectral decomposition, and SVD—see [Linear Algebra: Important Concepts and the Doubts That Connect Them](/posts/linear-algebra-important-concepts/). That article develops why symmetric matrices admit orthogonal eigendirections, how spectral decomposition reconstructs a matrix, and how SVD extends the geometry to rectangular matrices.

---

## 9. Choosing evaluation metrics {#metrics}

For a positive class,

$$
\text{precision}=\frac{TP}{TP+FP},
\qquad
\text{recall}=\frac{TP}{TP+FN}.
$$

Two related quantities are

$$
\text{specificity}=\frac{TN}{TN+FP},
\qquad
\text{false-positive rate}=1-\text{specificity}.
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

### Thresholds, prevalence, and calibration

A probability model becomes a hard classifier only after choosing a threshold. Lowering the threshold usually increases recall and false positives; raising it usually increases precision and false negatives. The correct operating point depends on costs and capacity, not on a default of $0.5$.

Suppose 1,000 transactions contain 20 fraud cases. A detector finds 18 frauds but flags 42 legitimate transactions:

$$
\text{recall}=\frac{18}{20}=0.90,
\qquad
\text{precision}=\frac{18}{18+42}=0.30.
$$

High recall and modest precision can both be appropriate if missing fraud is much more expensive than reviewing an alert. The same sensitivity and specificity can yield different precision when deployment prevalence changes, so evaluation should reflect the population in which the model will operate.

Ranking and calibration answer different questions. A model may rank positive examples above negative ones and achieve strong AUC while its stated probabilities are systematically too high or too low. Reliability diagrams, log loss, and the Brier score assess probability quality; threshold metrics assess decisions made from those probabilities.

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
