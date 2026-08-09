---
title: "Convex Quadratics, PSD Matrices, and the Schur Complement"
date: 2026-08-08 10:10:00 +0530
categories: [Mathematics, Optimization]
tags: [quadratic-forms, positive-semidefinite, schur-complement, convexity, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/convex-quadratics-psd-schur-complement/
description: "Why PSD matrices characterize convex quadratics, why only the symmetric part matters, and how the Schur complement enters optimization."
---

Quadratic objectives are the first place where linear algebra and convex optimization truly meet. Their curvature is stored in a matrix, positive semidefiniteness tells us whether that curvature points upward, and the Schur complement lets us reason about coupled block variables.

This chapter keeps the key doubts close to the definitions: Is $P$ one matrix or the PSD cone? Why does the skew-symmetric part disappear? What fails when $P$ is not PSD? Why does a block quadratic lead to a Schur complement?

## 1. What does a convex quadratic mean? {#convex-quadratics}

> **Doubt:** I think $P$ is a positive-semidefinite matrix or cone.

A quadratic function has the form

$$
f(x)=\frac12x^TPx+q^Tx+r.
$$

Here:

- $x\in\mathbb R^n$;
- $P\in\mathbb R^{n\times n}$;
- $q\in\mathbb R^n$;
- $r\in\mathbb R$.

Assume $P$ is symmetric. Then

$$
\nabla f(x)=Px+q,
\qquad
\nabla^2f(x)=P.
$$

A twice-differentiable function is convex when its Hessian is positive semidefinite everywhere. Therefore,

$$
f\text{ is convex}\iff P\succeq0.
$$

### What does positive semidefinite mean?

For the optimization argument, we need the condition

$$
z^TPz\ge0\qquad\text{for every }z.
$$

This says that the quadratic has nonnegative curvature in every direction. The eigenvalue characterization and full PSD theory are explained in [Positive semidefinite matrices](/posts/symmetric-matrices-quadratic-forms-matrix-norm/#positive-semidefinite).

### Is P a matrix or a cone?

$P$ is one matrix.

The notation

$$
P\in\mathbf S_+^n
$$

means that $P$ belongs to the set of all symmetric positive-semidefinite $n\times n$ matrices.

The set $\mathbf S_+^n$ is a convex cone.

The proof that this set is closed under nonnegative combinations is kept in [Why do PSD matrices form a convex cone?](/posts/symmetric-matrices-quadratic-forms-matrix-norm/#psd-convex-cone). The precise distinction here is:

- $P$ is a PSD matrix;
- $\mathbf S_+^n$ is the PSD cone.

### Why the factor one-half?

It is only a convenience:

$$
\nabla\left(\frac12x^TPx\right)=Px
$$

for symmetric $P$.

---

## 2. What happens when P is not PSD?

Take

$$
P=\begin{bmatrix}1&0\\0&-1\end{bmatrix}.
$$

Then

$$
f(x)=\frac12(x_1^2-x_2^2).
$$

Along the $x_1$ direction it curves upward. Along the $x_2$ direction it curves downward. The graph is a saddle, not a bowl, so the function is not convex.

If $P\succ0$, the quadratic is strictly convex and has a unique unconstrained minimizer.

If $P\succeq0$ but is singular, the function may be flat in some directions, so minimizers need not be unique.

---

## 3. Why can every quadratic form use a symmetric matrix? {#symmetric-quadratics}

For any square matrix $P$,

$$
x^TPx
=
x^T\left(\frac{P+P^T}{2}\right)x.
$$

The skew-symmetric part contributes zero. The complete cancellation proof is kept in [Must the matrix in a quadratic form be symmetric?](/posts/symmetric-matrices-quadratic-forms-matrix-norm/#symmetric-part-quadratic-form).

The optimization consequence is worth stating here. For

$$
f(x)=\frac12x^TPx+q^Tx+r
$$

with a possibly nonsymmetric $P$,

$$
\nabla f(x)
=
\frac12(P+P^T)x+q,
$$

and

$$
\nabla^2f(x)
=
\frac{P+P^T}{2}.
$$

Thus the skew-symmetric part affects neither the function nor its derivatives. The correct convexity condition is

$$
\frac{P+P^T}{2}\succeq0.
$$

After replacing $P$ by its symmetric part, we may rename that representative $P$ and write the familiar condition

$$
P\succeq0.
$$

---

## 4. Block quadratic forms and the Schur complement

The full block-matrix Schur-complement theorem and its congruence proof are given in [The Schur complement](/posts/conic-optimization-socp-sdp/#the-schur-complement). Here we keep only its optimization-specific interpretation.

Consider

$$
f(x,y)=x^TAx+2x^TBy+y^TCy.
$$

It can be written as

$$
\begin{bmatrix}x\\y\end{bmatrix}^T
\begin{bmatrix}A&B\\B^T&C\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}.
$$

The function is jointly convex in $(x,y)$ exactly when

$$
\begin{bmatrix}A&B\\B^T&C\end{bmatrix}\succeq0.
$$

If $C\succ0$, minimize over $y$:

$$
\nabla_y f=2B^Tx+2Cy=0,
$$

so

$$
y^\star=-C^{-1}B^Tx.
$$

Substituting back gives

$$
\min_y f(x,y)
=x^T(A-BC^{-1}B^T)x.
$$

The matrix

$$
A-BC^{-1}B^T
$$

is the Schur complement of $C$.

The intuition is that optimizing out $y$ leaves an effective quadratic curvature in $x$.

---

For a deeper treatment of symmetric matrices, definiteness, eigenvalues, and matrix norms, continue with [Symmetric Matrices, Quadratic Forms, and Matrix Norms](/posts/symmetric-matrices-quadratic-forms-matrix-norm/).

---

This article is part of the [Convex Optimization learning map](/posts/convex-optimization-doubt-log/).

## Sources and further study

- Stephen Boyd and Lieven Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
- Stephen Boyd, [EE364A lecture slides](https://web.stanford.edu/class/ee364a/lectures.html).
