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

It means

$$
z^TPz\ge0\qquad\text{for every }z.
$$

Equivalently, when $P$ is symmetric, all eigenvalues of $P$ are nonnegative.

Geometrically, the quadratic has nonnegative curvature in every direction.

### Is P a matrix or a cone?

$P$ is one matrix.

The notation

$$
P\in\mathbf S_+^n
$$

means that $P$ belongs to the set of all symmetric positive-semidefinite $n\times n$ matrices.

The set $\mathbf S_+^n$ is a convex cone.

Why is it a cone? If $P\succeq0$ and $\alpha\ge0$, then

$$
z^T(\alpha P)z=\alpha z^TPz\ge0.
$$

Why is it convex? If $P,Q\succeq0$ and $0\le\theta\le1$, then

$$
z^T(\theta P+(1-\theta)Q)z
=\theta z^TPz+(1-\theta)z^TQz\ge0.
$$

So the precise statement is:

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

## 3. Why can every quadratic form be represented by a symmetric matrix? {#symmetric-quadratics}

The phrase "we can assume that $P$ is symmetric" can be misleading. We are not proving that the original matrix $P$ is symmetric. Instead, we are showing that every possibly nonsymmetric matrix can be replaced by a symmetric matrix without changing the quadratic function.

Start with a general square matrix

$$
P\in\mathbb{R}^{n\times n}.
$$

It can always be decomposed as

$$
P=H+K,
$$

where

$$
H=\frac{P+P^T}{2}
$$

and

$$
K=\frac{P-P^T}{2}.
$$

The matrix $H$ is symmetric because

$$
H^T
=\left(\frac{P+P^T}{2}\right)^T
=\frac{P^T+P}{2}
=H.
$$

The matrix $K$ is skew-symmetric because

$$
K^T
=\left(\frac{P-P^T}{2}\right)^T
=\frac{P^T-P}{2}
=-K.
$$

Thus every square matrix is the sum of a symmetric part and a skew-symmetric part.

### Why does the skew-symmetric part disappear?

Consider the scalar

$$
a=x^TKx.
$$

Because $a$ is a scalar, transposing it does not change it:

$$
a=a^T.
$$

Now transpose the complete product:

$$
a^T
=(x^TKx)^T
=x^TK^Tx.
$$

Since $K^T=-K$,

$$
a^T=x^T(-K)x=-x^TKx=-a.
$$

We have therefore shown both

$$
a=a^T
$$

and

$$
a^T=-a.
$$

Hence $a=-a$, which is possible only when

$$
a=0.
$$

Therefore every skew-symmetric matrix satisfies

$$
x^TKx=0
$$

for every vector $x$.

### Apply this result to the quadratic form

Using $P=H+K$,

$$
x^TPx
=x^T(H+K)x
=x^THx+x^TKx.
$$

The second term is zero, so

$$
x^TPx=x^THx
=x^T\left(\frac{P+P^T}{2}\right)x.
$$

This is the precise meaning of "we can assume $P$ is symmetric": the original $P$ need not be symmetric, but its symmetric part represents exactly the same quadratic form.

### Concrete example

Take the nonsymmetric matrix

$$
P=
\begin{bmatrix}
1&1\\
-1&1
\end{bmatrix}.
$$

Its symmetric part is

$$
H
=\frac{P+P^T}{2}
=\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
=I,
$$

and its skew-symmetric part is

$$
K
=\frac{P-P^T}{2}
=\begin{bmatrix}
0&1\\
-1&0
\end{bmatrix}.
$$

For

$$
x=\begin{bmatrix}x_1\\x_2\end{bmatrix},
$$

direct expansion gives

$$
x^TPx
=x_1^2+x_1x_2-x_1x_2+x_2^2
=x_1^2+x_2^2.
$$

The two cross terms cancel. This is exactly the same expression obtained from the symmetric part:

$$
x^THx=x^TIx=x_1^2+x_2^2.
$$

### What happens to the gradient and Hessian?

For a general matrix $P$, consider

$$
f(x)=\frac12x^TPx+q^Tx+r.
$$

Before making any symmetry assumption, the gradient is

$$
\nabla f(x)
=\frac12(P+P^T)x+q,
$$

and the Hessian is

$$
\nabla^2f(x)=\frac{P+P^T}{2}=H.
$$

The Hessian is necessarily symmetric, and the skew-symmetric part of $P$ has no effect on either the function or its derivatives.

After replacing $P$ by $H$, we may rename $H$ as $P$. Then $P=P^T$, and the familiar formulas become

$$
\nabla f(x)=Px+q,
\qquad
\nabla^2f(x)=P.
$$

Therefore the correct convexity condition for a quadratic written with an arbitrary matrix is

$$
\frac{P+P^T}{2}\succeq0.
$$

Once the symmetric representative is used, this is written more simply as

$$
P\succeq0.
$$

The key conclusion is not that every matrix is symmetric. It is that every quadratic form has an equivalent symmetric matrix representation, and only that symmetric part determines its curvature and convexity.

---

## 4. Block quadratic forms and the Schur complement

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
