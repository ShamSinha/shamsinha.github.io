---
title: "Quadratic and Constrained Convex Optimization: Curvature, Geometry, and Optimality"
date: 2026-08-08 10:10:00 +0530
last_modified_at: 2026-08-09 00:00:00 +0530
categories: [Mathematics, Optimization]
tags: [quadratic-forms, positive-semidefinite, linear-programming, quadratic-programming, optimality, constraints, schur-complement, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/convex-quadratics-psd-schur-complement/
description: "How matrix curvature, LP and QP geometry, feasible directions, equality constraints, and the Schur complement fit together."
---

Quadratic objectives are the first place where linear algebra and convex optimization truly meet. Their curvature is stored in a matrix, while constraints determine which directions we are actually allowed to move. Understanding a constrained quadratic problem therefore requires both viewpoints: the shape of the objective and the geometry of the feasible set.

## Why this chapter exists

Knowing that $P\succeq0$ makes a quadratic convex is only the beginning. We still need to understand why an LP can have a whole optimal face, why a QP can select a point in the middle of that face, and why the familiar equation $\nabla f(x^\star)=0$ changes under constraints.

This chapter keeps those ideas together because they answer one question:

> How do the curvature of an objective and the feasible directions created by constraints determine an optimum?

## Before you start

Read [Convex Sets, Functions, and Quasiconvex Optimization](/posts/convex-sets-functions-first-order-geometry/) first for convex sets, convex functions, and first-order geometry. For matrices, eigenvalues, nullspaces, ranges, and orthogonality, use [Linear Algebra: Important Concepts and the Doubts That Connect Them](/posts/linear-algebra-important-concepts/) as a reference. The more detailed proofs about symmetric matrices and PSD matrices live in [Symmetric Matrices, Quadratic Forms, and Matrix Norm](/posts/symmetric-matrices-quadratic-forms-matrix-norm/).

## What you will learn

The argument proceeds in four stages:

1. A quadratic's Hessian explains its curvature.
2. LP and QP level sets explain where their optima can occur.
3. Feasible directions produce the correct constrained first-order condition.
4. Eliminating variables from a block quadratic produces the Schur complement.

The key doubts remain beside the relevant definitions: Is $P$ one matrix or the PSD cone? Why does the skew-symmetric part disappear? Must an LP optimum be a vertex? Why can a QP optimum lie inside a face? Why does equality feasibility lead to a nullspace? Why does a block quadratic lead to a Schur complement?

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

## 4. LP optimum: must it always be a vertex? {#lp-qp-geometry}

> **Doubt:** In a linear program, the solution is always present on a vertex of a face of the polyhedron, right?

The careful statement is:

> If an LP has an attained optimum and the feasible polyhedron has extreme points relevant to the optimum, then there exists at least one optimal extreme point.

It is not true that every optimal solution must be a vertex.

### Example: an entire edge is optimal

Minimize

$$
x_1
$$

subject to

$$
0\le x_1\le1,
\qquad
0\le x_2\le1.
$$

Every point with $x_1=0$ is optimal:

$$
\{(0,x_2)\mid0\le x_2\le1\}.
$$

This includes the vertices $(0,0)$ and $(0,1)$, but also the interior point $(0,1/2)$ of the edge.

So:

- the optimal set may be a face;
- at least one vertex of that face is optimal.

### Why can we move to a vertex without changing a linear objective?

If an optimal point $x$ is a convex combination

$$
x=\theta u+(1-\theta)v,
$$

then

$$
c^Tx=\theta c^Tu+(1-\theta)c^Tv.
$$

If $x$ has the minimum value, neither $u$ nor $v$ can have a smaller value. Therefore both must also be optimal. Repeating this movement leads to an extreme point, provided the relevant face contains one.

### Important caveat

Some polyhedra contain no vertices. For example, an affine line is a polyhedron but has no extreme points. A constant objective on that line has an optimum everywhere but no optimal vertex. Thus the slogan needs its usual assumptions.

---

## 5. Why can a quadratic-program solution be inside a face or in the interior?

A convex QP has objective

$$
\frac12x^TPx+q^Tx+r,
\qquad P\succeq0.
$$

Its level sets are curved rather than parallel hyperplanes.

Therefore the first contact with a feasible set can occur:

- at a vertex;
- in the relative interior of an edge or face;
- strictly inside the feasible region.

### Interior optimum example

Minimize

$$
x_1^2+x_2^2
$$

over

$$
[-1,1]^2.
$$

The minimum is at

$$
(0,0),
$$

which lies strictly inside the square.

The unconstrained stationary condition is

$$
\nabla f(x)=0.
$$

If that stationary point is feasible, it can be the constrained optimum without touching any boundary.

### Face-interior example

Minimize

$$
(x_1-2)^2+x_2^2
$$

over the square $[-1,1]^2$.

The unconstrained minimizer is $(2,0)$, which is outside. The closest feasible point is $(1,0)$, lying in the interior of the right edge, not at a vertex.

### Main contrast

A linear objective has flat level sets. If one interior point of a face is optimal, the objective is flat along the relevant face, so vertices can also be optimal.

A strictly convex quadratic has curved level sets. Curvature can select one unique point in the middle of a face or in the interior.

---

## 6. First-order optimality over a convex feasible set {#constrained-optimality}

For a differentiable convex function over a convex set $X$, $x^\star$ is optimal if and only if

$$
\nabla f(x^\star)^T(y-x^\star)\ge0
\qquad\text{for every }y\in X.
$$

The vector $y-x^\star$ points from the candidate toward another feasible point.

If the inner product were negative for some feasible $y$, a small step toward $y$ would reduce the objective.

At an interior optimum, all sufficiently small directions are feasible. Both $v$ and $-v$ are allowed, forcing

$$
\nabla f(x^\star)=0.
$$

At a boundary optimum, the gradient need not vanish. It must point so that no feasible direction is a descent direction.

---

## 7. Equality constraints and the nullspace

Consider

$$
\min f(x)
\qquad\text{subject to}\qquad Ax=b.
$$

If $x$ and $y$ are both feasible, then

$$
Ax=b,
\qquad
Ay=b.
$$

Subtract:

$$
A(y-x)=0.
$$

Thus every feasible displacement

$$
v=y-x
$$

belongs to the nullspace $\mathcal N(A)$.

The first-order optimality condition becomes

$$
\nabla f(x)^Tv\ge0
\qquad\text{for every }v\in\mathcal N(A).
$$

But the nullspace is a subspace: if $v$ is allowed, then $-v$ is also allowed. Applying the condition to both signs forces

$$
\nabla f(x)^Tv=0
\qquad\text{for every }v\in\mathcal N(A).
$$

Therefore

$$
\nabla f(x)\in\mathcal N(A)^\perp.
$$

Using

$$
\mathcal N(A)^\perp=\mathcal R(A^T),
$$

we get

$$
\nabla f(x)=A^T\nu
$$

for some multiplier $\nu$.

Depending on the sign convention for the Lagrangian, this may be written

$$
\nabla f(x)+A^T\lambda=0.
$$

---

## 8. Eliminating equality constraints

Choose one feasible point $x_0$ satisfying

$$
Ax_0=b.
$$

Let the columns of $F$ form a basis for $\mathcal N(A)$. Every feasible point can be written as

$$
x=x_0+Fz.
$$

Then the constrained problem becomes the unconstrained problem

$$
\min_z f(x_0+Fz).
$$

This is useful both conceptually and computationally: $z$ represents exactly the directions that preserve feasibility.

---

## 9. Nonnegative orthant constraints

Consider

$$
\min f(x)
\quad\text{subject to}\quad
x\succeq0.
$$

The first-order condition requires

$$
\nabla f(x)^T(y-x)\ge0
$$

for every $y\succeq0$.

This implies

$$
x\succeq0,
\qquad
\nabla f(x)\succeq0,
\qquad
x^T\nabla f(x)=0.
$$

Componentwise,

$$
x_i\nabla_i f(x)=0.
$$

Thus, for each coordinate, either $x_i>0$ and the corresponding gradient component is zero, or the gradient component is positive and $x_i=0$. This is the basic geometry behind complementary slackness.

---

## 10. Block quadratic forms and the Schur complement

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

The intuition is that optimizing out $y$ leaves an effective quadratic curvature in $x$. This is the same elimination idea used for equality constraints, now applied to one block of a quadratic objective.

---

## What you should now understand

You should now be able to separate three ingredients that are often mixed together:

- $P\succeq0$ describes the curvature of a convex quadratic;
- the feasible set determines which directions can be used to reduce the objective;
- first-order optimality says that none of those feasible directions is a descent direction.

This explains why LPs can have optimal faces, why curved QP objectives can select interior points, why equality constraints make the gradient orthogonal to a nullspace, and why eliminating a quadratic block produces the Schur complement.

---

**Previous:** [Convex Sets, Functions, and Quasiconvex Optimization](/posts/convex-sets-functions-first-order-geometry/)

**Next:** [Conic Optimization: From LPs to SOCPs and SDPs](/posts/conic-optimization-socp-sdp/)

Return to the [Convex Optimization learning map](/posts/convex-optimization-doubt-log/).

## Sources and further study

- Stephen Boyd and Lieven Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
- Stephen Boyd, [EE364A lecture slides](https://web.stanford.edu/class/ee364a/lectures.html).
