---
title: "LP and QP Geometry, Constraints, and First-Order Optimality"
date: 2026-08-08 10:30:00 +0530
categories: [Mathematics, Optimization]
tags: [linear-programming, quadratic-programming, optimality, equality-constraints, geometry, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/optimality-constraint-geometry/
description: "Where LP and QP optima occur, what constrained first-order optimality means, and how nullspaces eliminate equality constraints."
---

An unconstrained differentiable optimum often satisfies $\nabla f(x^\star)=0$. Under constraints, that rule is incomplete: feasible movement may exist only in certain directions. The correct question is whether any *feasible* direction can decrease the objective.

This chapter connects that principle to LP vertices, flat optimal faces, curved QP objectives, equality-constraint nullspaces, and nonnegative-orthant constraints.

## 1. LP optimum: must it always be a vertex? {#lp-qp-geometry}

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

## 2. Why can a quadratic-program solution be inside a face or in the interior?

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

## 3. First-order optimality over a convex feasible set {#constrained-optimality}

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

## 4. Equality constraints and the nullspace

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

## 5. Eliminating equality constraints

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

## 6. Nonnegative orthant constraints

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

This article is part of the [Convex Optimization learning map](/posts/convex-optimization-doubt-log/).

## Sources and further study

- Stephen Boyd and Lieven Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
- Stephen Boyd, [EE364A lecture slides](https://web.stanford.edu/class/ee364a/lectures.html).
