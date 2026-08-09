---
title: "Convex Sets, Functions, and First-Order Geometry"
date: 2026-08-08 10:00:00 +0530
categories: [Mathematics, Optimization]
tags: [convexity, convex-sets, convex-functions, gradients, log-concavity, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/convex-sets-functions-first-order-geometry/
description: "A doubt-driven introduction to convex sets, convex functions, supporting hyperplanes, global minima, and log-concavity."
---

Convex optimization begins with two geometric questions: what makes a set convex, and what makes a function convex? This chapter develops those definitions before using them to answer the doubts that naturally follow.

You will see why LP feasible sets are polyhedra, how Jensen's inequality encodes the shape of a convex function, why gradients define supporting hyperplanes, and why local minima become global. The last section applies the same viewpoint to log-concavity and Gaussian distributions.

## 1. Convex sets, halfspaces, and polyhedra {#convex-sets}

A set $C\subseteq\mathbb R^n$ is convex when, for every $x,y\in C$ and every $\theta\in[0,1]$,

$$
\theta x+(1-\theta)y\in C.
$$

In plain language: draw the line segment between any two points in the set. The whole segment must stay inside the set.

### Why are halfspaces convex?

A halfspace has the form

$$
\{x\mid a^Tx\le b\}.
$$

Suppose

$$
a^Tx\le b,
\qquad
a^Ty\le b.
$$

For $z=\theta x+(1-\theta)y$,

$$
a^Tz
=\theta a^Tx+(1-\theta)a^Ty
\le \theta b+(1-\theta)b=b.
$$

Thus $z$ also belongs to the halfspace.

The equality set

$$
\{x\mid a^Tx=b\}
$$

is a hyperplane and is also convex.

### Why do intersections remain convex?

If every set $C_i$ contains the segment between any two of its points, then a pair of points belonging to every $C_i$ has its entire segment in every $C_i$. Hence the segment belongs to their intersection.

This explains why a finite system of linear constraints produces a convex feasible set.

### What is a linear program (LP)?

**LP** stands for **linear program** or **linear programming problem**. Its standard form is

$$
\begin{array}{ll}
\text{minimize} & c^Tx+d\\
\text{subject to} & Gx\le h,\\
& Ax=b.
\end{array}
$$

Here:

- $x\in\mathbb R^n$ is the decision vector we choose;
- $c^Tx+d$ is the scalar objective we want to minimize;
- $Gx\le h$ represents linear inequality constraints, interpreted componentwise;
- $Ax=b$ represents linear equality constraints.

The word **linear** means that the objective and constraints contain no products such as $x_1x_2$, powers such as $x_1^2$, or other nonlinear functions of $x$. Strictly speaking, expressions with constant terms are affine, but “linear program” is the standard name.

The **feasible set** depends only on the constraints, not on the objective:

$$
\mathcal F=\{x\mid Gx\le h,\ Ax=b\}.
$$

### Doubt: why is an LP feasible set a polyhedron?

Take one row of $G$:

$$
g_i^Tx\le h_i.
$$

This defines a halfspace. In two dimensions, it is one side of a line. In three dimensions, it is one side of a plane.

Take one row of $A$:

$$
a_j^Tx=b_j.
$$

This defines a hyperplane.

A polyhedron is exactly an intersection of finitely many halfspaces and hyperplanes. Therefore, an LP feasible set is a polyhedron.

### Polyhedron versus polytope

A polyhedron can be unbounded.

A polytope is a bounded polyhedron.

For example,

$$
x_1\ge0,\qquad x_2\ge0,\qquad x_1+x_2\le1
$$

forms a triangle, hence a bounded polyhedron and therefore a polytope.

---

## 2. Convex functions and Jensen's inequality {#convex-functions}

A function $f$ is convex if

$$
f(\theta x+(1-\theta)y)
\le
\theta f(x)+(1-\theta)f(y)
$$

for all $x,y$ in its domain and all $\theta\in[0,1]$.

The graph lies below the chord joining two graph points.

### Jensen's inequality

For nonnegative weights $\theta_i$ satisfying $\sum_i\theta_i=1$,

$$
f\left(\sum_i\theta_i x_i\right)
\le
\sum_i\theta_i f(x_i).
$$

For a random variable $X$,

$$
f(\mathbb E[X])\le \mathbb E[f(X)].
$$

The two-point definition is the basic version; Jensen is the many-point or probabilistic version.

---

## 3. Why does the gradient define a tangent or supporting hyperplane?

For a differentiable function,

$$
f(x+v)\approx f(x)+\nabla f(x)^Tv.
$$

The inner product tells us the first-order change:

- $\nabla f(x)^Tv>0$: increase;
- $\nabla f(x)^Tv<0$: decrease;
- $\nabla f(x)^Tv=0$: no first-order change.

Suppose $x(t)$ moves along a level set

$$
f(x(t))=c.
$$

Differentiate:

$$
\frac{d}{dt}f(x(t))
=\nabla f(x(t))^Tx'(t)=0.
$$

The tangent direction $x'(t)$ is orthogonal to the gradient. Therefore the gradient is normal to the level set.

For a differentiable convex function,

$$
f(y)\ge f(x)+\nabla f(x)^T(y-x).
$$

So the tangent hyperplane lies below the graph. That is why it is called a supporting hyperplane.

---

## 4. Why does convexity make every local minimum global?

Suppose $x$ is a local minimum but there exists a feasible $y$ with

$$
f(y)<f(x).
$$

Take a point close to $x$ on the segment toward $y$:

$$
z_\theta=(1-\theta)x+\theta y,
\qquad \theta>0\text{ small}.
$$

Convexity gives

$$
f(z_\theta)
\le(1-\theta)f(x)+\theta f(y)
<f(x).
$$

But $z_\theta$ can be made arbitrarily close to $x$, contradicting local optimality.

This is one of the main reasons convex optimization is special.

---

## 5. Log-concavity and the Gaussian example {#log-concavity}

A positive function $p$ is log-concave when

$$
\log p(x)
$$

is concave.

For a multivariate Gaussian density,

$$
p(x)=C\exp\left(
-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)
\right),
$$

we have

$$
\log p(x)
=
-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)
+\text{constant}.
$$

Its Hessian is

$$
-\Sigma^{-1}.
$$

If $\Sigma\succ0$, then $\Sigma^{-1}\succ0$, so

$$
-\Sigma^{-1}\prec0.
$$

Thus $\log p$ is concave and the Gaussian density is log-concave.

### Why is a covariance matrix PSD?

For any vector $v$,

$$
v^T\Sigma v
=
\mathbb E\left[(v^T(X-\mu))^2\right]
\ge0.
$$

A variance cannot be negative.

---

This article is part of the [Convex Optimization learning map](/posts/convex-optimization-doubt-log/).

## Sources and further study

- Stephen Boyd and Lieven Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
- Stephen Boyd, [EE364A lecture slides](https://web.stanford.edu/class/ee364a/lectures.html).
