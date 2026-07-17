---
title: "Convex Optimization: A Structured Guide"
date: 2026-07-17 12:00:00 +0000
categories: [Mathematics, Optimization]
tags: [convexity, quasiconvexity, quadratic-programming, linear-programming, positive-semidefinite]
math: true
toc: true
comments: true
description: "A rigorous, intuition-first guide to convex sets, quadratic forms, quasiconvexity, LPs, QPs, and optimality conditions."
---

These notes collect the main ideas needed to understand convex optimization geometrically and algebraically. The focus is not only on formulas, but on why the formulas have the form they do.

## 1. Convex sets

A set $C\subseteq\mathbb{R}^n$ is convex if, for every $x,y\in C$ and every $\theta\in[0,1]$,

$$
\theta x+(1-\theta)y\in C.
$$

Geometrically, the entire line segment joining any two points of the set remains inside the set.

### Halfspaces and hyperplanes

A halfspace has the form

$$
\{x\mid a^Tx\le b\}
$$

or

$$
\{x\mid a^Tx\ge b\}.
$$

Both are convex. The equality set

$$
\{x\mid a^Tx=b\}
$$

is a hyperplane and is also convex.

A polyhedron is an intersection of finitely many halfspaces and hyperplanes. Therefore, the feasible set of a linear program is a polyhedron.

## 2. Convex functions

A function $f$ is convex when its domain is convex and

$$
f(\theta x+(1-\theta)y)
\le
\theta f(x)+(1-\theta)f(y)
$$

for all $x,y$ in its domain and all $\theta\in[0,1]$.

The graph lies below every chord joining two graph points.

### Jensen's inequality

For nonnegative weights $\theta_i$ satisfying $\sum_i\theta_i=1$,

$$
f\left(\sum_i\theta_i x_i\right)
\le
\sum_i\theta_i f(x_i).
$$

For a random variable $X$,

$$
f(\mathbb{E}X)\le \mathbb{E}[f(X)].
$$

### First-order condition

If $f$ is differentiable, then $f$ is convex if and only if

$$
f(y)\ge f(x)+\nabla f(x)^T(y-x)
$$

for all $x,y$ in the domain.

This says that a convex function lies above every tangent hyperplane.

### Second-order condition

If $f$ is twice differentiable, then

$$
f\text{ convex}
\iff
\nabla^2 f(x)\succeq0
$$

for every $x$ in the domain.

The Hessian being positive semidefinite means that the second directional derivative is nonnegative in every direction.

## 3. Gradients and level sets

The first-order approximation is

$$
f(x+v)\approx f(x)+\nabla f(x)^Tv.
$$

Therefore:

- $\nabla f(x)^Tv>0$ means the function increases to first order.
- $\nabla f(x)^Tv<0$ means the function decreases to first order.
- $\nabla f(x)^Tv=0$ means no first-order change.

A level set is defined by

$$
f(x)=c.
$$

If a curve $x(t)$ remains on that level set, then

$$
\frac{d}{dt}f(x(t))=\nabla f(x(t))^Tx'(t)=0.
$$

Thus the gradient is perpendicular to every tangent direction of the level set. This is why the gradient becomes the normal vector of a tangent or supporting hyperplane.

## 4. Quadratic forms and convex quadratics

A quadratic function has the form

$$
f(x)=\frac12x^TPx+q^Tx+r.
$$

Assuming $P$ is symmetric,

$$
\nabla f(x)=Px+q,
\qquad
\nabla^2f(x)=P.
$$

Therefore,

$$
f\text{ is convex}
\iff
P\succeq0.
$$

The notation $P\in\mathbf{S}\_+^n$ means that $P$ is one symmetric positive semidefinite matrix. The symbol $\mathbf{S}\_+^n$ denotes the cone of all $n\times n$ symmetric positive semidefinite matrices.

A matrix is positive semidefinite when

$$
z^TPz\ge0
$$

for every $z$.

This means the quadratic has nonnegative curvature in every direction.

### Why the factor $1/2$ appears

The factor is only for convenience:

$$
\nabla\left(\frac12x^TPx\right)=Px
$$

when $P$ is symmetric.

## 5. Block quadratic forms and Schur complements

Consider

$$
f(x,y)=x^TAx+2x^TBy+y^TCy.
$$

This can be written as

$$
\begin{bmatrix}x\\y\end{bmatrix}^T
\begin{bmatrix}A&B\\B^T&C\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}.
$$

The Hessian is

$$
2\begin{bmatrix}A&B\\B^T&C\end{bmatrix}.
$$

Hence the function is jointly convex in $(x,y)$ exactly when the block matrix is positive semidefinite.

If $C\succ0$, minimizing over $y$ gives

$$
\nabla_yf=2B^Tx+2Cy=0,
$$

so

$$
y^\star=-C^{-1}B^Tx.
$$

Substituting this back gives

$$
\min_y f(x,y)
=x^T\left(A-BC^{-1}B^T\right)x.
$$

The matrix

$$
A-BC^{-1}B^T
$$

is the Schur complement of $C$.

## 6. Quasiconvex functions

A function $f$ is quasiconvex if every sublevel set

$$
\{x\mid f(x)\le\alpha\}
$$

is convex.

This is weaker than convexity. A quasiconvex function can have regions of positive and negative curvature. Curvature is not the defining property; the geometry of sublevel sets is.

In one dimension, a quasiconvex function has interval sublevel sets. Informally, it has no separated low valleys.

### Modified Jensen inequality

A function is quasiconvex if and only if

$$
f(\theta x+(1-\theta)y)
\le
\max\{f(x),f(y)\}
$$

for all $x,y$ and $\theta\in[0,1]$.

### First-order condition

For differentiable $f$ on a convex domain,

$$
f(y)\le f(x)
\implies
\nabla f(x)^T(y-x)\le0.
$$

The direction $y-x$ points from $x$ to a point with no larger function value. The gradient points toward increasing function values, so the angle between $\nabla f(x)$ and $y-x$ must be obtuse or right.

Unlike convex optimization, a quasiconvex problem can have a local minimum that is not global, especially when the function contains flat regions.

## 7. Quasiconvex optimization by bisection

Consider

$$
\min_x f_0(x)
$$

subject to convex constraints.

For a fixed scalar $t$, ask whether there exists an $x$ satisfying

$$
f_0(x)\le t
$$

and all original constraints.

Because $f_0$ is quasiconvex, its $t$-sublevel set is convex. Therefore each fixed-$t$ test is a convex feasibility problem.

Feasibility is monotone in $t$:

- if the problem is feasible at $t$, it remains feasible for every larger value;
- if it is infeasible at $t$, every smaller value is also infeasible.

This monotonicity makes bisection possible.

The scalar $t$ represents a candidate optimal objective value. The vector $x$ remains the actual decision variable.

### Ratio example

Suppose

$$
f_0(x)=\frac{p(x)}{q(x)},
$$

where $p$ is convex, $q$ is positive and concave, and $p\ge0$.

For fixed $t\ge0$,

$$
\frac{p(x)}{q(x)}\le t
\iff
p(x)-tq(x)\le0.
$$

Since $p$ is convex and $-tq$ is convex, the right-hand side is a convex inequality.

## 8. Quasilinear and log-concave examples

### Linear-fractional function

Consider

$$
f(x)=\frac{c^Tx+d}{e^Tx+f},
\qquad e^Tx+f>0.
$$

For fixed $t$,

$$
f(x)\le t
\iff
(c-te)^Tx+(d-tf)\le0.
$$

This is a halfspace. Therefore the function is quasiconvex. Its superlevel sets are also halfspaces, so it is quasilinear.

### Other examples

The function $\sqrt{\lvert x\rvert}$ is quasiconvex because

$$
\sqrt{|x|}\le\alpha
\iff
|x|\le\alpha^2,
$$

which gives an interval.

The function $\log x$ on $x>0$ is quasilinear because it is strictly increasing; both its sublevel and superlevel sets are intervals.

The function $x_1x_2$ on $\mathbb{R}_{++}^2$ is quasiconcave because

$$
x_1x_2\ge\alpha
\iff
\log x_1+\log x_2\ge\log\alpha,
$$

and $\log x_1+\log x_2$ is concave.

## 9. Log-concavity and Gaussian distributions

A positive function $f$ is log-concave when $\log f$ is concave.

For a multivariate Gaussian density,

$$
p(x)=C\exp\left(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right),
$$

we have

$$
\log p(x)
=-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)+\text{constant}.
$$

Its Hessian is

$$
-\Sigma^{-1},
$$

which is negative definite when $\Sigma\succ0$. Therefore the Gaussian density is log-concave.

The covariance matrix is always positive semidefinite because

$$
v^T\Sigma v
=\mathbb{E}\left[(v^T(X-\mu))^2\right]\ge0.
$$

## 10. Optimality in convex problems

For a convex differentiable objective over a convex feasible set $X$, a point $x^\star$ is optimal if and only if

$$
\nabla f(x^\star)^T(y-x^\star)\ge0
$$

for every feasible $y\in X$.

The vector $y-x^\star$ is the direction from the candidate solution toward another feasible point. If even one feasible direction gave a negative inner product, moving slightly in that direction would reduce the objective.

This condition also proves that every local minimum of a convex problem is global. If a better distant point existed, a small step toward it would already improve the objective, contradicting local optimality.

## 11. Equality constraints and nullspaces

For

$$
\min f(x)
\quad\text{subject to}\quad
Ax=b,
$$

both $x$ and any feasible comparison point $y$ satisfy the same equality. Therefore

$$
A(y-x)=0.
$$

Let

$$
v=y-x.
$$

Then $v\in\mathcal{N}(A)$, the nullspace of $A$.

The optimality condition becomes

$$
\nabla f(x)^Tv\ge0
$$

for every $v\in\mathcal{N}(A)$.

Because the nullspace is a subspace, if $v$ belongs to it, then $-v$ also belongs to it. The condition must therefore hold for both $v$ and $-v$, which forces

$$
\nabla f(x)^Tv=0
$$

for every $v\in\mathcal{N}(A)$.

Hence

$$
\nabla f(x)\in\mathcal{N}(A)^\perp.
$$

A fundamental linear-algebra identity is

$$
\mathcal{N}(A)^\perp=\mathcal{R}(A^T).
$$

Therefore

$$
\nabla f(x)=A^T\nu
$$

for some multiplier $\nu$.

### Eliminating equality constraints

Choose one particular solution $x_0$ such that

$$
Ax_0=b.
$$

Let the columns of $F$ form a basis for $\mathcal{N}(A)$. Every feasible point can then be written as

$$
x=x_0+Fz.
$$

Indeed,

$$
A(x_0+Fz)=Ax_0+AFz=b.
$$

This replaces a constrained variable $x$ by a free variable $z$.

## 12. Nonnegative orthant constraints

For

$$
\min f(x)
\quad\text{subject to}\quad
x\succeq0,
$$

optimality requires

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
x_i\,\nabla_i f(x)=0.
$$

Thus either $x_i>0$ and the corresponding gradient component is zero, or the gradient component is positive and $x_i=0$. This is the basic geometry behind complementary slackness.

## 13. Epigraph form and linear objectives

A convex problem

$$
\min_x f_0(x)
$$

can be rewritten as

$$
\min_{x,t} t
$$

subject to

$$
f_0(x)\le t.
$$

At the optimum, $t=f_0(x)$. This is why a linear objective is called universal for convex optimization: every convex objective can be moved into a convex constraint by adding one scalar variable.

There is nothing wrong with a nonlinear convex objective. The epigraph form is simply a useful standard representation.

## 14. Linear programming

A linear program has the form

$$
\min_x c^Tx+d
$$

subject to

$$
Gx\le h,
\qquad
Ax=b.
$$

Its feasible set is a polyhedron.

A common statement is that an LP optimum occurs at a vertex. The precise statement is:

> When an optimum is attained and the feasible polyhedron has extreme points, there exists an optimal extreme point.

The full optimal set can still be an edge or a higher-dimensional face. For example, minimizing $x_1$ over a square makes the entire left edge optimal, including nonvertex points.

## 15. Piecewise-linear minimization

A convex piecewise-linear function can be written as

$$
f(x)=\max_{i=1,\dots,m}(a_i^Tx+b_i).
$$

To minimize it, introduce $t$:

$$
\min_{x,t} t
$$

subject to

$$
a_i^Tx+b_i\le t,
\qquad i=1,\dots,m.
$$

This is a linear program.

The absolute value is the simplest example:

$$
|x|=\max\{x,-x\}.
$$

## 16. Quadratic programming

A convex quadratic program has the form

$$
\min_x \frac12x^TPx+q^Tx+r
$$

subject to affine constraints, with

$$
P\succeq0.
$$

Unlike an LP, the solution of a QP need not be at a vertex. It can occur:

- at a vertex;
- in the relative interior of an edge or face;
- in the interior of the feasible set.

For example,

$$
\min_{x\in[-1,1]^2} x_1^2+x_2^2
$$

has the solution $(0,0)$, which lies strictly inside the square.

The difference is curvature. An LP objective has flat level sets, while a strictly convex quadratic objective has curved level sets that can touch the feasible region away from vertices.

## 17. Final comparison

| Concept | Defining property | Geometric object |
|---|---|---|
| Convex function | Jensen inequality | Graph lies below chords |
| Quasiconvex function | Convex sublevel sets | No separated low regions |
| Concave function | Reverse Jensen inequality | Graph lies above chords |
| Log-concave function | $\log f$ is concave | Convex superlevel sets |
| Linear program | Linear objective and constraints | Polyhedral feasible set |
| Convex quadratic program | PSD quadratic objective | Curved convex level sets |

The central lesson is that convex optimization is fundamentally about geometry. Gradients describe local change, Hessians describe curvature, level sets describe global shape, and constraints determine which directions are feasible.
