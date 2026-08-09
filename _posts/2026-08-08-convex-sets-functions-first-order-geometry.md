---
title: "Convex Sets, Functions, and Quasiconvex Optimization"
date: 2026-08-08 10:00:00 +0530
last_modified_at: 2026-08-09 00:00:00 +0530
categories: [Mathematics, Optimization]
tags: [convexity, convex-sets, convex-functions, quasiconvexity, bisection, log-concavity, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/convex-sets-functions-first-order-geometry/
description: "A doubt-driven guide to convex sets, convex functions, first-order geometry, log-concavity, quasiconvexity, and bisection."
---

Convex optimization begins with two geometric questions: what makes a set convex, and what makes a function convex? This chapter develops those definitions before using them to answer the doubts that naturally follow.

## Why this chapter exists

Before studying particular optimization models, we need a reliable way to recognize convex sets and convex functions. These definitions explain why local information can solve a global problem and why a quasiconvex objective can be handled through convex feasibility tests even when it is not itself convex.

## Before you start

Only basic vector and matrix notation is assumed. If terms such as affine map, inner product, matrix transpose, or covariance are unfamiliar, keep [Linear Algebra: Important Concepts and the Doubts That Connect Them](/posts/linear-algebra-important-concepts/) nearby as a reference.

## What you will learn

You will see why LP feasible sets are polyhedra, how Jensen's inequality encodes the shape of a convex function, why gradients define supporting hyperplanes, and why local minima become global. The same set-based viewpoint then leads to log-concavity, quasiconvexity, epigraph formulations, and optimization by bisection.

The chapter moves from convex sets, to convex functions, to first-order geometry, and finally to quasiconvex optimization. Each later section uses the definitions established before it.

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

### Why is a covariance matrix PSD? {#why-is-a-covariance-matrix-psd}

Let $X\in\mathbb R^n$ be a random vector with mean

$$
\mu=\mathbb E[X].
$$

Its covariance matrix is

$$
\Sigma
=
\mathbb E\left[(X-\mu)(X-\mu)^T\right].
$$

For every realization of $X$, the outer-product matrix

$$
(X-\mu)(X-\mu)^T
$$

is symmetric. Taking its expectation preserves symmetry, so $\Sigma=\Sigma^T$.

The diagonal entry $\Sigma_{ii}$ is the variance of $X_i$, while an off-diagonal entry $\Sigma_{ij}$ measures how $X_i$ and $X_j$ vary together. But checking the entries individually is not enough to prove that the whole matrix is PSD. We must prove

$$
v^T\Sigma v\ge0
$$

for **every** direction $v\in\mathbb R^n$.

Take an arbitrary $v$. Substituting the definition of $\Sigma$ gives

$$
\begin{aligned}
v^T\Sigma v
&=
v^T\mathbb E\left[(X-\mu)(X-\mu)^T\right]v\\
&=
\mathbb E\left[
v^T(X-\mu)(X-\mu)^Tv
\right]\\
&=
\mathbb E\left[
\left(v^T(X-\mu)\right)^2
\right].
\end{aligned}
$$

The last step works because $(X-\mu)^Tv$ is the same scalar as $v^T(X-\mu)$.

Now define the scalar random variable

$$
Y=v^TX.
$$

Its mean is

$$
\mathbb E[Y]
=
v^T\mathbb E[X]
=
v^T\mu.
$$

Therefore,

$$
v^T\Sigma v
=
\mathbb E\left[
\left(Y-\mathbb E[Y]\right)^2
\right]
=
\operatorname{Var}(Y)
=
\operatorname{Var}(v^TX).
$$

This gives the geometric meaning of the quadratic form:

> $v^T\Sigma v$ is the variance of the data after projecting it onto the direction $v$.

For every outcome, the squared quantity

$$
\left(Y-\mathbb E[Y]\right)^2
$$

is nonnegative. Its expectation therefore cannot be negative:

$$
v^T\Sigma v
=
\operatorname{Var}(v^TX)
\ge0.
$$

Since the inequality holds for every $v$,

$$
\boxed{\Sigma\succeq0}.
$$

### Why is a covariance matrix not always positive definite?

Positive definite would require

$$
v^T\Sigma v>0
$$

for every nonzero $v$. This fails when the data has no variation along some direction.

For example, suppose $Z$ has mean zero and variance $\sigma^2$, and let

$$
X=
\begin{bmatrix}
Z\\
2Z
\end{bmatrix}.
$$

Then

$$
\Sigma
=
\sigma^2
\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}.
$$

Choose

$$
v=
\begin{bmatrix}
2\\
-1
\end{bmatrix}.
$$

The projection in this direction is

$$
v^TX=2Z-2Z=0.
$$

Hence

$$
v^T\Sigma v
=
\operatorname{Var}(v^TX)
=0
$$

even though $v\ne0$. The covariance matrix is PSD but not positive definite. Geometrically, all observations lie on the line $x_2=2x_1$, so there is no spread in the perpendicular direction.

This also explains the assumption $\Sigma\succ0$ in the Gaussian formula above. A full-dimensional Gaussian density needs $\Sigma^{-1}$. If $\Sigma$ is singular, the Gaussian distribution is confined to a lower-dimensional subspace and does not have the usual density over all of $\mathbb R^n$.

---

Convex functions are not the end of the story. Some nonconvex functions still have convex sublevel sets, which is enough to solve their optimization problems through a sequence of convex feasibility tests. The following sections develop that idea without separating its definitions from the doubts about $x$, $t$, and bisection.

## 6. Epigraph form and piecewise-linear minimization {#epigraph-form}

Before the piecewise-linear example, it helps to understand a general representation used throughout convex optimization.

For a convex objective, the problem

$$
\min_x f_0(x)
$$

is equivalent to its **epigraph form**:

$$
\begin{array}{ll}
\text{minimize} & t\\
\text{subject to} & f_0(x)\le t.
\end{array}
$$

At an optimum, $t=f_0(x)$; otherwise $t$ could be reduced. The objective is now linear, while the original objective appears as a convex constraint.

This does **not** mean that convex optimization requires a linear objective. Epigraph form is simply a useful equivalent representation.

It is also different from quasiconvex bisection later in this guide:

- for a convex $f_0$, the entire epigraph $\{(x,t)\mid f_0(x)\le t\}$ is convex, so $x$ and $t$ can be optimized jointly;
- for a merely quasiconvex $f_0$, only each fixed-$t$ sublevel set is guaranteed to be convex, so bisection handles $t$ externally.

### Piecewise-linear objectives

A standard convex piecewise-linear function is

$$
f(x)=\max_{i=1,\ldots,m}(a_i^Tx+b_i).
$$

Different affine pieces dominate in different regions. Their maximum forms an upper envelope with flat pieces and corners.

### Why is it convex?

Each affine function is convex, and the pointwise maximum of convex functions is convex.

### How do we minimize it?

Start from

$$
\min_x\max_i(a_i^Tx+b_i).
$$

Introduce a scalar $t$:

$$
\begin{array}{ll}
\text{minimize} & t\\
\text{subject to} & a_i^Tx+b_i\le t,\quad i=1,\ldots,m.
\end{array}
$$

The constraints say that $t$ must lie above every affine piece. Minimizing $t$ pushes it down until it touches the upper envelope.

If the remaining constraints are linear, this is a linear program.

### Small example

$$
\min_x\max(x+1,-x+2).
$$

Equivalent LP:

$$
\min_{x,t}t
$$

subject to

$$
x+1\le t,
\qquad
-x+2\le t.
$$

At the optimum, the two active pieces meet:

$$
x+1=-x+2,
$$

so

$$
x=\frac12,
\qquad t=\frac32.
$$

---

## 7. Convex versus quasiconvex {#quasiconvexity}

A function is quasiconvex when every sublevel set

$$
S_\alpha=\{x\mid f(x)\le\alpha\}
$$

is convex.

This is weaker than ordinary convexity.

A quasiconvex function need not lie below its chords. Instead, it satisfies

$$
f(\theta x+(1-\theta)y)
\le\max\{f(x),f(y)\}.
$$

The important geometric idea is that low-value regions do not split into separated pieces.

### Convex implies quasiconvex

If $f$ is convex, then

$$
f(\theta x+(1-\theta)y)
\le\theta f(x)+(1-\theta)f(y)
\le\max\{f(x),f(y)\}.
$$

The reverse is false.

### First-order condition for a differentiable quasiconvex function

For differentiable $f$ on a convex domain, quasiconvexity is equivalent to

$$
f(y)\le f(x)
\implies
\nabla f(x)^T(y-x)\le0.
$$

Why? If $f(y)\le f(x)$, then $y$ belongs to the sublevel set

$$
\{z\mid f(z)\le f(x)\}.
$$

The vector $y-x$ points from $x$ into that convex low-value region, while $\nabla f(x)$ points toward increasing values. Their inner product therefore cannot be positive.

Compare this with the stronger convex condition

$$
f(y)\ge f(x)+\nabla f(x)^T(y-x),
$$

which supplies a global affine lower bound. Quasiconvexity gives only the directional implication above.

---

## 8. Why is a linear-fractional objective quasiconvex?

Consider

$$
f(x)=\frac{c^Tx+d}{e^Tx+f},
\qquad e^Tx+f>0.
$$

To prove quasiconvexity, examine a sublevel set:

$$
\frac{c^Tx+d}{e^Tx+f}\le t.
$$

Because the denominator is positive, multiplication does not reverse the inequality:

$$
c^Tx+d\le t(e^Tx+f).
$$

Rearrange:

$$
(c-te)^Tx+(d-tf)\le0.
$$

For fixed $t$, this is an affine inequality in $x$, hence a halfspace.

The domain condition

$$
e^Tx+f>0
$$

is also a halfspace. Their intersection is convex. Thus every sublevel set is convex, so the function is quasiconvex.

### Why is it actually quasilinear?

A function is quasiconcave when all superlevel sets are convex.

Here,

$$
\frac{c^Tx+d}{e^Tx+f}\ge t
$$

becomes

$$
(c-te)^Tx+(d-tf)\ge0,
$$

which is another halfspace. Therefore the function is both quasiconvex and quasiconcave: it is quasilinear.

### Why is positivity of the denominator crucial?

If the denominator could be negative, multiplying by it might reverse the inequality. If it could be zero, the ratio would not even be defined. The domain condition is essential, not decorative.

---

## 9. Proof: the square root of an absolute value is quasiconvex

Consider

$$
f(x)=\sqrt{|x|}.
$$

For $\alpha<0$, the sublevel set is empty, which is convex.

For $\alpha\ge0$,

$$
\sqrt{|x|}\le\alpha
\iff |x|\le\alpha^2
\iff -\alpha^2\le x\le\alpha^2.
$$

Thus every sublevel set is an interval, hence convex.

The function is quasiconvex even though it is not convex on all of $\mathbb R$.

---

## 10. Proof: the ceiling function is quasilinear

The ceiling function is monotone nondecreasing.

For a real number $\alpha$, let $m=\lfloor\alpha\rfloor$. Then

$$
\{x\mid\lceil x\rceil\le\alpha\}
=\{x\mid x\le m\}
=(-\infty,m],
$$

which is convex.

For the superlevel set, let $k=\lceil\alpha\rceil$. Then

$$
\{x\mid\lceil x\rceil\ge\alpha\}
=\{x\mid x>k-1\}
=(k-1,\infty),
$$

also convex.

Therefore $\lceil x\rceil$ is both quasiconvex and quasiconcave.

A discontinuous function can be quasilinear; quasilinear does not mean affine.

---

## 11. Proof: the logarithm is quasilinear on the positive domain

For a sublevel set,

$$
\log x\le\alpha
\iff x\le e^\alpha.
$$

Including the domain $x>0$ gives

$$
(0,e^\alpha],
$$

an interval.

For a superlevel set,

$$
\log x\ge\alpha
\iff x\ge e^\alpha,
$$

so the set is

$$
[e^\alpha,\infty).
$$

Both are convex. Therefore $\log x$ is quasilinear on $\mathbb R_{++}$.

It is also concave, but it is not affine.

---

## 12. Proof: the product of two positive variables is quasiconcave

We inspect superlevel sets:

$$
\{(x_1,x_2)\mid x_1x_2\ge\alpha,\ x_1>0,\ x_2>0\}.
$$

If $\alpha\le0$, the entire positive orthant satisfies the inequality.

If $\alpha>0$, take logarithms:

$$
x_1x_2\ge\alpha
\iff
\log x_1+\log x_2\ge\log\alpha.
$$

The function

$$
g(x_1,x_2)=\log x_1+\log x_2
$$

is concave. A superlevel set of a concave function is convex. Therefore $x_1x_2$ is quasiconcave on the positive orthant.

Another geometric form is

$$
x_2\ge\frac{\alpha}{x_1},
$$

which is the region above the convex curve $\alpha/x_1$.

---

## 13. Quasiconvex optimization by bisection: x versus t {#quasiconvex-bisection}

Consider

$$
\begin{array}{ll}
\text{minimize} & f_0(x)\\
\text{subject to} & f_i(x)\le0,\quad i=1,\ldots,m,\\
& Ax=b,
\end{array}
$$

where $f_0$ is quasiconvex and the remaining constraints define a convex feasible set.

Here, $x$ is the original **decision variable**. It can be a scalar, but it is usually a vector in $\mathbb R^n$. The function $f_0$ maps each $x$ to a scalar objective value.

The optimal objective value is

$$
p^\star
=
\inf\{f_0(x)\mid f_i(x)\le0,\ Ax=b\}.
$$

If this value is attained, an optimal point is denoted by $x^\star$, and

$$
f_0(x^\star)=p^\star.
$$

### Why introduce t?

The original problem contains no $t$. We introduce it only as a **trial objective value**.

For a fixed scalar $t$, solve the feasibility problem

$$
\begin{array}{ll}
\text{find} & x\\
\text{subject to} & f_0(x)\le t,\\
& f_i(x)\le0,\quad i=1,\ldots,m,\\
& Ax=b.
\end{array}
$$

This asks one yes-or-no question:

$$
\text{Does there exist an }x\text{ satisfying every constraint and }f_0(x)\le t?
$$

Because $f_0$ is quasiconvex, the sublevel set

$$
\{x\mid f_0(x)\le t\}
$$

is convex for every fixed $t$. Intersecting it with the original convex constraints leaves a convex feasibility problem.

This is the context in which the following doubt arose:

> **Doubt:** Here we want the optimal value of $x$ or $t$? $t$ is scalar, while $x$ may or may not be scalar.

The phrase “optimal value of $x$” mixes two different objects:

- $x^\star$ is an **optimal point**. It has the same dimension as $x$.
- $p^\star$ is the **optimal value** of the objective. It is always scalar.
- $t$ is a temporary scalar guess for $p^\star$.

For each guess, the feasibility solver searches for an $x$, while $t$ remains fixed. It is not minimizing $x$; it only needs to find one witness satisfying all the constraints.

### Why bisection works

Feasibility is monotone in $t$:

- feasible at $t$ implies feasible at every larger value;
- infeasible at $t$ implies infeasible at every smaller value.

Maintain a lower bound $l$ known to be infeasible and an upper bound $u$ known to be feasible.

Set

$$
t=\frac{l+u}{2}.
$$

- If feasible, store the witness $x$ and set $u=t$.
- If infeasible, set $l=t$.

Stop when $u-l$ is small.

Then:

- the shrinking scalar interval approximates $p^\star$;
- the best stored feasible witness approximates an optimizer $x^\star$.

There may be several optimal points $x^\star$, but the optimal objective value $p^\star$ is a single scalar. It is also possible for the infimum to exist without being attained, in which case no exact optimizer exists.

### A tiny numerical example

Consider

$$
\begin{array}{ll}
\text{minimize} & x^2\\
\text{subject to} & x\ge1.
\end{array}
$$

The answer is

$$
x^\star=1,
\qquad
p^\star=1.
$$

Pretend we do not yet know it.

- Try $t=4$. An $x\ge1$ with $x^2\le4$ exists; for example, $x=1.5$. The test is feasible, so $p^\star\le4$.
- Try $t=0.5$. No $x\ge1$ can satisfy $x^2\le0.5$. The test is infeasible, so $p^\star>0.5$.

Bisection keeps updating $t$ to locate the smallest feasible threshold. Successful tests also provide candidate points $x$.

### Why not optimize x and t jointly?

For a general quasiconvex function, the epigraph

$$
\{(x,t)\mid f_0(x)\le t\}
$$

need not be convex. Quasiconvexity guarantees convexity of the sublevel set in $x$ only after $t$ has been fixed. That is why bisection handles $t$ externally.

So the final mental model is:

- bisection searches over $t$ to approximate $p^\star$;
- each feasibility test searches for an $x$;
- the result includes both an objective value and, when it is attained, an optimizer.

---

## 14. Ratio example in quasiconvex optimization

Suppose

$$
f_0(x)=\frac{p(x)}{q(x)},
$$

where

- $p$ is convex;
- $q$ is concave and positive;
- $p(x)\ge0$.

For fixed $t\ge0$,

$$
\frac{p(x)}{q(x)}\le t
\iff
p(x)-tq(x)\le0.
$$

Since $q$ is concave, $-q$ is convex. For a fixed nonnegative $t$, $-tq$ is convex. Hence

$$
\phi_t(x)=p(x)-tq(x)
$$

is convex in $x$.

The word **fixed** matters. If $t$ were simultaneously a variable, the product $tq(x)$ would generally destroy convexity.

Also notice the inequality is

$$
p(x)-tq(x)\le0,
$$

not an equality.

---

## What you should now understand

You should now be able to test whether a set is convex, recognize why affine constraints produce polyhedra, and use definitions, gradients, or Hessians to reason about convex functions. You should also be able to distinguish convexity from quasiconvexity and explain why quasiconvex bisection searches over a scalar level while each subproblem searches for a feasible point.

---

**Next:** [Quadratic and Constrained Convex Optimization: Curvature, Geometry, and Optimality](/posts/convex-quadratics-psd-schur-complement/)

Return to the [Convex Optimization learning map](/posts/convex-optimization-doubt-log/).

## Sources and further study

- Stephen Boyd and Lieven Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
- Stephen Boyd, [EE364A lecture slides](https://web.stanford.edu/class/ee364a/lectures.html).
