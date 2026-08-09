---
title: "Quasiconvex Optimization and Bisection"
date: 2026-08-08 10:20:00 +0530
categories: [Mathematics, Optimization]
tags: [quasiconvexity, quasiconcavity, bisection, epigraph, linear-fractional, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/quasiconvex-optimization-bisection/
description: "Sublevel-set geometry, quasilinear examples, epigraphs, and the crucial distinction between x and the bisection level t."
---

Quasiconvexity is weaker than convexity, but it preserves exactly the geometry needed for level-set search. Instead of directly minimizing a curved objective, we repeatedly ask whether a proposed objective level is achievable.

That shift creates the most important doubt in this chapter: are we optimizing $x$ or $t$? The answer becomes clear once epigraph variables and fixed-level feasibility tests are separated carefully.

## 1. Epigraph form and piecewise-linear minimization {#epigraph-form}

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

## 2. Convex versus quasiconvex {#quasiconvexity}

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

## 3. Why is a linear-fractional objective quasiconvex?

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

## 4. Proof: the square root of an absolute value is quasiconvex

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

## 5. Proof: the ceiling function is quasilinear

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

## 6. Proof: the logarithm is quasilinear on the positive domain

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

## 7. Proof: the product of two positive variables is quasiconcave

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

## 8. Quasiconvex optimization by bisection: x versus t {#quasiconvex-bisection}

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

## 9. Ratio example in quasiconvex optimization

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

This article is part of the [Convex Optimization learning map](/posts/convex-optimization-doubt-log/).

## Sources and further study

- Stephen Boyd and Lieven Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
- Stephen Boyd, [EE364A lecture slides](https://web.stanford.edu/class/ee364a/lectures.html).
