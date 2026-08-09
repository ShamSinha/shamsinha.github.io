---
title: "Convex Optimization: Introduction and Learning Map"
date: 2026-07-17 00:00:00 +0000
last_modified_at: 2026-08-09 00:00:00 +0530
categories: [Mathematics, Optimization]
tags: [convexity, convex-optimization, linear-programming, quadratic-programming, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/convex-optimization-doubt-log/
description: "A growing, foundations-first map of convex optimization definitions, intuition, proofs, examples, and doubts."
---

Convex optimization is too large for one endlessly growing article. This page is now the stable entry point: it introduces the language of optimization and links each topic to a focused chapter where its definitions and doubts stay together.

The notes currently follow the early part of Stephen Boyd's convex optimization course, through material being studied around Lecture 7. As the course continues, new topics can be added here without making any one chapter unmanageable.

## 1. What is an optimization problem?

A standard constrained optimization problem looks like

$$
\begin{aligned}
\text{minimize}\quad & f_0(x)\\
\text{subject to}\quad & f_i(x)\le0,\qquad i=1,\ldots,m,\\
& h_j(x)=0,\qquad j=1,\ldots,p.
\end{aligned}
$$

The pieces have different jobs:

- $x$ is the **decision variable**. It may be a scalar, vector, or matrix.
- $f_0(x)$ is the **objective function**. It returns the scalar quantity we want to minimize.
- The inequalities and equalities are the **constraints**.
- The values of $x$ satisfying every constraint form the **feasible set**.
- A feasible point with the smallest objective value is an **optimal point** $x^\star$.
- The number $p^\star=f_0(x^\star)$ is the **optimal value**.

This distinction prevents a recurring confusion: the point $x^\star$ and the value $p^\star$ are not the same object.

### A tiny example

Consider

$$
\begin{aligned}
\text{minimize}\quad & (x-3)^2\\
\text{subject to}\quad & x\le2.
\end{aligned}
$$

Without the constraint, the minimum occurs at $x=3$. That point is infeasible, so the constrained optimum is

$$
x^\star=2,
\qquad
p^\star=(2-3)^2=1.
$$

Optimization is therefore not only about the shape of the objective. It is about how that shape interacts with the feasible set.

## 2. What makes an optimization problem convex?

A convex optimization problem has:

1. a convex objective $f_0$;
2. convex inequality functions $f_i$;
3. affine equality functions $h_j$.

These conditions make the feasible set convex. More importantly, they give the problem a powerful local-to-global property: every local minimum is global.

Convexity does **not** mean every expression is linear. Quadratic functions, norms, exponential functions, and matrix inequalities can all appear in convex problems when their domains and curvature satisfy the right conditions.

### Why the assumptions matter

A slogan such as “a local minimum is global” is only useful with its assumptions attached. Remove convexity and a function may contain many local valleys. Reverse an inequality involving a convex function and the feasible set may become nonconvex. Use an indefinite quadratic and the supposed bowl can become a saddle.

The detailed chapters below keep those assumptions beside the doubts they answer.

## 3. Learning path

The chapters are ordered so that each one supplies the definitions needed by the next.

<span id="convex-sets"></span>
<span id="convex-functions"></span>
<span id="log-concavity"></span>
<span id="epigraph-form"></span>
<span id="quasiconvexity"></span>
<span id="quasiconvex-bisection"></span>

### 3.1 [Convex Sets, Functions, and Quasiconvex Optimization](/posts/convex-sets-functions-first-order-geometry/)

Start here. This combined chapter covers convex combinations, halfspaces, affine sets, polyhedra, Jensen's inequality, supporting hyperplanes, local versus global minima, log-concavity, epigraph formulations, quasiconvex and quasiconcave functions, and optimization by bisection.

Doubts answered include:

- What is an LP, and why is its feasible set a polyhedron?
- Why do intersections preserve convexity?
- Why does the gradient define a supporting hyperplane?
- Why does convexity make a local minimum global?
- Why is a covariance matrix PSD?
- How is quasiconvexity different from convexity?
- In quasiconvex bisection, are we optimizing $x$ or $t$?
- Why must the denominator of a linear-fractional objective be positive?

<span id="convex-quadratics"></span>
<span id="symmetric-quadratics"></span>

### 3.2 [Convex Quadratics, PSD Matrices, and the Schur Complement](/posts/convex-quadratics-psd-schur-complement/)

This is where linear algebra begins controlling curvature. The chapter explains convex quadratic objectives, positive-semidefinite matrices, symmetric and skew-symmetric parts, Hessians, block quadratic forms, and the Schur complement.

Doubts answered include:

- Does $P$ mean one PSD matrix or the entire PSD cone?
- Why is there a factor of $1/2$ in a quadratic objective?
- Why does the skew-symmetric part disappear from $x^TPx$?
- What goes wrong when the Hessian is not PSD?
- Why does minimizing over one block produce a Schur complement?

<span id="lp-qp-geometry"></span>
<span id="constrained-optimality"></span>

### 3.3 [LP and QP Geometry, Constraints, and First-Order Optimality](/posts/optimality-constraint-geometry/)

This chapter asks where optimal points can occur and how constraints change the usual gradient condition. It covers LP vertices and optimal faces, QP interior solutions, first-order optimality over a convex set, equality-constraint nullspaces, elimination of equalities, and the nonnegative orthant.

Doubts answered include:

- Must an LP have one unique optimal vertex?
- Can a QP optimum lie on a face or at a vertex?
- Why is $\nabla f(x^\star)=0$ insufficient under constraints?
- Why must the gradient be orthogonal to feasible equality-constrained directions?
- What changes when a variable is already at the boundary $x_i=0$?

### 3.4 [Conic Optimization: From LPs to SOCPs and SDPs](/posts/conic-optimization-socp-sdp/)

Conic optimization shows that LPs, second-order cone programs, and semidefinite programs share one structure. The chapter covers convex and proper cones, cone-induced partial orders, the nonnegative orthant, second-order and PSD cones, LMIs, the Schur complement, and exact conic representations.

Doubts answered include:

- How can a cone define “less than or equal to”?
- Why can two vectors be incomparable?
- Are the variables of an SDP vectors or matrices?
- Why must LMI coefficient matrices be symmetric?
- Why is every diagonal entry of a PSD matrix nonnegative?
- How can an LP or SOCP be represented as an SDP?

## 4. A compact map of problem families

| Problem family | Typical objective | Characteristic constraints | Main geometry |
|---|---|---|---|
| LP | affine | affine equalities and inequalities | polyhedra and the nonnegative orthant |
| Convex QP | convex quadratic | affine, sometimes convex quadratic | ellipsoidal level sets meeting a convex feasible set |
| Quasiconvex problem | quasiconvex | convex fixed-level feasibility tests | nested convex sublevel sets |
| SOCP | affine | affine expressions in second-order cones | norm cones |
| SDP | affine | affine matrix expressions in the PSD cone | positive-semidefinite matrix cone |

These families overlap. An LP is a conic program over the nonnegative orthant. Many convex quadratic constraints can be represented with second-order cones. LPs and SOCPs can also be embedded in SDPs, although the smallest suitable formulation is usually preferable computationally.

## 5. Doubts worth keeping

Good doubts expose missing assumptions. These recur across the chapters:

### “Convex means the Hessian is positive definite.”

Not always. For twice-differentiable functions, a PSD Hessian is enough. Convexity is also defined for nonsmooth functions, where a Hessian may not exist.

### “An LP solution is always one vertex.”

No. A linear objective can be constant across an entire edge or face. Under common conditions an optimal vertex exists, but other points may also be optimal.

### “Quasiconvex means positive curvature.”

No. Quasiconvexity means every sublevel set is convex. A quasiconvex function need not be convex.

### “A matrix with nonnegative diagonal entries is PSD.”

Not necessarily. PSD implies nonnegative diagonal entries, but the converse fails because off-diagonal coupling can create a negative eigenvalue.

### “A matrix inequality compares entries one by one.”

Only when the cone is the nonnegative orthant applied to a vector of entries. The PSD order $X\preceq Y$ instead means $Y-X\succeq0$, or

$$
z^TXz\le z^TYz
$$

for every direction $z$.

## 6. A practical checklist

When you meet a new optimization problem, ask:

1. What exactly is the decision variable?
2. What scalar quantity is being minimized or maximized?
3. What is the domain?
4. Which points are feasible?
5. Is the feasible set convex?
6. Is the objective convex, quasiconvex, or neither?
7. For a quadratic, is its symmetric Hessian PSD?
8. For a ratio, is the denominator known to be positive?
9. Does an auxiliary $t$ represent an epigraph variable or a fixed bisection level?
10. Can an unfamiliar constraint be expressed as membership in a standard cone?
11. Is a statement about an optimizer claiming existence, uniqueness, or both?

## 7. Where the map grows next

As later Boyd lectures are studied, the natural next chapters are:

- Lagrange duality, weak and strong duality;
- KKT conditions and complementary slackness;
- optimization applications and modeling patterns;
- numerical algorithms, Newton's method, and interior-point methods.

They will become separate topic pages and will be linked here. That keeps the introduction readable while allowing the detailed doubt log to grow without losing context.

## Sources and further study

- Stephen Boyd and Lieven Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
- Stephen Boyd, [EE364A lecture slides](https://web.stanford.edu/class/ee364a/lectures.html).
