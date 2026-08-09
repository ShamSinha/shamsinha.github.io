---
title: "Conic Optimization: From LPs to SOCPs and SDPs"
date: 2026-08-08 10:40:00 +0530
categories: [Mathematics, Optimization]
tags: [cones, generalized-inequalities, socp, semidefinite-programming, lmi, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/conic-optimization-socp-sdp/
description: "Why cones define useful notions of inequality, and how LP, SOCP, and SDP fit into one optimization framework."
---

Linear programs, second-order cone programs, and semidefinite programs can look like unrelated families. Conic optimization reveals that they are variations of one idea:

> minimize a linear objective while requiring an affine expression to belong to a convex cone.

The cone tells us what “nonnegative” means. For ordinary vectors it can mean componentwise nonnegative; for matrices it can mean positive semidefinite; for a scalar-vector pair it can mean that the scalar dominates the vector's Euclidean norm.

This viewpoint exists because many nonlinear-looking constraints become linear membership statements in the right geometric space.

## 1. What is a cone?

A set $K$ is a cone if scaling any member by a nonnegative number keeps it in the set:

$$
xin K, 	hetage 0
quadLongrightarrowquad
	heta xin K.
$$

A cone is a **convex cone** if it is also closed under addition:

$$
x,yin K
quadLongrightarrowquad
x+yin K.
$$

Equivalently, for all $	heta_1,	heta_2ge0$,

$$
	heta_1x+	heta_2yin K.
$$

The word *cone* is geometric, not merely visual. A cone may be the familiar ice-cream shape, but it may also be an orthant in $mathbb R^n$ or a set of matrices.

### Doubt: why is the scaling factor only nonnegative?

If negative scaling were required, then $xin K$ would force $-xin K$. Many important “positive” objects would no longer qualify. For example, a nonnegative vector becomes nonpositive after multiplication by $-1$.

A cone captures direction away from the origin. Nonnegative scaling moves along that direction without reversing it.

## 2. Three cones that organize common optimization problems

### The nonnegative orthant

$$
mathbb R_+^n
=
{xinmathbb R^nmid x_ige0, i=1,ldots,n}.
$$

Membership in this cone is simply componentwise nonnegativity.

### The second-order cone

The second-order cone, also called the Lorentz cone, is

$$
mathcal Q^{n+1}
=
{(t,x)inmathbb R	imesmathbb R^nmid lVert xVert_2le t}.
$$

Here $t$ must be large enough to dominate the Euclidean length of $x$. This cone packages a norm inequality into one membership condition.

### The positive-semidefinite cone

$$
mathbf S_+^n
=
{Xinmathbf S^nmid Xsucceq0},
$$

where $mathbf S^n$ is the vector space of real symmetric $n	imes n$ matrices. The condition

$$
Xsucceq0
$$

means

$$
z^TXzge0
qquad	ext{for every }zinmathbb R^n.
$$

This is a cone whose *points are matrices*. One PSD matrix $P$ is an element of $mathbf S_+^n$; it is not itself the whole cone.

For the linear-algebra foundation behind this definition, see [Convex Quadratics, PSD Matrices, and the Schur Complement](/posts/convex-quadratics-psd-schur-complement/) and [Symmetric Matrices, Quadratic Forms, and Matrix Norms](/posts/symmetric-matrices-quadratic-forms-matrix-norm/).

## 3. A cone creates a generalized inequality

The usual scalar order says

$$
xle y
quadLongleftrightarrowquad
y-xge0.
$$

A cone replaces the meaning of “$ge0$.” Given a cone $K$, define

$$
xpreceq_K y
quadLongleftrightarrowquad
y-xin K.
$$

### Componentwise order

If $K=mathbb R_+^n$, then

$$
xpreceq_K y
$$

means

$$
x_ile y_i
qquad	ext{for every }i.
$$

### Matrix order

If $K=mathbf S_+^n$, then

$$
Xpreceq Y
quadLongleftrightarrowquad
Y-Xsucceq0.
$$

This does **not** compare the entries one by one. It compares the quadratic action of the matrices:

$$
z^TXzle z^TYz
qquad	ext{for every }z.
$$

### Doubt: can every pair be compared?

No. Cone-induced order is usually a **partial order**, not a total order.

For example,

$$
x=(1,0),qquad y=(0,1)
$$

are incomparable under the componentwise order. Neither $y-x$ nor $x-y$ belongs to $mathbb R_+^2$.

That is not a defect. In multiple dimensions, one point can be larger in one direction and smaller in another. The cone records which differences count as positive without pretending every pair has a single ranking.

## 4. The common conic form

One standard conic optimization form is

$$
egin{aligned}
	ext{minimize}quad & c^Tx\
	ext{subject to}quad & Ax=b,\
& h-Gxin K.
end{aligned}
$$

Equivalently,

$$
Gxpreceq_K h.
$$

The objective and equality constraints are affine. The type of problem is determined by $K$:

| Cone $K$ | Resulting problem family |
|---|---|
| $mathbb R_+^m$ | linear program (LP) |
| product of second-order cones | second-order cone program (SOCP) |
| $mathbf S_+^m$ | semidefinite program (SDP) |

This unification matters both theoretically and computationally. Once a problem is expressed with a known cone, general conic duality and solver machinery become available.

## 5. LP: inequality through the nonnegative orthant

An LP has a linear objective and affine constraints, for example

$$
egin{aligned}
	ext{minimize}quad & c^Tx\
	ext{subject to}quad & Axle b.
end{aligned}
$$

The vector inequality means

$$
b-Axinmathbb R_+^m.
$$

So an LP is already a conic program; its cone is the nonnegative orthant.

### Doubt: what does “linear” mean here?

It describes the objective and constraint expressions, not the shape of the feasible region. Each scalar inequality creates a halfspace. Their intersection is a polyhedron, which can have edges, corners, or unbounded directions.

For the geometry of LP solutions, see [LP and QP Geometry, Constraints, and First-Order Optimality](/posts/optimality-constraint-geometry/).

## 6. SOCP: inequality through a norm cone

A typical second-order cone constraint is

$$
lVert A_ix+b_iVert_2
le
c_i^Tx+d_i.
$$

It is equivalent to

$$
(c_i^Tx+d_i, A_ix+b_i)inmathcal Q.
$$

The left side looks nonlinear because of the norm, but the pair inside the cone is affine in $x$.

SOCPs appear in robust optimization, portfolio models, control, signal processing, and estimation because Euclidean uncertainty and quadratic bounds often produce norm constraints.

### Doubt: is every quadratic constraint an SOCP constraint?

No. The quadratic structure must have the right convex form. A norm bound is convex, and several convex quadratic inequalities can be rewritten as second-order cone constraints after factoring a PSD matrix. An arbitrary indefinite quadratic constraint need not be convex and need not admit an SOCP representation.

## 7. SDP: inequality through the PSD cone

A semidefinite program commonly has the form

$$
egin{aligned}
	ext{minimize}quad & c^Tx\
	ext{subject to}quad &
F(x)=F_0+x_1F_1+cdots+x_nF_nsucceq0,
end{aligned}
$$

where every $F_i$ is symmetric.

The constraint $F(x)succeq0$ is called a **linear matrix inequality** (LMI). It is linear or affine in the decision variables $x_i$, even though testing it involves eigenvalues or quadratic forms.

### Doubt: are we optimizing a matrix or a vector?

In the form above, $xinmathbb R^n$ is the decision vector. The affine function $F(x)$ produces a matrix, and that matrix must lie in the PSD cone.

Some SDPs instead use a symmetric matrix $X$ directly as the decision variable. These are not conflicting definitions; they are two representations of the same framework.

### Why is the feasible set convex?

Suppose $F(x)succeq0$ and $F(y)succeq0$. For $	hetain[0,1]$,

$$
F(	heta x+(1-	heta)y)
=
	heta F(x)+(1-	heta)F(y).
$$

The PSD cone is convex, so this matrix is PSD. Therefore the set of feasible $x$ is convex.

## 8. Why do LMIs use symmetric matrices?

Positive semidefiniteness is defined through the quadratic form

$$
z^TFzge0.
$$

For any square matrix $F$, decompose it into symmetric and skew-symmetric parts:

$$
F=rac{F+F^T}{2}+rac{F-F^T}{2}.
$$

The skew-symmetric part contributes zero:

$$
z^Tleft(rac{F-F^T}{2}ight)z=0.
$$

Therefore

$$
z^TFz
=
z^Tleft(rac{F+F^T}{2}ight)z.
$$

Only the symmetric part affects the quadratic form. Working in $mathbf S^n$ removes an invisible, redundant part and gives real eigenvalues that can be ordered.

For complex matrices, the corresponding object is a **Hermitian** matrix $F=F^*$, and the definition uses $z^*Fzge0$.

## 9. How an LP fits inside an SDP

The scalar inequalities

$$
a_i^Txle b_i,qquad i=1,ldots,m
$$

can be placed on the diagonal of a matrix:

$$
operatorname{diag}
left(
b_1-a_1^Tx,ldots,b_m-a_m^Tx
ight)
succeq0.
$$

A diagonal matrix is PSD exactly when all of its diagonal entries are nonnegative. Thus every LP can be written as an SDP.

This does not mean we should solve every LP with a generic SDP solver. Specialized LP methods are usually more efficient. The representation shows an inclusion of expressive power:

$$
	ext{LP}subseteq	ext{SOCP}subseteq	ext{SDP},
$$

when the problem classes are compared through suitable conic representations.

### Doubt: why must every diagonal entry of a PSD matrix be nonnegative?

Let $e_i$ be the $i$th standard basis vector. If $Xsucceq0$, then

$$
e_i^TXe_ige0.
$$

But $e_i^TXe_i=X_{ii}$. Hence

$$
X_{ii}ge0
$$

for every diagonal entry.

The converse is false for a general symmetric matrix: nonnegative diagonal entries alone do not guarantee PSD. For example,

$$
egin{bmatrix}
1&2\
2&1
end{bmatrix}
$$

has positive diagonal entries but eigenvalues $3$ and $-1$.

## 10. The Schur complement turns inverse expressions into LMIs

Consider the symmetric block matrix

$$
M=
egin{bmatrix}
A&B\
B^T&C
end{bmatrix},
$$

with $Asucc0$. Then

$$
Msucceq0
quadLongleftrightarrowquad
C-B^TA^{-1}Bsucceq0.
$$

The matrix

$$
C-B^TA^{-1}B
$$

is the Schur complement of $A$ in $M$.

A particularly useful scalar version is

$$
tge x^TP^{-1}x,qquad Psucc0,
$$

which is equivalent to

$$
egin{bmatrix}
P&x\
x^T&t
end{bmatrix}
succeq0.
$$

The right side is an LMI in $(x,t)$. The Schur complement has converted an inverse-quadratic inequality into PSD-cone membership.

## 11. A second-order cone constraint as an LMI

The constraint

$$
lVert xVert_2le t
$$

can be represented as

$$
egin{bmatrix}
tI&x\
x^T&t
end{bmatrix}
succeq0.
$$

When $t>0$, take the Schur complement of $tI$:

$$
t-x^T(tI)^{-1}x
=
t-rac{lVert xVert_2^2}{t}
ge0.
$$

Multiplying by $t>0$ gives

$$
t^2gelVert xVert_2^2,
$$

hence $tgelVert xVert_2$. The boundary case $t=0$ forces $x=0$ and also satisfies both descriptions.

This demonstrates why an SOCP can be represented as an SDP. Again, the smaller cone-specific formulation is often computationally preferable.

## 12. Why SDP is useful beyond matrix-valued problems

SDPs are useful whenever a condition can be expressed as nonnegativity of a quadratic form or as a bound on eigenvalues. Examples include:

- finding Lyapunov functions for linear systems;
- bounding the largest eigenvalue of a symmetric matrix;
- experimental design and covariance estimation;
- relaxations of difficult combinatorial problems;
- control-system design through LMIs;
- sum-of-squares certificates for polynomial nonnegativity.

The matrix is not decoration. It stores infinitely many scalar inequalities compactly:

$$
z^TF(x)zge0
qquad	ext{for every }z.
$$

One PSD constraint therefore controls all directions $z$ simultaneously.

## 13. Connection to Perron–Frobenius theory

Cones also explain how “positivity” can extend beyond componentwise nonnegative vectors. Perron–Frobenius theory studies linear maps that preserve the nonnegative orthant and, more generally, maps that preserve a proper cone.

If

$$
A Ksubseteq K,
$$

then $A$ respects the order induced by $K$. This viewpoint connects positive dynamical systems, invariant cones, eigenvectors, and generalized inequalities.

The role of a cone is the same in both subjects: it selects which directions count as positive.

## 14. A compact mental checklist

When a new constraint looks unfamiliar, ask:

1. Can it be written as an affine expression belonging to a cone?
2. Is the cone the nonnegative orthant, a second-order cone, or the PSD cone?
3. What generalized inequality does that cone define?
4. Are two vector or matrix values actually comparable under that order?
5. If a matrix inequality appears, are its coefficient matrices symmetric?
6. Can a Schur complement remove an inverse or quadratic-over-linear term?
7. Is a specialized LP or SOCP form available before using the more general SDP form?

The central idea is simple: conic optimization moves nonlinearity out of the algebraic expression and into a well-understood geometric set.

---

This article is part of the [Convex Optimization learning map](/posts/convex-optimization-doubt-log/).

## Sources and further study

- Stephen Boyd and Lieven Vandenberghe, [*Convex Optimization*, Chapters 2 and 4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
- Stephen Boyd, [EE364A lecture slides](https://web.stanford.edu/class/ee364a/lectures.html).
