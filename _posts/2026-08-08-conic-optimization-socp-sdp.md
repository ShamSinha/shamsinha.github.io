---
title: "Conic Optimization: From LPs to SOCPs and SDPs"
date: 2026-08-08 10:40:00 +0530
last_modified_at: 2026-08-09 00:00:00 +0530
categories: [Mathematics, Optimization]
tags: [cones, generalized-inequalities, socp, semidefinite-programming, lmi, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/conic-optimization-socp-sdp/
description: "Why cones define useful notions of inequality, and how LP, SOCP, and SDP fit into one optimization framework."
---

Linear programs, second-order cone programs, and semidefinite programs can look like unrelated families. Conic optimization reveals that they are versions of one idea:

> Minimize a linear objective while requiring affine expressions to belong to convex cones.

The cone determines what “nonnegative” means. For a vector, it can mean componentwise nonnegative. For a scalar-vector pair, it can mean that the scalar dominates a Euclidean norm. For a symmetric matrix, it can mean positive semidefinite.

This viewpoint is useful because many nonlinear-looking constraints become affine membership statements once we choose the right cone.

## 1. Cone and convex cone

A nonempty set $K$ is a **cone** if

$$
x\in K,\quad \alpha\ge 0
\quad\Longrightarrow\quad
\alpha x\in K.
$$

Thus, once a direction belongs to the cone, every nonnegative scaling of that direction also belongs to it.

A cone is a **convex cone** if it is closed under nonnegative linear combinations:

$$
x,y\in K,\quad \alpha,\beta\ge 0
\quad\Longrightarrow\quad
\alpha x+\beta y\in K.
$$

This single condition includes both nonnegative scaling and addition.

### Doubt: why are negative scaling factors excluded?

A cone is meant to represent a collection of positive directions. If $x\in K$ automatically forced $-x\in K$, then the cone could not distinguish a direction from its opposite.

For example, a cone used to represent nonnegative quantities must contain a positive direction without being forced to contain the corresponding negative direction.

## 2. Three important convex cones

### The nonnegative orthant

The nonnegative orthant is

$$
\mathbb R_+^n
=
\left\{
x\in\mathbb R^n
\;\middle|\;
x_i\ge 0,\ i=1,\ldots,n
\right\}.
$$

Membership in this cone means componentwise nonnegativity.

### The second-order cone

The second-order cone, also called the Lorentz cone, is

$$
\mathcal Q^{m+1}
=
\left\{
(t,u)\in\mathbb R\times\mathbb R^m
\;\middle|\;
\lVert u\rVert_2\le t
\right\}.
$$

Membership packages a norm inequality into one geometric statement. The scalar $t$ must dominate the Euclidean norm of $u$.

### The positive-semidefinite cone

Let $\mathbf S^n$ denote the vector space of real symmetric $n\times n$ matrices. The positive-semidefinite cone is

$$
\mathbf S_+^n
=
\left\{
X\in\mathbf S^n
\;\middle|\;
X\succeq 0
\right\}.
$$

The condition $X\succeq0$ means

$$
z^TXz\ge 0
\qquad
\text{for every }z\in\mathbb R^n.
$$

The members of this cone are matrices. One matrix $P\succeq0$ is an element of $\mathbf S_+^n$; it is not the entire cone.

The PSD cone is convex. If $X,Y\succeq0$ and $\alpha,\beta\ge0$, then for every $z$,

$$
z^T(\alpha X+\beta Y)z
=
\alpha z^TXz+\beta z^TYz
\ge0.
$$

Therefore,

$$
\alpha X+\beta Y\succeq0.
$$

For more linear-algebra background, see [Convex Quadratics, PSD Matrices, and the Schur Complement](/posts/convex-quadratics-psd-schur-complement/) and [Symmetric Matrices, Quadratic Forms, and Matrix Norms](/posts/symmetric-matrices-quadratic-forms-matrix-norm/).

## 3. Proper cones and generalized inequalities

To define a well-behaved generalized inequality, we normally use a **proper cone**. A proper cone $K$ is:

- convex;
- closed;
- pointed, meaning $K\cap(-K)=\{0\}$;
- solid, meaning it has nonempty interior.

The cone defines the relation

$$
x\preceq_K y
\quad\Longleftrightarrow\quad
y-x\in K.
$$

It also defines the strict relation

$$
x\prec_K y
\quad\Longleftrightarrow\quad
y-x\in\operatorname{int}K.
$$

The ordinary scalar inequality is recovered by choosing $K=\mathbb R_+$:

$$
x\le y
\quad\Longleftrightarrow\quad
y-x\in\mathbb R_+.
$$

### Why does pointedness matter?

A partial order must be antisymmetric:

$$
x\preceq_K y
\quad\text{and}\quad
y\preceq_K x
\quad\Longrightarrow\quad
x=y.
$$

The two inequalities imply

$$
y-x\in K
\quad\text{and}\quad
-(y-x)\in K.
$$

Thus $y-x\in K\cap(-K)$. If $K$ is pointed, the intersection contains only $0$, so $y-x=0$ and therefore $x=y$.

Without pointedness, the cone can contain a nonzero direction and its negative, and antisymmetry can fail. Calling the relation a partial order without this condition would be imprecise.

### Componentwise order

For $K=\mathbb R_+^n$,

$$
x\preceq_K y
$$

means

$$
x_i\le y_i
\qquad
\text{for every }i.
$$

### Matrix order

For $K=\mathbf S_+^n$,

$$
X\preceq Y
\quad\Longleftrightarrow\quad
Y-X\succeq0.
$$

This is not an entrywise comparison. It means

$$
z^TXz\le z^TYz
\qquad
\text{for every }z\in\mathbb R^n.
$$

### Doubt: must every pair be comparable?

No. A partial order need not be a total order.

Under componentwise order, one vector can be larger in one coordinate and smaller in another. Neither difference then belongs to the nonnegative orthant. The vectors are incomparable.

The same phenomenon occurs with the PSD order: $X-Y$ can have both positive and negative curvature directions, so neither $X\preceq Y$ nor $Y\preceq X$ holds.

## 4. The common conic form

A conic optimization problem can be written as

$$
\begin{aligned}
\text{minimize}\quad & c^Tx\\
\text{subject to}\quad & Ax=b,\\
& h-Gx\in K.
\end{aligned}
$$

Using the order induced by $K$, the cone constraint is equivalently

$$
Gx\preceq_K h.
$$

The objective and equality constraints are affine. The choice of cone determines the problem family:

| Cone | Problem family |
|---|---|
| $\mathbb R_+^m$ | linear program (LP) |
| a product of second-order cones | second-order cone program (SOCP) |
| $\mathbf S_+^m$ | semidefinite program (SDP) |

The key modeling step is not to force every problem into the same notation. It is to recognize which cone describes the constraint naturally.

## 5. Linear programming as conic optimization

An LP has an affine objective and affine equality and inequality constraints:

$$
\begin{aligned}
\text{minimize}\quad & c^Tx\\
\text{subject to}\quad & Gx\le h,\\
& Ax=b.
\end{aligned}
$$

The inequality is componentwise, so

$$
Gx\le h
\quad\Longleftrightarrow\quad
h-Gx\in\mathbb R_+^m.
$$

Therefore an LP is a conic program over the nonnegative orthant.

### Doubt: why is it called linear when the feasible set has corners?

“Linear” describes the algebraic expressions in the objective and constraints. It does not mean that the feasible set is a line.

Each scalar affine inequality defines a halfspace. The intersection of finitely many halfspaces and affine sets is a polyhedron, which may have faces, edges, vertices, and unbounded directions.

For more about this geometry, see [LP and QP Geometry, Constraints, and First-Order Optimality](/posts/optimality-constraint-geometry/).

## 6. Second-order cone programming

A standard second-order cone constraint has the form

$$
\lVert A_i x+b_i\rVert_2
\le
c_i^Tx+d_i.
$$

It is equivalent to

$$
\left(c_i^Tx+d_i,\ A_i x+b_i\right)
\in
\mathcal Q.
$$

Both components of the pair are affine in $x$. The norm is represented by membership in the second-order cone.

The constraint also forces

$$
c_i^Tx+d_i\ge0,
$$

because a norm is nonnegative. This domain condition is already encoded by the cone and does not need to be added separately.

### Doubt: is every quadratic constraint an SOCP constraint?

No. A second-order cone represents a particular convex quadratic structure.

If $P\succeq0$, then a constraint involving

$$
\lVert P^{1/2}x\rVert_2
$$

has an SOCP representation because

$$
x^TPx
=
\lVert P^{1/2}x\rVert_2^2.
$$

An indefinite quadratic form has negative curvature in some direction. Its sublevel constraint is generally nonconvex and cannot automatically be represented as an SOCP constraint.

## 7. Semidefinite programming and LMIs

A semidefinite program commonly has the form

$$
\begin{aligned}
\text{minimize}\quad & c^Tx\\
\text{subject to}\quad &
F(x)=F_0+x_1F_1+\cdots+x_nF_n\succeq0,
\end{aligned}
$$

where each $F_i\in\mathbf S^m$.

The constraint

$$
F(x)\succeq0
$$

is called a **linear matrix inequality** (LMI). The name refers to the affine dependence of $F(x)$ on the decision variables. It does not mean that checking positive semidefiniteness is an entrywise linear comparison.

### Doubt: are we optimizing a vector or a matrix?

In the form above, the decision variable is $x\in\mathbb R^n$. The affine map $F$ turns $x$ into a symmetric matrix, and that matrix must belong to $\mathbf S_+^m$.

Another equivalent SDP notation can use a symmetric matrix $X$ directly as the decision variable, with affine constraints such as

$$
\operatorname{tr}(A_iX)=b_i,
\qquad
X\succeq0.
$$

The two notations emphasize different representations; they do not define different problem classes.

### Why is an LMI feasible set convex?

Suppose $F(x)\succeq0$ and $F(y)\succeq0$. For $0\le\theta\le1$, affinity gives

$$
F(\theta x+(1-\theta)y)
=
\theta F(x)+(1-\theta)F(y).
$$

The PSD cone is convex, so

$$
\theta F(x)+(1-\theta)F(y)\succeq0.
$$

Therefore every point on the segment between $x$ and $y$ is feasible.

## 8. Why LMIs use symmetric matrices

Positive semidefiniteness is defined through a quadratic form. For any real square matrix $F$, write

$$
F=S+N,
$$

where

$$
S=\frac{F+F^T}{2},
\qquad
N=\frac{F-F^T}{2}.
$$

The matrix $S$ is symmetric and $N$ is skew-symmetric.

Let

$$
a=z^TNz.
$$

Because $a$ is a scalar,

$$
a=a^T=z^TN^Tz.
$$

Since $N^T=-N$,

$$
a=-z^TNz=-a.
$$

Hence $a=0$, and therefore

$$
z^TFz=z^TSz.
$$

Only the symmetric part affects the quadratic form. Restricting an LMI to symmetric matrices removes the redundant skew-symmetric part and gives a real spectral description of positive semidefiniteness.

For complex matrices, “symmetric” is replaced by **Hermitian**:

$$
F=F^*,
$$

and the PSD condition becomes

$$
z^*Fz\ge0
\qquad
\text{for every complex }z.
$$

## 9. How an LP fits inside an SDP

The componentwise inequalities

$$
a_i^Tx\le b_i,
\qquad
i=1,\ldots,m,
$$

are equivalent to the diagonal LMI

$$
\operatorname{diag}
\left(
b_1-a_1^Tx,\ldots,b_m-a_m^Tx
\right)
\succeq0.
$$

A diagonal matrix is PSD exactly when each diagonal entry is nonnegative. Thus every LP has an SDP representation.

This is a statement about representability, not a recommendation to solve every LP with a generic SDP solver. A formulation using the smallest suitable cone is generally more direct and computationally efficient.

In this representational sense,

$$
\mathrm{LP}
\subseteq
\mathrm{SOCP}
\subseteq
\mathrm{SDP}.
$$

The first inclusion holds because a scalar nonnegativity constraint is a one-dimensional second-order cone constraint. The second inclusion is proved explicitly in Section 11 by representing a second-order cone constraint as an LMI.

### Doubt: why is every diagonal entry of a PSD matrix nonnegative?

Let $e_i$ be the $i$-th standard basis vector. If $X\succeq0$, then

$$
e_i^TXe_i\ge0.
$$

But

$$
e_i^TXe_i=X_{ii}.
$$

Therefore,

$$
X_{ii}\ge0
\qquad
\text{for every }i.
$$

The converse does not hold for a general symmetric matrix. Nonnegative diagonal entries control only the quadratic form along the coordinate axes. Off-diagonal terms can still make the quadratic form negative in another direction.

## 10. The Schur complement

Consider the symmetric block matrix

$$
M=
\begin{bmatrix}
A&B\\
B^T&C
\end{bmatrix},
\qquad
A\succ0.
$$

Define

$$
L=
\begin{bmatrix}
I&-A^{-1}B\\
0&I
\end{bmatrix}.
$$

The matrix $L$ is invertible. Direct block multiplication gives

$$
L^TML
=
\begin{bmatrix}
A&0\\
0&C-B^TA^{-1}B
\end{bmatrix}.
$$

An invertible congruence transformation preserves positive semidefiniteness. Therefore,

$$
M\succeq0
\quad\Longleftrightarrow\quad
C-B^TA^{-1}B\succeq0.
$$

The matrix

$$
C-B^TA^{-1}B
$$

is the Schur complement of $A$ in $M$.

### Why is this useful in optimization?

The expression $B^TA^{-1}B$ is nonlinear in the matrix blocks and contains an inverse. The equivalent block-matrix condition can be affine in the optimization variables and can therefore appear as an LMI.

For example, when $P\succ0$,

$$
t\ge x^TP^{-1}x
$$

is equivalent to

$$
\begin{bmatrix}
P&x\\
x^T&t
\end{bmatrix}
\succeq0.
$$

The Schur complement of $P$ in this block matrix is

$$
t-x^TP^{-1}x.
$$

Thus the PSD condition is exactly the original scalar inequality.

## 11. Representing a second-order cone as an LMI

The second-order cone constraint

$$
\lVert x\rVert_2\le t
$$

is equivalent to

$$
\begin{bmatrix}
tI&x\\
x^T&t
\end{bmatrix}
\succeq0.
$$

First suppose $t>0$. The Schur complement of $tI$ is

$$
t-x^T(tI)^{-1}x
=
t-\frac{\lVert x\rVert_2^2}{t}.
$$

The block matrix is PSD exactly when

$$
t-\frac{\lVert x\rVert_2^2}{t}\ge0.
$$

Multiplying by $t>0$ gives

$$
t^2\ge\lVert x\rVert_2^2.
$$

Because $t>0$, this is equivalent to

$$
t\ge\lVert x\rVert_2.
$$

If $t=0$, consider any component $x_i$. The corresponding principal submatrix is

$$
\begin{bmatrix}
0&x_i\\
x_i&0
\end{bmatrix}.
$$

Its eigenvalues are $x_i$ and $-x_i$, so it can be PSD only when $x_i=0$. This holds for every component, hence $x=0$, which agrees with $\lVert x\rVert_2\le t$. A negative $t$ is impossible because every diagonal entry of a PSD matrix must be nonnegative.

This proves the representation for every feasible $t$. It also explains the inclusion of SOCP in SDP at the level of representable constraints.

## 12. What the larger cones add

The three problem families differ in the type of positivity they can express:

- an LP constrains scalar affine expressions to be nonnegative;
- an SOCP constrains a scalar to dominate a Euclidean norm;
- an SDP constrains a quadratic form to be nonnegative in every direction.

An SDP constraint

$$
F(x)\succeq0
$$

compactly represents the infinite family of scalar inequalities

$$
z^TF(x)z\ge0
\qquad
\text{for every }z.
$$

This is why matrix-cone constraints are more expressive than componentwise scalar inequalities. The additional expressive power also typically makes the problems more expensive to solve, which is another reason to use the smallest cone that captures the model.

## 13. Connection to Perron–Frobenius theory

A cone selects which directions count as positive. This same idea appears in Perron–Frobenius theory.

If a linear map satisfies

$$
AK\subseteq K,
$$

then it preserves the cone: applying $A$ to a positive direction produces another positive direction. When $K$ induces an order, the map respects that ordered geometry.

This viewpoint connects generalized inequalities with positive linear systems, invariant cones, and eigenvector theory. The role of the cone is consistent across both subjects.

## 14. Consistency checklist

When reading or modeling a conic problem, ask:

1. What space does the cone live in: vectors, scalar-vector pairs, or symmetric matrices?
2. Is the cone merely convex, or does the argument require it to be proper?
3. What exactly does $x\preceq_K y$ mean?
4. Is the comparison componentwise or induced by the PSD cone?
5. Which object is the decision variable, and which affine map produces the cone-valued expression?
6. Are all matrices in an LMI symmetric?
7. Are the assumptions required by a Schur complement, such as $A\succ0$, stated?
8. Is the claimed LP, SOCP, or SDP representation an equivalence, or only a relaxation?
9. Is the formulation using the smallest standard cone that captures the constraint?

The unifying idea is:

> Keep the expressions affine and place the nonlinear geometry inside a convex cone.

---

This article is part of the [Convex Optimization learning map](/posts/convex-optimization-doubt-log/).

## Sources and further study

- Stephen Boyd and Lieven Vandenberghe, [*Convex Optimization*, Chapters 2 and 4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
- Stephen Boyd, [EE364A lecture slides](https://web.stanford.edu/class/ee364a/lectures.html).
