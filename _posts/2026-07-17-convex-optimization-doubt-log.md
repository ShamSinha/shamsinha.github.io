---
title: "Convex Optimization Doubt Log — The Full Learning Conversation"
date: 2026-07-17 00:00:00 +0000
categories: [Mathematics, Optimization]
tags: [convexity, quasiconvexity, linear-programming, quadratic-programming, doubts, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/convex-optimization-doubt-log/
description: "A conversational, doubt-by-doubt set of convex optimization notes that keeps the questions, corrections, intuition, proofs, and examples."
---

These are deliberately **not** polished like a textbook chapter.

The goal is to preserve the learning process: the doubts, the almost-correct statements, the corrections, the geometric intuition, and the small examples that made the ideas click.

Some repetition has been removed, but most of the useful discussion has been kept.

---

## 1. First confusion: what are we actually optimizing, $x$ or $t$?

> **Doubt:** Here we want the optimal value of $x$ or $t$? $t$ is scalar, while $x$ may or may not be scalar.

Start with the original problem

$$
\min_x f_0(x)
$$

subject to some constraints.

There are two different objects:

- $x$ is the **decision variable**. It may be a scalar, but usually it is a vector in $\mathbb{R}^n$.
- $f_0(x)$ is the objective value, which is scalar.

An optimal decision point is written $x^\star$.

The optimal value is a scalar, often written

$$
p^\star = \inf\{f_0(x)\mid x\text{ is feasible}\}.
$$

So:

- $x^\star$: optimal point or optimizer;
- $p^\star$: optimal objective value.

In a level-set or bisection method, we introduce a scalar $t$. It is a **trial objective value**.

For fixed $t$, we ask:

$$
\text{Does there exist an }x\text{ such that }f_0(x)\le t
$$

and all the original constraints hold?

If yes, then the true optimum satisfies $p^\star\le t$.

If no, then $p^\star>t$.

We are not optimizing $x$ inside the feasibility test. We are only asking for any feasible witness $x$.

At the end of bisection:

- the scalar interval converges to the optimal value;
- a stored feasible $x$ is an approximate optimizer.

There may be many optimal $x$ values but only one optimal objective value. Also, an infimum may exist without being attained, in which case there is no exact $x^\star$.

---

## 2. Why is an LP feasible set a polyhedron?

Consider

$$
Gx\le h,\qquad Ax=b.
$$

The feasible set is

$$
\mathcal F=\{x\mid Gx\le h,\ Ax=b\}.
$$

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

## 3. What does convex mean for a set?

A set $C$ is convex when, for every $x,y\in C$ and every $\theta\in[0,1]$,

$$
\theta x+(1-\theta)y\in C.
$$

In plain language: draw the line segment between any two points in the set. The whole segment must stay inside the set.

### Why are halfspaces convex?

Suppose

$$
a^Tx\le b,\qquad a^Ty\le b.
$$

For $z=\theta x+(1-\theta)y$,

$$
a^Tz
=\theta a^Tx+(1-\theta)a^Ty
\le \theta b+(1-\theta)b=b.
$$

Thus $z$ also belongs to the halfspace.

### Why do intersections remain convex?

If every set $C_i$ contains the segment between any two of its points, then a pair of points that belongs to every $C_i$ has its entire segment in every $C_i$. Hence the segment is in the intersection.

That one fact explains why systems of linear inequalities give convex feasible regions.

---

## 4. What is a convex function?

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

## 5. Why does the gradient define a tangent or supporting hyperplane?

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

## 6. Why does convexity make every local minimum global?

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

## 7. What does a convex quadratic mean?

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

### What does $P\succeq0$ mean?

It means

$$
z^TPz\ge0\qquad\text{for every }z.
$$

Equivalently, when $P$ is symmetric, all eigenvalues of $P$ are nonnegative.

Geometrically, the quadratic has nonnegative curvature in every direction.

### Is $P$ a matrix or a cone?

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

### Why the factor $1/2$?

It is only a convenience:

$$
\nabla\left(\frac12x^TPx\right)=Px
$$

for symmetric $P$.

---

## 8. What happens when $P$ is not PSD?

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

## 9. Why can we assume the quadratic matrix is symmetric?

Any square matrix can be decomposed as

$$
P=\frac{P+P^T}{2}+\frac{P-P^T}{2}.
$$

The first term is symmetric and the second is skew-symmetric.

For a skew-symmetric matrix $S^T=-S$,

$$
x^TSx=0.
$$

Therefore,

$$
x^TPx=x^T\left(\frac{P+P^T}{2}\right)x.
$$

Only the symmetric part affects the quadratic form.

---

## 10. Block quadratic forms and the Schur complement

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

## 11. Piecewise-linear minimization

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

## 12. Convex versus quasiconvex

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

---

## 13. Why is a linear-fractional objective quasiconvex?

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

## 14. Proof: $\sqrt{|x|}$ is quasiconvex

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

## 15. Proof: $\lceil x\rceil$ is quasilinear

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

## 16. Proof: $\log x$ is quasilinear on $x>0$

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

## 17. Proof: $x_1x_2$ is quasiconcave on $\mathbb R_{++}^2$

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

## 18. Quasiconvex optimization by bisection

Consider

$$
\begin{array}{ll}
\text{minimize} & f_0(x)\\
\text{subject to} & f_i(x)\le0,\quad i=1,\ldots,m,\\
& Ax=b,
\end{array}
$$

where $f_0$ is quasiconvex and the feasible constraints are convex.

For fixed $t$, solve the feasibility problem

$$
\begin{array}{ll}
\text{find} & x\\
\text{subject to} & f_0(x)\le t,\\
& f_i(x)\le0,\\
& Ax=b.
\end{array}
$$

Because a quasiconvex function has convex sublevel sets, the additional condition $f_0(x)\le t$ is convex as a set.

Feasibility is monotone in $t$:

- feasible at $t$ implies feasible at every larger value;
- infeasible at $t$ implies infeasible at every smaller value.

This monotonicity permits bisection.

### Bisection logic

Maintain a lower bound $l$ known to be infeasible and an upper bound $u$ known to be feasible.

Set

$$
t=\frac{l+u}{2}.
$$

- If feasible, store the witness $x$ and set $u=t$.
- If infeasible, set $l=t$.

Stop when $u-l$ is small.

Then $u$ approximates the optimal value and the stored feasible point approximates an optimizer.

### Why not optimize $x$ and $t$ jointly?

For a general quasiconvex function, the epigraph

$$
\{(x,t)\mid f_0(x)\le t\}
$$

need not be convex. The sublevel set is guaranteed convex only when $t$ is fixed. That is why $t$ is handled externally by bisection.

---

## 19. Ratio example in quasiconvex optimization

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

## 20. LP optimum: must it always be a vertex?

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

## 21. Why can a quadratic-program solution be inside a face or in the interior?

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

## 22. First-order optimality over a convex feasible set

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

## 23. Equality constraints and the nullspace

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

## 24. Eliminating equality constraints

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

## 25. Log-concavity and the Gaussian example

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

## 26. Frequent statements that need correction

### "Quasiconvex means positive curvature"

No. Quasiconvexity is defined by convex sublevel sets, not by a Hessian condition.

### "Quasilinear means affine"

No. It means both quasiconvex and quasiconcave. Functions such as $\log x$ and $\lceil x\rceil$ are quasilinear on their domains but not affine.

### "The LP solution is always a single vertex"

No. The optimal set can be an edge or a higher-dimensional face. Under standard assumptions, at least one vertex is optimal.

### "A QP optimum cannot be at a vertex"

It can be at a vertex. It simply does not have to be.

### "$P$ is the PSD cone"

No. $P$ is one PSD matrix. $\mathbf S_+^n$ is the PSD cone.

### "In bisection, $t$ is the decision vector"

No. $t$ is a scalar candidate objective level. $x$ remains the original decision variable.

### "$p(x)/q(x)\le t$ becomes $p(x)-tq(x)=0$"

No. It becomes

$$
p(x)-tq(x)\le0.
$$

### "For a ratio, multiply by the denominator without checking it"

Never. The sign of the denominator determines whether the inequality direction is preserved.

### "Every polyhedron has a vertex"

No. An affine line or affine plane is a polyhedron and may have no extreme points.

---

## 27. A compact comparison table

| Topic | Key object | Convexity condition | Where an optimum may occur |
|---|---|---|---|
| LP | $c^Tx$ | objective affine, feasible set polyhedral | vertex, edge, face; an optimal vertex often exists |
| Convex QP | $\frac12x^TPx+q^Tx+r$ | $P\succeq0$ | interior, face interior, edge, or vertex |
| Piecewise-linear | $\max_i(a_i^Tx+b_i)$ | maximum of affine functions | often at a nonsmooth intersection, but not necessarily |
| Quasiconvex problem | convex sublevel sets | fixed-level feasibility must be convex | found by objective-level search such as bisection |
| Linear-fractional | ratio of affine functions | positive denominator | level sets are halfspaces |

---

## 28. A practical mental checklist

When you see a new optimization problem, ask:

1. What is the decision variable? Is it scalar or vector?
2. Is the objective scalar-valued?
3. What is the feasible set?
4. Are the constraints convex?
5. Is the objective convex, quasiconvex, or neither?
6. For a quadratic, is the symmetric Hessian PSD?
7. For a ratio, what is the sign of the denominator?
8. Can a max objective be converted using an epigraph variable $t$?
9. Does $t$ represent a joint variable or a fixed level in a feasibility test?
10. Is the claim about an optimal vertex an existence statement or a statement about every optimizer?

That checklist prevents many common mistakes.

---

## 29. Final picture

Convex optimization becomes easier when the algebra and geometry are connected:

- linear inequalities create halfspaces;
- intersections of halfspaces create polyhedra;
- gradients are normals to level sets;
- PSD Hessians mean nonnegative curvature;
- max-of-affine objectives become LPs through an epigraph variable;
- quasiconvexity is about the shape of sublevel sets;
- bisection searches objective values using convex feasibility tests;
- linear objectives can be flat across a face;
- curved quadratic objectives can choose an interior point of a face or of the entire feasible set.

The main lesson is not to memorize slogans such as "LP means vertex" or "PSD means bowl." Keep the assumptions attached to every statement and test each claim with a two-dimensional example.
