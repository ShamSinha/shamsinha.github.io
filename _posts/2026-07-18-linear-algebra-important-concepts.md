---
title: "Linear Algebra: Important Concepts and the Doubts That Connect Them"
date: 2026-07-18 10:00:00 +0530
last_modified_at: 2026-08-09 00:00:00 +0530
categories: [Mathematics, Linear Algebra]
tags: [matrices, eigenvalues, diagonalization, svd, determinants, projections, gram-schmidt, perron-frobenius, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/linear-algebra-important-concepts/
description: "A focused, doubt-driven guide to matrices as transformations, eigenvalues, diagonalization, SVD, projections, determinants, matrix powers, and Perron–Frobenius theory."
---

A matrix can stretch, rotate, reflect, mix, or collapse directions. The notation $Ax$ hides all of this behavior inside one short expression. Linear algebra gives us different tools for exposing that behavior: eigenvectors reveal simple directions, diagonalization separates coupled motion, determinants detect lost dimensions, and matrix powers describe repeated action.

These tools can look like unrelated definitions. They become one story once a matrix is viewed as an action on space.

This is not a complete linear algebra textbook. It is a focused collection of the concepts I found important, organized so that each definition appears before the doubt that depends on it.

The central idea connecting the whole post is:

> A matrix is not merely a table of numbers. It represents a transformation, and its algebraic properties tell us what that transformation preserves, scales, collapses, or reveals.

## How the ideas solve different problems

| Starting problem | Idea introduced | What becomes clear |
|---|---|---|
| A table of coefficients does not show what happens geometrically to an input. | [Matrices as transformations](#matrix-transformations) | The columns show where basis directions go, and therefore determine every output. |
| Some transformations preserve lengths, while others have especially clean directional structure. | [Symmetric and orthogonal matrices](#special-matrices) | Orthogonal matrices preserve geometry; symmetric matrices admit orthogonal eigenvector coordinates. |
| A transformation mixes coordinates, making its behavior difficult to follow. | [Eigenvalues and eigenvectors](#eigenvalues) | Certain directions change only by a scalar factor. |
| We need compact summaries of scaling and dimensional collapse. | [Trace and determinant](#trace-determinant) | Trace adds the eigenvalues; determinant multiplies them and detects singularity. |
| Even after finding eigenvectors, the original matrix may still look coupled. | [Diagonalization](#diagonalization), [spectral decomposition, and SVD](#spectral-svd) | The transformation separates into independent modes or input-output directions. |
| An independent basis need not make lengths, angles, or nearest points easy to compute. | [Projection and Gram–Schmidt](#projection-gram-schmidt) | Orthogonal coordinates separate components and make projection direct. |
| Applying the same matrix repeatedly is expensive and hard to interpret entry by entry. | [Matrix powers](#matrix-powers) | Eigenmodes and graph walks explain repeated evolution. |
| In a connected system with nonnegative interactions, we want to identify a mode whose entries are all positive. | [Perron–Frobenius theory](#perron-frobenius) | Under the theorem's conditions, the spectral radius has a positive eigenvector. |
| A transformation may lose information, making reversal impossible. | [Determinant and invertibility](#determinant-invertibility) | A zero determinant, a collapsed direction, a nontrivial nullspace, and non-invertibility are the same failure viewed differently. |

## How to read these notes

For a first pass, follow the sections in order: transformation $\rightarrow$ eigenvectors $\rightarrow$ diagonalization $\rightarrow$ repeated motion. If your immediate interest is optimization, read Sections 1–7 and then use the focused companion [Symmetric Matrices, Quadratic Forms, and Matrix Norm](/posts/symmetric-matrices-quadratic-forms-matrix-norm/). If your interest is dynamics or networks, continue through matrix powers and Perron–Frobenius theory.

The companion article goes deeper into one continuous chain—symmetric matrices, quadratic forms, PSD matrices, ellipsoids, and matrix gain—while this article remains the broader map.

---

## 1. Matrices as transformations {#matrix-transformations}

Let

$$
A\in\mathbb R^{m\times n}.
$$

Multiplication by $A$ defines a linear transformation

$$
x\in\mathbb R^n
\longmapsto
Ax\in\mathbb R^m.
$$

So a rectangular matrix can change the dimension of a vector. A square matrix

$$
A\in\mathbb R^{n\times n}
$$

maps $\mathbb R^n$ back into the same space.

### What do the columns mean?

If

$$
A=
\begin{bmatrix}
a_1&a_2&\cdots&a_n
\end{bmatrix},
$$

then

$$
Ax=x_1a_1+x_2a_2+\cdots+x_na_n.
$$

The columns tell us where the standard basis vectors go:

$$
Ae_i=a_i.
$$

Once we know what the transformation does to a basis, linearity tells us what it does to every vector.

### Geometric examples

A diagonal matrix

$$
D=\operatorname{diag}(d_1,\ldots,d_n)
$$

scales each coordinate axis independently:

$$
Dx=(d_1x_1,\ldots,d_nx_n).
$$

An orthogonal matrix preserves lengths and angles. It can represent a rotation, a reflection, or a combination of the two. Saying “orthogonal matrix means rotation” is therefore incomplete.

---

## 2. Symmetric and orthogonal matrices {#special-matrices}

These two classes are easy to confuse because both use the transpose, but they express different ideas.

### Symmetric matrix

A square matrix is symmetric if

$$
A=A^T.
$$

Symmetry is associated with real eigenvalues, orthogonal eigenspaces, quadratic forms, covariance matrices, and many optimization problems.

### Orthogonal matrix

A real square matrix $Q$ is orthogonal if

$$
Q^TQ=I,
$$

or equivalently,

$$
Q^{-1}=Q^T.
$$

If the columns are $q_1,\ldots,q_n$, this says

$$
q_i^Tq_j=
\begin{cases}
1,&i=j,\\
0,&i\ne j.
\end{cases}
$$

The columns form an orthonormal basis.

### Why does an orthogonal matrix preserve length?

For every $x$,

$$
\|Qx\|_2^2
=(Qx)^T(Qx)
=x^TQ^TQx
=x^Tx
=\|x\|_2^2.
$$

It also preserves inner products:

$$
(Qx)^T(Qy)=x^Ty.
$$

That is why angles and distances survive the transformation.

---

## 3. Eigenvalues and eigenvectors {#eigenvalues}

For a square matrix $A$, a nonzero vector $v$ is a right eigenvector with eigenvalue $\lambda$ if

$$
Av=\lambda v.
$$

### Doubt: what is lambda actually doing?

Usually, applying $A$ changes both the magnitude and direction of a vector. An eigenvector is a special direction whose orientation is preserved up to sign or complex phase. The matrix only scales it by $\lambda$.

For real $\lambda$:

- $\lambda>1$: the vector grows and keeps its direction;
- $0<\lambda<1$: it shrinks and keeps its direction;
- $\lambda<0$: it reverses direction and is scaled by $\lvert\lambda\rvert$;
- $\lambda=0$: it is collapsed to zero.

Eigenvectors themselves are not unique. If $v$ is an eigenvector, then every nonzero multiple $cv$ is also an eigenvector for the same eigenvalue.

### How do we find eigenvalues?

Rearrange the eigenvalue equation:

$$
(A-\lambda I)v=0.
$$

A nonzero solution exists only when $A-\lambda I$ is singular, so

$$
\det(A-\lambda I)=0.
$$

This is the characteristic equation.

### Left eigenvectors

A row vector $w^T$ is a left eigenvector if

$$
w^TA=\lambda w^T.
$$

Equivalently, $w$ is a right eigenvector of $A^T$.

If $A$ is diagonalizable as

$$
A=V\Lambda V^{-1},
$$

then

$$
V^{-1}A=\Lambda V^{-1}.
$$

Therefore the rows $w_i^T$ of $V^{-1}$ are left eigenvectors:

$$
w_i^TA=\lambda_iw_i^T.
$$

With this normalization, the left and right eigenvectors form dual bases:

$$
w_i^Tv_j=\delta_{ij}.
$$

### Can a real matrix have complex eigenvalues?

Yes. A real matrix may have complex eigenvalues and eigenvectors. Nonreal eigenvalues of a real matrix occur in conjugate pairs. If

$$
Av=\lambda v,
$$

then, because $A$ is real,

$$
A\overline v=\overline\lambda\,\overline v.
$$

If both $A$ and $\lambda$ are real, at least one of $\operatorname{Re}(v)$ and $\operatorname{Im}(v)$ is a nonzero real eigenvector for $\lambda$. So a real eigenvalue of a real matrix always has a real eigenvector available.

For example, a two-dimensional rotation through an angle other than $0$ or $\pi$ has no real eigenvector: no real direction remains fixed.

### Doubt: are eigenvectors of a symmetric matrix always orthogonal?

The careful statements are:

- eigenvectors belonging to distinct eigenvalues are orthogonal;
- within a repeated eigenspace, arbitrary eigenvectors need not already be orthogonal, but an orthonormal basis can be chosen.

The proof, together with the proof that every eigenvalue is real, is kept in [Why symmetric matrices are special](/posts/symmetric-matrices-quadratic-forms-matrix-norm/#symmetric-matrices).

---

## 4. Trace and determinant through eigenvalues {#trace-determinant}

For a square matrix $A$, the trace is the sum of its diagonal entries:

$$
\operatorname{tr}(A)=\sum_{i=1}^n A_{ii}.
$$

Important properties include

$$
\operatorname{tr}(A+B)
=\operatorname{tr}(A)+\operatorname{tr}(B),
$$

$$
\operatorname{tr}(cA)=c\operatorname{tr}(A),
$$

$$
\operatorname{tr}(A^T)=\operatorname{tr}(A),
$$

and the cyclic identity

$$
\operatorname{tr}(AB)=\operatorname{tr}(BA).
$$

More generally,

$$
\operatorname{tr}(ABC)
=\operatorname{tr}(BCA)
=\operatorname{tr}(CAB),
$$

but arbitrary reordering is not allowed.

If the eigenvalues of $A$ are $\lambda_1,\ldots,\lambda_n$, counted with algebraic multiplicity, then

$$
\operatorname{tr}(A)=\sum_{i=1}^n\lambda_i,
$$

and

$$
\det(A)=\prod_{i=1}^n\lambda_i.
$$

These identities remain true even when $A$ is not diagonalizable.

### Why is trace the sum of eigenvalues?

For a diagonalizable matrix,

$$
A=V\Lambda V^{-1}.
$$

Using the cyclic property,

$$
\operatorname{tr}(A)
=\operatorname{tr}(V\Lambda V^{-1})
=\operatorname{tr}(\Lambda V^{-1}V)
=\operatorname{tr}(\Lambda).
$$

The trace of $\Lambda$ is the sum of its diagonal entries, which are the eigenvalues:

$$
\operatorname{tr}(A)=\lambda_1+\cdots+\lambda_n.
$$

This proof uses diagonalization for intuition. The identity remains valid for a defective matrix as well, for example by reading the coefficient of $\lambda^{n-1}$ in its characteristic polynomial.

Similarly, when $A$ is diagonalizable,

$$
\det(A)
=\det(V)\det(\Lambda)\det(V^{-1})
=\det(\Lambda)
=\prod_{i=1}^n\lambda_i.
$$

The memory pair is:

- trace = sum of eigenvalues;
- determinant = product of eigenvalues.

---

## 5. Diagonalization {#diagonalization}

A matrix $A\in\mathbb R^{n\times n}$ is diagonalizable if it has $n$ linearly independent eigenvectors.

Place them into the columns of

$$
V=
\begin{bmatrix}
v_1&\cdots&v_n
\end{bmatrix},
$$

and place the associated eigenvalues into

$$
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n).
$$

The equations $Av_i=\lambda_i v_i$ combine into

$$
AV=V\Lambda.
$$

Since the eigenvectors are linearly independent, $V$ is invertible, giving

$$
A=V\Lambda V^{-1}
$$

or

$$
V^{-1}AV=\Lambda.
$$

### Why is diagonalization useful?

Working with a diagonal matrix is easy. For example,

$$
A^k=V\Lambda^kV^{-1},
$$

where

$$
\Lambda^k
=\operatorname{diag}(\lambda_1^k,\ldots,\lambda_n^k).
$$

Diagonalization reveals independent directions along which the transformation simply scales.

### Are all matrices diagonalizable?

No. Consider

$$
A=
\begin{bmatrix}
0&1\\
0&0
\end{bmatrix}.
$$

Its only eigenvalue is $0$, and its eigenspace is one-dimensional. A $2\times2$ diagonalization would require two independent eigenvectors, so this matrix is defective.

Distinct eigenvalues guarantee diagonalizability. The converse is false: a matrix may have repeated eigenvalues and still have enough independent eigenvectors.

For a repeated eigenvalue $\lambda$:

- its **algebraic multiplicity** is how many times it appears as a root of the characteristic polynomial;
- its **geometric multiplicity** is the dimension of $\ker(A-\lambda I)$, which counts the independent eigenvectors available for that eigenvalue.

The geometric multiplicity can never exceed the algebraic multiplicity. Assuming the characteristic polynomial splits over the field being used, $A$ is diagonalizable exactly when these two multiplicities agree for every eigenvalue.

---

## 6. Spectral decomposition and SVD {#spectral-svd}

This section records the distinction needed for a general linear-algebra map. The geometric derivations, matrix-norm connection, and detailed proofs are kept in [Symmetric Matrices, Quadratic Forms, and Matrix Norms](/posts/symmetric-matrices-quadratic-forms-matrix-norm/).

### Spectral decomposition

A real symmetric matrix has an orthogonal spectral decomposition

$$
A=Q\Lambda Q^T.
$$

The columns of $Q$ are orthonormal eigenvectors and $\Lambda$ contains the real eigenvalues. Conceptually, $Q^T$ changes to eigenvector coordinates, $\Lambda$ scales those coordinates, and $Q$ changes back.

A general diagonalizable matrix instead has

$$
A=V\Lambda V^{-1},
$$

where $V$ need not be orthogonal. The stronger orthogonal form is a consequence of symmetry.

### Singular value decomposition

Every real matrix, including rectangular and nondiagonalizable matrices, has an SVD:

$$
A=U\Sigma V^T.
$$

The columns of $V$ are orthonormal input directions, the columns of $U$ are orthonormal output directions, and the singular values in $\Sigma$ are nonnegative gains. Unlike symmetric eigendecomposition, SVD may use different input and output directions.

For the derivation from $A^TA$, the interpretation of singular values, and the relation

$$
\lVert A\rVert_2=\sigma_{\max}(A),
$$

continue with [SVD as the full directional explanation of matrix norm](/posts/symmetric-matrices-quadratic-forms-matrix-norm/#svd-and-norm).

---

## 7. Independence, orthogonality, projection, and Gram–Schmidt {#projection-gram-schmidt}

### Independent does not mean orthogonal

Vectors $v_1,\ldots,v_k$ are linearly independent if

$$
c_1v_1+\cdots+c_kv_k=0
$$

implies $c_1=\cdots=c_k=0$.

Orthogonality is stronger:

$$
v_i^Tv_j=0
\qquad(i\ne j).
$$

Every set of nonzero mutually orthogonal vectors is independent, but independent vectors need not be perpendicular.

### Projection of one vector onto another

To project a vector $v$ onto the direction $u$, we seek a vector $\alpha u$ such that the error is perpendicular to $u$:

$$
u^T(v-\alpha u)=0.
$$

Solving gives

$$
\alpha=\frac{u^Tv}{u^Tu}.
$$

Therefore

$$
\operatorname{proj}_u(v)
=\frac{u^Tv}{u^Tu}u.
$$

The Notion heading says “projection of $q$ along $a$,” but the handwritten equations decompose $a$ into a component parallel to $q$ and a perpendicular remainder. Therefore the calculation in the image is the projection of $a$ onto $q$. Substituting $v=a$ and $u=q$ gives

$$
\operatorname{proj}_q(a)
=\frac{q^Ta}{q^Tq}q.
$$

If $q$ is a unit vector, this simplifies to

$$
\operatorname{proj}_q(a)=(q^Ta)q.
$$

### Why Gram–Schmidt?

Gram–Schmidt turns independent vectors into an orthonormal basis spanning the same subspace.

Starting with $v_1,\ldots,v_k$,

$$
u_1=v_1,
$$

$$
u_2=v_2-\operatorname{proj}_{u_1}(v_2),
$$

and generally

$$
u_j
=v_j-
\sum_{i=1}^{j-1}
\operatorname{proj}_{u_i}(v_j).
$$

Normalize with

$$
q_j=\frac{u_j}{\|u_j\|}.
$$

Each step removes the components pointing along earlier directions. The result underlies QR factorization, least squares, and stable coordinate systems.

---

## 8. Matrix powers {#matrix-powers}

For a square matrix $A$, positive powers mean repeated composition:

$$
A^k=\underbrace{A\cdots A}_{k\text{ times}}.
$$

If $A$ is invertible,

$$
A^{-k}=(A^{-1})^k
$$

and $A^0=I$.

For integer exponents for which the expressions are defined,

$$
A^{k+\ell}=A^kA^\ell,
$$

and

$$
(A^k)^{-1}=(A^{-1})^k=A^{-k}.
$$

### Graph interpretation

If $A$ is the adjacency matrix of a directed graph, then

$$
(A^k)_{ij}
$$

counts the number of directed walks of length $k$ from node $i$ to node $j$.

Why? Matrix multiplication sums over every possible intermediate node. Repeating it $k$ times counts every possible sequence of $k$ edges.

### Long-term behavior

When $A$ is diagonalizable,

$$
A^k=V\Lambda^kV^{-1}.
$$

The eigenvalues determine which modes grow, decay, oscillate, or disappear. This is why eigenanalysis matters in dynamical systems, Markov chains, graph algorithms, and iterative methods.

---

## 9. Perron–Frobenius theory {#perron-frobenius}

This section follows naturally from matrix powers. For a nonnegative matrix, repeated multiplication propagates influence through its graph, while the dominant eigenvalue determines the long-term growth of $A^k$.

Perron–Frobenius theory makes that dominant eigenstructure precise.

### Nonnegative and irreducible matrices

A matrix is nonnegative if

$$
A_{ij}\ge0
\qquad\text{for every }i,j.
$$

Associate a directed graph with $A$ by drawing an edge $i\to j$ whenever $A_{ij}>0$.

The matrix is **irreducible** when this graph is strongly connected: every node can reach every other node.

For an $n\times n$ nonnegative matrix, an equivalent test is

$$
(I+A)^{n-1}>0
$$

elementwise. Adding $I$ creates self-loops, allowing a path of length at most $n-1$ to be extended to exactly $n-1$ steps.

### What does the theorem guarantee?

For a nonnegative irreducible matrix $A$:

- its spectral radius $\rho(A)$ is a positive eigenvalue;
- $\rho(A)$ has a strictly positive eigenvector $v\succ0$;
- the Perron eigenvalue is algebraically simple;
- every eigenvalue satisfies $\lvert\lambda_i\rvert\le\rho(A)$.

Thus

$$
Av=\rho(A)v,
\qquad
v\succ0.
$$

### Important correction: irreducible versus positive

Irreducibility alone does not guarantee

$$
\lvert\lambda_i\rvert<\rho(A)
$$

for every other eigenvalue. An irreducible but periodic matrix may have other eigenvalues on the same spectral circle.

If $A$ is **positive**, meaning every entry is strictly positive, then the Perron eigenvalue is strictly dominant in magnitude.

Why require irreducibility at all? Consider the reducible matrix

$$
A=
\begin{bmatrix}
3&0\\
0&1
\end{bmatrix}.
$$

Its dominant eigenvector is $(1,0)^T$, which is nonnegative but not strictly positive. The two coordinates form disconnected blocks, so dominance in one block says nothing about the other. Irreducibility removes this separation.

### Why can the second eigenvector not be strictly positive?

For a positive matrix, the interior of the positive cone contains only the Perron eigenvector direction. Another eigenvector with all positive components would have to be proportional to the Perron eigenvector, which is impossible if it corresponds to a different eigenvalue.

### Collatz–Wielandt viewpoint

One useful characterization is

$$
\rho(A)
=
\inf_{v\succ0}
\max_i\frac{(Av)_i}{v_i}.
$$

Equivalently,

$$
\rho(A)
=
\inf\{\lambda\mid Av\preceq\lambda v,\ v\succ0\}.
$$

This form connects Perron–Frobenius theory directly to optimization.

---

## 10. Why determinant zero means non-invertible {#determinant-invertibility}

The important point is not merely the rule

$$
\det(A)=0
\quad\Longleftrightarrow\quad
A\text{ has no inverse}.
$$

We want to understand **why** it happens.

### Start with what an inverse must do

If

$$
y=Ax,
$$

then $A^{-1}$ must recover the one original input:

$$
x=A^{-1}y.
$$

Therefore different inputs cannot be allowed to produce the same output. If $Ax_1=Ax_2$ for $x_1\ne x_2$, an inverse receiving that output would not know which input to return.

### What does a zero determinant tell us?

The columns of $A$ are the transformed standard basis vectors:

$$
Ae_i=a_i.
$$

The quantity $\lvert\det(A)\rvert$ is the volume of the shape spanned by these transformed basis vectors. In two dimensions, for

$$
A=
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix},
$$

the two columns span a parallelogram whose signed area is

$$
ad-bc=\det(A).
$$

If this area is zero, the two columns do not span a genuine two-dimensional region: they lie on the same line. In higher dimensions the same idea holds—zero $n$-dimensional volume means the columns lie in a lower-dimensional subspace.

So $\det(A)=0$ means the columns are linearly dependent. There are coefficients, not all zero, such that

$$
z_1a_1+\cdots+z_na_n=0.
$$

If $z=(z_1,\ldots,z_n)^T$, this is exactly

$$
Az=0
\qquad\text{for some }z\ne0.
$$

This is the missing mechanism: **the matrix collapses a nonzero direction $z$ to zero**.

### Why does a collapsed direction destroy the inverse?

For any input $x$,

$$
A(x+z)=Ax+Az=Ax.
$$

The two different inputs $x$ and $x+z$ therefore have the same output. In fact, every point $x+tz$ on that line has the same output:

$$
A(x+tz)=Ax
\qquad\text{for every scalar }t.
$$

Information about movement in the $z$ direction has disappeared. No inverse can reconstruct information that the forward transformation erased.

We can also prove the contradiction directly. If $A^{-1}$ existed, applying it to $A(x+z)=Ax$ would give

$$
x+z=x,
$$

which would imply $z=0$. But we already found $z\ne0$.

### Concrete singular matrix

Consider

$$
A=
\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}.
$$

The second column is twice the first, so the columns are dependent. Also,

$$
\det(A)=1\cdot4-2\cdot2=0.
$$

The dependency can be written as

$$
-2
\begin{bmatrix}
1\\
2
\end{bmatrix}
+
\begin{bmatrix}
2\\
4
\end{bmatrix}
=0.
$$

Therefore the nonzero vector

$$
z=
\begin{bmatrix}
-2\\
1
\end{bmatrix}
$$

is collapsed:

$$
Az=0.
$$

Now take

$$
x_1=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
x_2=
\begin{bmatrix}
-1\\
1
\end{bmatrix}.
$$

Although $x_1\ne x_2$,

$$
Ax_1=
\begin{bmatrix}
1\\
2
\end{bmatrix}
=Ax_2.
$$

Given that output, an inverse would have to return two different inputs at once. That is impossible.

### What if the determinant is nonzero?

If $\det(A)\ne0$, the columns span the full space, so there is no nonzero collapsed direction:

$$
Az=0
\quad\Longrightarrow\quad
z=0.
$$

The transformation is then one-to-one. For a square matrix from $\mathbb R^n$ to $\mathbb R^n$, full rank also makes it onto, so every output has exactly one input and the inverse exists.

Now the usual equivalences have a reason behind them:

$$
\det(A)\ne0
\Longleftrightarrow
\operatorname{rank}(A)=n
\Longleftrightarrow
\ker(A)=\{0\}
\Longleftrightarrow
A\text{ is invertible}.
$$

### Connection to eigenvalues

An eigenvalue $\lambda$ makes

$$
A-\lambda I
$$

singular:

$$
\det(A-\lambda I)=0.
$$

In particular, $A$ is singular exactly when $0$ is one of its eigenvalues. The associated eigenvector is precisely a nonzero direction that $A$ collapses to zero.

---

## Final mental model

- A matrix is a transformation.
- Its columns show where basis vectors go.
- Eigenvectors reveal directions transformed only by scaling.
- Diagonalization separates a transformation into independent eigenvector modes.
- Spectral decomposition is especially clean for symmetric matrices.
- SVD provides orthogonal input and output directions for every matrix.
- Projection and Gram–Schmidt create useful orthogonal coordinates.
- Trace and determinant summarize eigenvalues by their sum and product.
- A zero determinant signals collapsed dimension and lost invertibility.
- Matrix powers describe repeated transformations and graph walks.
- Perron–Frobenius identifies the dominant positive mode of a nonnegative irreducible system.

The unifying question is always geometric: **what does this matrix do to directions, lengths, dimensions, and repeated motion?**

---

## Where to continue

- Follow the geometric deep dive in [Symmetric Matrices, Quadratic Forms, and Matrix Norm](/posts/symmetric-matrices-quadratic-forms-matrix-norm/).
- See how eigenvalues, PSD matrices, nullspaces, and feasible directions control optimization in [Quadratic and Constrained Convex Optimization](/posts/convex-quadratics-psd-schur-complement/).
- Return to the broader [Convex Optimization learning map](/posts/convex-optimization-doubt-log/) when you want to connect these tools to convex sets, quasiconvexity, and conic optimization.

## Sources

- [EE263: Introduction to Linear Dynamical Systems](https://ee263.stanford.edu/archive/), Stephen Boyd, Stanford University
- [EE364A: Convex Optimization I](https://see.stanford.edu/Course/EE364A), Stephen Boyd, Stanford Engineering Everywhere
- [Visualize Spectral Decomposition — SEE Matrix, Chapter 2](https://www.youtube.com/watch?v=mhy-ZKSARxI), Visual Kernel
