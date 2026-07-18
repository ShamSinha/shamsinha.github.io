---
title: "Linear Algebra: Important Concepts and the Doubts That Connect Them"
date: 2026-07-18 10:00:00 +0530
categories: [Mathematics, Linear Algebra]
tags: [matrices, eigenvalues, diagonalization, svd, determinants, projections, gram-schmidt, perron-frobenius, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/linear-algebra-important-concepts/
description: "A focused, doubt-driven guide to matrices as transformations, eigenvalues, diagonalization, SVD, projections, determinants, matrix powers, and Perron–Frobenius theory."
---

This is not a complete linear algebra textbook. It is a focused collection of the concepts I found important, organized so that each definition appears before the doubt that depends on it.

The central idea connecting the whole post is:

> A matrix is not merely a table of numbers. It represents a transformation, and its algebraic properties tell us what that transformation preserves, scales, collapses, or reveals.

## Topic map

- [Matrices as transformations](#matrix-transformations)
- [Symmetric and orthogonal matrices](#special-matrices)
- [Eigenvalues and eigenvectors](#eigenvalues)
- [Trace and determinant through eigenvalues](#trace-determinant)
- [Diagonalization](#diagonalization)
- [Spectral decomposition and SVD](#spectral-svd)
- [Independence, orthogonality, projection, and Gram–Schmidt](#projection-gram-schmidt)
- [Matrix powers](#matrix-powers)
- [Why determinant zero means non-invertible](#determinant-invertibility)
- [Perron–Frobenius theory](#perron-frobenius)

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
- $\lambda<0$: it reverses direction and is scaled by $|\lambda|$;
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

### Can a real matrix have complex eigenvalues?

Yes. A real matrix may have complex eigenvalues and eigenvectors. Nonreal eigenvalues of a real matrix occur in conjugate pairs.

For example, a two-dimensional rotation through an angle other than $0$ or $\pi$ has no real eigenvector: no real direction remains fixed.

### Doubt: are eigenvectors of a symmetric matrix always orthogonal?

The careful statement is:

> Eigenvectors of a real symmetric matrix that correspond to distinct eigenvalues are orthogonal.

Suppose

$$
Av=\lambda v,
\qquad
Aw=\mu w,
$$

with $A=A^T$ and $\lambda\ne\mu$. Then

$$
v^TAw=\mu v^Tw.
$$

But symmetry also gives

$$
v^TAw=(Av)^Tw=\lambda v^Tw.
$$

Therefore

$$
(\lambda-\mu)v^Tw=0.
$$

Since $\lambda\ne\mu$,

$$
v^Tw=0.
$$

For a repeated eigenvalue, arbitrary vectors in its eigenspace need not already be orthogonal, but we can choose an orthonormal basis for that eigenspace.

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

---

## 6. Spectral decomposition and SVD {#spectral-svd}

### Spectral decomposition

Every real symmetric matrix has an orthonormal eigenbasis. Therefore

$$
A=Q\Lambda Q^T,
$$

where $Q$ is orthogonal and $\Lambda$ is diagonal.

Equivalently,

$$
A=\sum_{i=1}^n\lambda_iq_iq_i^T.
$$

Each term acts along one orthogonal eigenvector direction.

### Singular value decomposition

The singular value decomposition applies to **every** real matrix, including rectangular and non-diagonalizable matrices:

$$
A=U\Sigma V^T.
$$

Here:

- the columns of $V$ are orthonormal input directions;
- $\Sigma$ scales those directions by nonnegative singular values;
- the columns of $U$ are orthonormal output directions.

Geometrically:

$$
\text{rotate/reflect input}
\;\longrightarrow\;
\text{scale}
\;\longrightarrow\;
\text{rotate/reflect output}.
$$

The singular values satisfy

$$
\sigma_i=\sqrt{\lambda_i(A^TA)}.
$$

### Spectral decomposition versus SVD

| Question | Spectral decomposition | SVD |
|---|---|---|
| Applies to | Symmetric/normal matrices | Every matrix |
| Matrix shape | Square | Square or rectangular |
| Diagonal values | Eigenvalues, possibly negative | Singular values, always nonnegative |
| Same input/output directions? | Eigenvector directions | Left and right singular vectors may differ |

For a symmetric PSD matrix, the two decompositions align closely because its eigenvalues are already nonnegative.

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

In the handwritten note, the question is the projection of $a$ along $q$. Substituting $v=a$ and $u=q$ gives

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

## 9. Why determinant zero means non-invertible {#determinant-invertibility}

For a square matrix, the following statements are equivalent:

$$
\det(A)\ne0,
$$

$$
A\text{ is invertible},
$$

$$
\operatorname{rank}(A)=n,
$$

$$
Ax=0\text{ has only }x=0,
$$

and the columns of $A$ are linearly independent.

### Geometric explanation

The absolute determinant is a volume-scaling factor:

- in two dimensions, $|\det(A)|$ scales area;
- in three dimensions, it scales volume.

If $\det(A)=0$, a nonzero-dimensional region is collapsed into a lower-dimensional one—a plane may collapse to a line, for example. Different inputs can then produce the same output, so the original input cannot be uniquely recovered.

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

The transformation sends the entire plane onto a line. Information is lost, so no inverse exists.

### Connection to eigenvalues

An eigenvalue $\lambda$ makes

$$
A-\lambda I
$$

singular:

$$
\det(A-\lambda I)=0.
$$

In particular, $A$ is singular exactly when $0$ is one of its eigenvalues.

---

## 10. Perron–Frobenius theory {#perron-frobenius}

Perron–Frobenius theory describes the dominant eigenstructure of nonnegative matrices.

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
- every eigenvalue satisfies $|\lambda_i|\le\rho(A)$.

Thus

$$
Av=\rho(A)v,
\qquad
v\succ0.
$$

### Important correction: irreducible versus positive

Irreducibility alone does not guarantee

$$
|\lambda_i|<\rho(A)
$$

for every other eigenvalue. An irreducible but periodic matrix may have other eigenvalues on the same spectral circle.

If $A$ is **positive**, meaning every entry is strictly positive, then the Perron eigenvalue is strictly dominant in magnitude.

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

These notes were reorganized from my public [Linear Algebra concept page](https://incongruous-donkey-948.notion.site/Linear-Algebra-eb62fdfc083043b9b869170af727e2df).
