---
title: "Symmetric Matrices, Quadratic Forms, and Matrix Norm: Why These Ideas Exist"
date: 2026-07-18 16:00:00 +0530
categories: [Mathematics, Linear Algebra]
tags: [symmetric-matrices, eigenvalues, quadratic-forms, positive-semidefinite, matrix-norm, ellipsoids, svd, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/symmetric-matrices-quadratic-forms-matrix-norm/
description: "A motivation-first guide to symmetric matrices, quadratic forms, Rayleigh quotients, positive semidefinite matrices, ellipsoids, matrix gain, and the role of SVD in understanding the matrix 2-norm."
---

A matrix can be studied from several viewpoints:

- as a **transformation** that sends $x$ to $Ax$;
- as a **quadratic measurement** that assigns the number $x^TAx$ to a direction $x$;
- as an **amplifier** whose gain in direction $x$ is $\lVert Ax\rVert_2/\lVert x\rVert_2$.

At first, symmetric matrices, quadratic forms, positive semidefinite matrices, ellipsoids, matrix norms, and singular values can look like separate topics. They are actually one story. Symmetry gives us special directions; quadratic forms measure those directions; eigenvalues bound the measurements; ellipsoids turn those bounds into geometry; and SVD extends the same directional picture to any matrix.

This article develops that story and answers the doubts that usually appear along the way.

## The map of the ideas

| Question | Concept that answers it |
|---|---|
| Can a transformation be separated into independent directions? | Spectral decomposition of a symmetric matrix |
| How can a matrix assign one scalar to a vector? | Quadratic form $x^TAx$ |
| What are the largest and smallest possible values? | Rayleigh quotient and extreme eigenvalues |
| How do we certify that the value is never negative? | Positive semidefinite matrices |
| What geometry does $x^TAx\leq 1$ describe? | Ellipsoids |
| What is the largest factor by which $A$ can stretch a vector? | Matrix $2$-norm |
| What are all the input and output stretching directions? | Singular value decomposition |

---

## 1. Why symmetric matrices are special {#symmetric-matrices}

A real square matrix is symmetric when

$$
A=A^T.
$$

This small-looking condition produces a remarkably clean geometry:

1. every eigenvalue of $A$ is real;
2. there is an orthonormal basis made of eigenvectors of $A$;
3. therefore $A$ can be diagonalized by an orthogonal change of coordinates.

The third statement is the **spectral theorem**. There is an orthogonal matrix

$$
Q=\begin{bmatrix}q_1&q_2&\cdots&q_n\end{bmatrix},
\qquad Q^TQ=I,
$$

and a real diagonal matrix

$$
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)
$$

such that

$$
A=Q\Lambda Q^T.
$$

The columns $q_i$ are orthonormal eigenvectors, and the diagonal entries $\lambda_i$ are their eigenvalues.

### Why must the eigenvalues be real?

Suppose temporarily that an eigenvector may be complex:

$$
Av=\lambda v, \qquad v\neq 0.
$$

For a real symmetric matrix, the scalar $v^*Av$ equals its own complex conjugate, so it is real. But

$$
v^*Av=\lambda v^*v.
$$

Since $v^*v>0$ is real,

$$
\lambda=\frac{v^*Av}{v^*v}
$$

must also be real. Symmetry prevents the transformation from needing complex scaling directions.

### Doubt: are all eigenvectors of a symmetric matrix orthogonal?

Not literally. The precise statement is:

> The eigenvectors of a real symmetric matrix **can be chosen** to form an orthonormal basis.

There are two reasons for this careful wording.

First, an eigenvector can be multiplied by any nonzero number, so eigenvectors are not automatically normalized. Second, if an eigenvalue is repeated, every nonzero vector in its eigenspace is an eigenvector. One may choose two nonorthogonal vectors from that same eigenspace.

The identity matrix is the simplest example. Every nonzero vector is its eigenvector, including many nonorthogonal pairs. Nevertheless, we can choose the standard orthonormal basis.

For two **distinct** eigenvalues, orthogonality is forced. If

$$
Av_i=\lambda_i v_i,
\qquad
Av_j=\lambda_j v_j,
$$

then symmetry gives

$$
v_i^TAv_j=(Av_i)^Tv_j.
$$

Consequently,

$$
\lambda_jv_i^Tv_j=\lambda_iv_i^Tv_j,
$$

or

$$
(\lambda_i-\lambda_j)v_i^Tv_j=0.
$$

When $\lambda_i\neq\lambda_j$, we must have $v_i^Tv_j=0$.

### What the spectral decomposition actually does

The equation

$$
Ax=Q\Lambda Q^Tx
$$

describes three simple operations:

1. $Q^Tx$ resolves $x$ into the eigenvector coordinates;
2. $\Lambda$ independently scales coordinate $i$ by $\lambda_i$;
3. $Q$ reconstructs the result in the original coordinates.

Equivalently,

$$
A=\sum_{i=1}^n \lambda_i q_iq_i^T,
$$

so

$$
Ax=\sum_{i=1}^n \lambda_i q_i(q_i^Tx).
$$

Here $q_iq_i^Tx$ is the orthogonal projection of $x$ onto the line spanned by $q_i$. The matrix separates $x$ into mutually perpendicular components, scales each component, and adds them back together. A negative eigenvalue also reverses its component's direction.

This is why symmetric matrices are so manageable: there are no interactions between the coordinates after we move to the eigenvector basis.

### A two-dimensional example

Consider

$$
A=
\begin{bmatrix}
-\tfrac12 & \tfrac32\\
\tfrac32 & -\tfrac12
\end{bmatrix}.
$$

Its orthonormal eigenvectors and eigenvalues are

$$
q_1=\frac{1}{\sqrt2}
\begin{bmatrix}1\\1\end{bmatrix},
\quad \lambda_1=1,
\qquad
q_2=\frac{1}{\sqrt2}
\begin{bmatrix}1\\-1\end{bmatrix},
\quad \lambda_2=-2.
$$

Thus

$$
Ax=q_1(q_1^Tx)-2q_2(q_2^Tx).
$$

The component along the diagonal line $q_1$ is unchanged. The component along the perpendicular diagonal $q_2$ is doubled and reversed. This geometric description says much more than multiplying entries mechanically.

### Why these matrices appear so often

Symmetric matrices arise whenever a quantity treats pairs of coordinates without an orientation:

- a Gram matrix $B^TB$ records inner products;
- a covariance matrix records how variables vary together;
- the Hessian of a sufficiently smooth scalar function records local curvature;
- graph Laplacians encode pairwise differences across edges.

The common benefit is the same: orthogonal directions in which the system decouples.

---

## 2. Quadratic forms: turning a direction into a number {#quadratic-forms}

A quadratic form is a scalar-valued function

$$
f(x)=x^TAx.
$$

Why would we want a scalar instead of the transformed vector $Ax$? Because many questions ask for a **size, energy, cost, curvature, or alignment**, not another vector.

For example,

$$
\lVert Bx\rVert_2^2
=(Bx)^T(Bx)
=x^TB^TBx.
$$

So every squared residual or squared transformed length is a quadratic form.

Another useful example is a roughness penalty:

$$
\sum_{i=1}^{n-1}(x_{i+1}-x_i)^2.
$$

It is small when adjacent entries of $x$ are similar and large when $x$ changes rapidly. If a difference matrix $D$ satisfies $(Dx)_i=x_{i+1}-x_i$, then this penalty is

$$
\lVert Dx\rVert_2^2=x^TD^TDx.
$$

This is the mathematical core of many smoothing and regularization methods.

### Doubt: must $A$ be symmetric in $x^TAx$?

We lose nothing by assuming symmetry. Decompose any square matrix as

$$
A=\frac{A+A^T}{2}+\frac{A-A^T}{2}.
$$

The first part is symmetric and the second is skew-symmetric. If

$$
K=\frac{A-A^T}{2},
\qquad K^T=-K,
$$

then the scalar $x^TKx$ satisfies

$$
x^TKx=(x^TKx)^T=x^TK^Tx=-x^TKx.
$$

Therefore $x^TKx=0$, and

$$
x^TAx=x^T\left(\frac{A+A^T}{2}\right)x.
$$

The quadratic form sees only the symmetric part of a matrix.

### Quadratic forms in eigenvector coordinates

If $A=Q\Lambda Q^T$ and $z=Q^Tx$, then

$$
x^TAx=z^T\Lambda z
=\sum_{i=1}^n\lambda_i z_i^2
=\sum_{i=1}^n\lambda_i(q_i^Tx)^2.
$$

This formula is the bridge to everything that follows. It says that a quadratic form is a weighted sum of squared components along the eigenvector directions.

---

## 3. Rayleigh quotient: which direction matters most? {#rayleigh-quotient}

The value $x^TAx$ grows when we scale $x$, so it cannot compare directions fairly. If $x$ is replaced by $10x$, the value becomes $100x^TAx$.

The **Rayleigh quotient** removes that dependence on length:

$$
R_A(x)=\frac{x^TAx}{x^Tx},
\qquad x\neq 0.
$$

Write $x=\sum_i\alpha_iq_i$. Then

$$
R_A(x)
=\frac{\sum_i\lambda_i\alpha_i^2}
       {\sum_i\alpha_i^2}.
$$

The coefficients $\alpha_i^2/\sum_j\alpha_j^2$ are nonnegative and add to one. Therefore the Rayleigh quotient is a weighted average of the eigenvalues. If

$$
\lambda_{max}=\lambda_1
\geq \cdots \geq
\lambda_n=\lambda_{min},
$$

then

$$
\lambda_{\min}
\leq R_A(x)
\leq \lambda_{\max}.
$$

Equivalently,

$$
\lambda_{\min}\lVert x\rVert_2^2
\leq x^TAx
\leq \lambda_{\max}\lVert x\rVert_2^2.
$$

These bounds are exact:

$$
\max_{\lVert x\rVert_2=1}x^TAx=\lambda_{\max},
\qquad
\min_{\lVert x\rVert_2=1}x^TAx=\lambda_{\min}.
$$

The maximizing and minimizing directions are corresponding eigenvectors.

### Why this viewpoint exists

The Rayleigh quotient converts an eigenvalue problem into an optimization problem. It answers questions such as:

- In which direction is a quadratic energy largest?
- What is the greatest curvature of a quadratic objective?
- Which unit direction produces the largest transformed length after we replace $A$ by $A^TA$?

That last question will lead directly to the matrix norm.

---

## 4. Positive semidefinite matrices: certifying nonnegative quadratic behavior {#positive-semidefinite}

A symmetric matrix is **positive semidefinite** (PSD) if

$$
x^TAx\geq 0
\qquad\text{for every }x.
$$

We write $A\succeq 0$. It is **positive definite** (PD), written $A\succ 0$, if

$$
x^TAx>0
\qquad\text{for every }x\neq 0.
$$

From the eigenvector-coordinate formula,

$$
x^TAx=\sum_i\lambda_i(q_i^Tx)^2,
$$

we immediately obtain

$$
A\succeq0
\quad\Longleftrightarrow\quad
\lambda_i\geq0\text{ for every }i,
$$

and

$$
A\succ0
\quad\Longleftrightarrow\quad
\lambda_i>0\text{ for every }i.
$$

### Doubt: does “positive matrix” mean all entries are positive?

No. Positive semidefinite describes the quadratic form, not the individual entries.

For example,

$$
\begin{bmatrix}
1&-1\\
-1&1
\end{bmatrix}
$$

contains negative entries, but

$$
\begin{bmatrix}x_1&x_2\end{bmatrix}
\begin{bmatrix}1&-1\\-1&1\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}
=(x_1-x_2)^2\geq0.
$$

So the matrix is PSD.

### Where PSD matrices come from

The most important construction is

$$
B^TB\succeq0,
$$

because

$$
x^TB^TBx=\lVert Bx\rVert_2^2\geq0.
$$

This explains why PSD matrices appear in many places:

- **least squares:** the Hessian of $\lVert Bx-b\rVert_2^2$ is a multiple of $B^TB$;
- **covariance:** $z^T\Sigma z$ is the variance in direction $z$;
- **convex optimization:** a twice-differentiable function is convex when its Hessian is PSD everywhere;
- **energy models:** a PSD quadratic form cannot assign negative energy.

### Comparing quadratic forms: the Loewner order

For symmetric matrices, define

$$
A\succeq B
\quad\Longleftrightarrow\quad
A-B\succeq0.
$$

This means

$$
x^TAx\geq x^TBx
\qquad\text{for every }x.
$$

It is a comparison of the two matrices in **all directions at once**.

This is only a partial order. Two matrices need not be comparable. For example,

$$
A=\begin{bmatrix}2&0\\0&0\end{bmatrix},
\qquad
B=\begin{bmatrix}0&0\\0&2\end{bmatrix}
$$

are stronger in different directions. The difference $A-B$ has one positive and one negative eigenvalue, so neither $A\succeq B$ nor $B\succeq A$ holds.

---

## 5. Ellipsoids: the geometry of a positive quadratic form {#ellipsoids}

If $A\succ0$, consider the set

$$
\mathcal E_A=\{x\mid x^TAx\leq1\}.
$$

Using $A=Q\Lambda Q^T$ and $z=Q^Tx$, its boundary satisfies

$$
\sum_{i=1}^n\lambda_i z_i^2=1.
$$

This is an ellipsoid. Its principal axes point along the eigenvectors $q_i$, and the semi-axis in direction $q_i$ has length

$$
\frac{1}{\sqrt{\lambda_i}}.
$$

This inverse relationship is worth pausing over:

- a large eigenvalue makes $x^TAx$ grow quickly, so the ellipsoid is **thin** in that direction;
- a small eigenvalue makes the form grow slowly, so the ellipsoid is **wide** in that direction.

The ratio between the longest and shortest semi-axes is

$$
\sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}.
$$

Thus the eigenvalue spread measures how elongated the geometry is.

### Matrix order becomes set containment

Suppose $A\succ0$ and $B\succ0$. Then

$$
A\succeq B
\quad\Longrightarrow\quad
\mathcal E_A\subseteq\mathcal E_B.
$$

Indeed, if $x^TAx\leq1$, then $x^TBx\leq x^TAx\leq1$. The apparently larger matrix produces the smaller ellipsoid because its quadratic penalty is stronger.

### Why ellipsoids matter

Ellipsoids are not decorative pictures of quadratic forms. They represent:

- level sets of quadratic objectives;
- anisotropic distance, where movement costs more in some directions;
- uncertainty regions, often with an inverse covariance matrix defining the quadratic form;
- local approximations of smooth optimization problems;
- constraints that limit a weighted combination of variables.

The eigenvectors identify the important directions, and the eigenvalues determine how restrictive each direction is.

---

## 6. Matrix norm: the largest possible gain {#matrix-norm}

For a vector, $\lVert x\rVert_2$ measures size. For a matrix, the most useful question is often operational:

> By what largest factor can this transformation amplify an input?

For $A\in\mathbb R^{m\times n}$, define the induced matrix $2$-norm

$$
\lVert A\rVert_2
=\max_{x\neq0}\frac{\lVert Ax\rVert_2}{\lVert x\rVert_2}
=\max_{\lVert x\rVert_2=1}\lVert Ax\rVert_2.
$$

This definition applies to rectangular and nonsymmetric matrices. To connect it to the symmetric theory, square the gain:

$$
\left(\frac{\lVert Ax\rVert_2}{\lVert x\rVert_2}\right)^2
=\frac{x^TA^TAx}{x^Tx}.
$$

The matrix $A^TA$ is always symmetric and PSD. The right-hand side is its Rayleigh quotient, so

$$
\boxed{\lVert A\rVert_2=\sqrt{\lambda_{\max}(A^TA)}}.
$$

This formula is not an unrelated trick. We use $A^TA$ because it converts output length into a quadratic form on the input space.

Similarly, the smallest directional gain is

$$
\min_{x\neq0}\frac{\lVert Ax\rVert_2}{\lVert x\rVert_2}
=\sqrt{\lambda_{\min}(A^TA)}.
$$

It is zero when $A$ collapses some nonzero input to zero.

### A complete numerical example

Let

$$
A=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix}.
$$

Then

$$
A^TA=
\begin{bmatrix}
35&44\\
44&56
\end{bmatrix}.
$$

Its eigenvalues are approximately

$$
90.7
\qquad\text{and}\qquad
0.265.
$$

Therefore

$$
\lVert A\rVert_2\approx\sqrt{90.7}=9.53,
$$

and the minimum gain is

$$
\sqrt{0.265}\approx0.514.
$$

The corresponding unit input directions are approximately

$$
v_{\max}=
\begin{bmatrix}0.620\\0.785\end{bmatrix},
\qquad
v_{\min}=
\begin{bmatrix}0.785\\-0.620\end{bmatrix}.
$$

So every nonzero input satisfies

$$
0.514
\leq
\frac{\lVert Ax\rVert_2}{\lVert x\rVert_2}
\leq
9.53.
$$

This is a much stronger statement than saying that the entries of $A$ are “large” or “small.” It identifies the best- and worst-case directions of the transformation.

### Why the norm is useful

From its definition,

$$
\lVert Ax\rVert_2\leq\lVert A\rVert_2\lVert x\rVert_2.
$$

Thus an input perturbation $\delta x$ produces an output perturbation bounded by

$$
\lVert A\delta x\rVert_2
\leq
\lVert A\rVert_2\lVert\delta x\rVert_2.
$$

For two compatible transformations,

$$
\lVert BA\rVert_2
\leq
\lVert B\rVert_2\lVert A\rVert_2.
$$

So matrix norms let us propagate error and gain bounds through a chain of operations without knowing the exact input direction.

---

## 7. SVD as the full directional explanation of matrix norm {#svd-and-norm}

The singular value decomposition is included here for one reason: it completes the geometric meaning of matrix gain.

For a matrix $A\in\mathbb R^{m\times n}$,

$$
A=U\Sigma V^T.
$$

The columns $v_i$ of $V$ are orthonormal **input directions**, the columns $u_i$ of $U$ are orthonormal **output directions**, and the nonnegative numbers $\sigma_i$ are the singular values. In these directions,

$$
Av_i=\sigma_i u_i.
$$

Thus $A$ performs three conceptual steps:

1. resolve the input along the directions $v_i$;
2. scale component $i$ by $\sigma_i$;
3. reconstruct the output along directions $u_i$.

The link with the previous section is

$$
A^TA=V\Sigma^2V^T.
$$

Therefore

$$
\sigma_i=\sqrt{\lambda_i(A^TA)},
$$

and in particular,

$$
\boxed{\lVert A\rVert_2=\sigma_1}.
$$

The matrix $2$-norm is the largest singular value because $v_1$ is precisely the unit input direction that receives the greatest amplification.

### How this differs from symmetric eigendecomposition

For a symmetric matrix, the input and output eigenvector directions are the same basis. For a general rectangular matrix they may live in different spaces, so SVD needs both $v_i$ and $u_i$.

This is why SVD, rather than ordinary eigendecomposition, is the natural tool for the gain of an arbitrary matrix.

### Small singular values and effective rank

Suppose a matrix has singular values

$$
10,\quad 7,\quad 0.1,\quad 0.05.
$$

Formally, all four directions are present, so the matrix has rank four. But the last two directions are attenuated so strongly that, relative to the first two, the transformation behaves approximately like a rank-two map.

This is the basis of low-rank approximation and compression: keep the directions that carry substantial gain and discard directions whose singular values are negligible for the application.

The largest-to-smallest gain ratio here is

$$
\frac{10}{0.05}=200.
$$

Such a large ratio also signals sensitivity: recovering an input from the output can greatly amplify errors in weak directions.

---

## 8. The whole story in one chain {#summary}

The concepts now fit together:

1. **Symmetry** gives an orthonormal eigenvector basis and real eigenvalues.
2. **Spectral decomposition** separates a transformation into independent orthogonal directions.
3. A **quadratic form** converts directional behavior into a scalar energy, cost, or curvature.
4. The **Rayleigh quotient** removes vector length and reveals that extreme directional values are eigenvalues.
5. **Positive semidefiniteness** says the quadratic value is nonnegative in every direction.
6. A positive definite quadratic form creates an **ellipsoid** whose directions and widths come from eigenvectors and eigenvalues.
7. Applying the Rayleigh quotient to $A^TA$ gives the **matrix $2$-norm**, the largest directional gain.
8. **SVD** names every input direction, output direction, and gain; its largest singular value is the matrix norm.

The central mental model is:

> Find the right orthogonal coordinates, understand what the matrix does independently in each coordinate, and then reconstruct the global behavior.

That single idea explains the algebra, geometry, optimization meaning, and sensitivity analysis behind the entire lecture.

## Sources and further study

- Stephen Boyd, [EE263 Lecture 15 slides: *Symmetric Matrices, Quadratic Forms, Matrix Norm, and SVD*](https://see.stanford.edu/materials/lsoeldsee263/15-symm.pdf).
- Stephen Boyd, [EE263 Lecture 15 transcript](https://see.stanford.edu/materials/lsoeldsee263/transcripts/IntroToLinearDynamicalSystems-Lecture15.pdf).
- Stanford Engineering Everywhere, [EE263: Introduction to Linear Dynamical Systems](https://see.stanford.edu/Course/EE263).

The exposition here is an original synthesis of the lecture slides and transcript, with the circuit example omitted and SVD kept specifically to explain matrix norm and directional gain.
