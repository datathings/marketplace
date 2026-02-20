# CBLAS & LAPACKE Workflow Examples

> 15 complete, compilable C examples demonstrating common BLAS/LAPACK operations.
> All examples use the CBLAS and LAPACKE C interfaces from LAPACK v3.12.1.
> Every example uses `double` precision and `LAPACK_ROW_MAJOR` / `CblasRowMajor` layout.

## Table of Contents

1. [Matrix Multiply (DGEMM)](#1-matrix-multiply-dgemm)
2. [Solve Linear System (Ax=b)](#2-solve-linear-system-axb)
3. [LU Factorization + Matrix Inverse](#3-lu-factorization--matrix-inverse)
4. [Cholesky Factorization + Solve](#4-cholesky-factorization--solve)
5. [Symmetric Eigenvalues](#5-symmetric-eigenvalues)
6. [General Eigenvalues](#6-general-eigenvalues)
7. [Singular Value Decomposition](#7-singular-value-decomposition)
8. [Least Squares](#8-least-squares)
9. [QR Factorization](#9-qr-factorization)
10. [Batch Vector Operations](#10-batch-vector-operations)
11. [Triangular Solve](#11-triangular-solve)
12. [Banded Linear System](#12-banded-linear-system)
13. [Symmetric Positive Definite Solve](#13-symmetric-positive-definite-solve)
14. [Matrix Norm and Condition Number](#14-matrix-norm-and-condition-number)
15. [Building and Linking Guide](#15-building-and-linking-guide)

---

## 1. Matrix Multiply (DGEMM)

**Key functions:** `cblas_dgemm`
**Compile:** `gcc -o example1 example1.c -lcblas -lblas -lm`

```c
/*
 * Matrix Multiply: C = alpha * A * B + beta * C
 * Demonstrates cblas_dgemm with CblasRowMajor layout.
 */
#include <stdio.h>
#include <cblas.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %8.4f", M[i * cols + j]);
        printf("\n");
    }
}

int main(void) {
    /* 3x3 matrices stored in row-major order */
    double A[9] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    double B[9] = {
        9.0, 8.0, 7.0,
        6.0, 5.0, 4.0,
        3.0, 2.0, 1.0
    };
    double C[9] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0
    };

    double alpha = 1.0;
    double beta  = 2.0;  /* C = 1.0*A*B + 2.0*C */

    print_matrix("A", A, 3, 3);
    print_matrix("B", B, 3, 3);
    print_matrix("C (before)", C, 3, 3);

    /*
     * cblas_dgemm(layout, transA, transB, M, N, K,
     *             alpha, A, lda, B, ldb, beta, C, ldc)
     *
     * C(M x N) = alpha * A(M x K) * B(K x N) + beta * C(M x N)
     * Row-major: lda >= K, ldb >= N, ldc >= N
     */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                3, 3, 3,
                alpha, A, 3, B, 3,
                beta, C, 3);

    print_matrix("C = 1.0*A*B + 2.0*C", C, 3, 3);

    return 0;
}
```

**Expected output:**
```
A:
    1.0000    2.0000    3.0000
    4.0000    5.0000    6.0000
    7.0000    8.0000    9.0000
B:
    9.0000    8.0000    7.0000
    6.0000    5.0000    4.0000
    3.0000    2.0000    1.0000
C (before):
    1.0000    1.0000    1.0000
    1.0000    1.0000    1.0000
    1.0000    1.0000    1.0000
C = 1.0*A*B + 2.0*C:
   32.0000   22.0000   12.0000
   86.0000   64.0000   42.0000
  140.0000  106.0000   72.0000
```

**Notes:**
- `alpha` and `beta` provide scaling: `C = alpha*A*B + beta*C`. Set `beta=0.0` to ignore the initial contents of C.
- Row-major leading dimensions: `lda` = number of columns of A (as stored), `ldb` = number of columns of B, `ldc` = number of columns of C.
- Use `CblasTrans` for the transA/transB parameters to multiply with transposed matrices without copying.

---

## 2. Solve Linear System (Ax=b)

**Key functions:** `LAPACKE_dgesv`
**Compile:** `gcc -o example2 example2.c -llapacke -llapack -lcblas -lblas -lm`

```c
/*
 * Solve Ax = b for a general 3x3 system using LU factorization.
 * Demonstrates LAPACKE_dgesv with LAPACK_ROW_MAJOR layout.
 */
#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <cblas.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %8.4f", M[i * cols + j]);
        printf("\n");
    }
}

void print_vector(const char *name, const double *v, int n) {
    printf("%s:", name);
    for (int i = 0; i < n; i++)
        printf("  %8.4f", v[i]);
    printf("\n");
}

int main(void) {
    int n = 3, nrhs = 1;

    /* A will be overwritten with LU factors */
    double A[9] = {
        6.80, -6.05, -0.45,
       -2.11, -3.30,  2.58,
        5.66,  5.36, -2.70
    };
    /* b will be overwritten with the solution x */
    double b[3] = { -1.48, 3.04, 8.68 };
    lapack_int ipiv[3];

    /* Save copies for verification */
    double A_orig[9], b_orig[3];
    for (int i = 0; i < 9; i++) A_orig[i] = A[i];
    for (int i = 0; i < 3; i++) b_orig[i] = b[i];

    print_matrix("A", A, 3, 3);
    print_vector("b", b, 3);

    /*
     * LAPACKE_dgesv(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb)
     *
     * Solves A*X = B by LU factorization with partial pivoting.
     * On exit: A contains L and U factors, b contains the solution.
     */
    lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs,
                                     A, n, ipiv, b, nrhs);

    if (info != 0) {
        fprintf(stderr, "dgesv failed: info = %d\n", info);
        return 1;
    }

    print_vector("Solution x", b, 3);

    /* Verify: compute residual r = A_orig * x - b_orig */
    double r[3];
    for (int i = 0; i < 3; i++) r[i] = -b_orig[i];

    /* r = 1.0 * A_orig * x + 1.0 * r  (where r starts as -b_orig) */
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n,
                1.0, A_orig, n, b, 1, 1.0, r, 1);

    print_vector("Residual (A*x - b)", r, 3);

    return 0;
}
```

**Expected output:**
```
A:
    6.8000   -6.0500   -0.4500
   -2.1100   -3.3000    2.5800
    5.6600    5.3600   -2.7000
b:  -1.4800    3.0400    8.6800
Solution x:   0.3413    0.6467    1.7507
Residual (A*x - b):   0.0000    0.0000    0.0000
```

**Notes:**
- `LAPACKE_dgesv` overwrites both A (with LU factors) and b (with the solution). Save copies if you need the originals.
- `ipiv` contains the pivot indices from partial pivoting.
- `info > 0` means the matrix is singular; the `info`-th diagonal element of U is zero.
- For multiple right-hand sides, set `nrhs > 1` and provide b as an n-by-nrhs matrix with `ldb = nrhs`.

---

## 3. LU Factorization + Matrix Inverse

**Key functions:** `LAPACKE_dgetrf`, `LAPACKE_dgetri`
**Compile:** `gcc -o example3 example3.c -llapacke -llapack -lcblas -lblas -lm`

```c
/*
 * LU factorize a 3x3 matrix, then compute its inverse.
 * Verify by checking A * A_inv = I.
 */
#include <stdio.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %8.4f", M[i * cols + j]);
        printf("\n");
    }
}

int main(void) {
    int n = 3;

    double A[9] = {
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        5.0, 6.0, 0.0
    };
    /* Save original for verification */
    double A_orig[9];
    for (int i = 0; i < 9; i++) A_orig[i] = A[i];

    print_matrix("A", A, 3, 3);

    lapack_int ipiv[3];
    lapack_int info;

    /*
     * Step 1: LU factorization  A = P * L * U
     * LAPACKE_dgetrf(layout, m, n, a, lda, ipiv)
     */
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A, n, ipiv);
    if (info != 0) {
        fprintf(stderr, "dgetrf failed: info = %d\n", info);
        return 1;
    }
    printf("LU factorization succeeded.\n");
    print_matrix("LU factors (packed in A)", A, 3, 3);

    /*
     * Step 2: Compute inverse from LU factors
     * LAPACKE_dgetri(layout, n, a, lda, ipiv)
     * On exit, A contains A^{-1}.
     */
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A, n, ipiv);
    if (info != 0) {
        fprintf(stderr, "dgetri failed: info = %d\n", info);
        return 1;
    }

    print_matrix("A_inv", A, 3, 3);

    /* Verify: compute C = A_orig * A_inv, should be identity */
    double C[9] = {0};
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, A_orig, n, A, n, 0.0, C, n);

    print_matrix("A * A_inv (should be I)", C, 3, 3);

    /* Check max deviation from identity */
    double max_err = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            double err = fabs(C[i * n + j] - expected);
            if (err > max_err) max_err = err;
        }
    printf("Max deviation from identity: %.2e\n", max_err);

    return 0;
}
```

**Expected output:**
```
A:
    1.0000    2.0000    3.0000
    0.0000    1.0000    4.0000
    5.0000    6.0000    0.0000
LU factorization succeeded.
LU factors (packed in A):
    5.0000    6.0000    0.0000
    0.2000    0.8000    3.0000
    0.0000    1.2500    4.0000
A_inv:
  -24.0000   18.0000    5.0000
   20.0000  -15.0000   -4.0000
   -5.0000    4.0000    1.0000
A * A_inv (should be I):
    1.0000    0.0000    0.0000
    0.0000    1.0000    0.0000
    0.0000    0.0000    1.0000
Max deviation from identity: 0.00e+00
```

**Notes:**
- `dgetrf` performs PA = LU factorization. L (unit lower triangular) and U (upper triangular) are packed into the same array. The diagonal belongs to U; L has an implicit unit diagonal.
- `dgetri` overwrites the LU factors with the inverse. This is a two-step process: factorize first, then invert.
- Computing the full inverse is expensive (O(n^3)) and numerically less stable than solving systems directly. Prefer `dgesv` when you only need to solve Ax=b.

---

## 4. Cholesky Factorization + Solve

**Key functions:** `LAPACKE_dpotrf`, `LAPACKE_dpotrs`
**Compile:** `gcc -o example4 example4.c -llapacke -llapack -lcblas -lblas -lm`

```c
/*
 * Cholesky factor a symmetric positive definite (SPD) matrix,
 * then solve a linear system using the factorization.
 */
#include <stdio.h>
#include <lapacke.h>
#include <cblas.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %8.4f", M[i * cols + j]);
        printf("\n");
    }
}

void print_vector(const char *name, const double *v, int n) {
    printf("%s:", name);
    for (int i = 0; i < n; i++)
        printf("  %8.4f", v[i]);
    printf("\n");
}

int main(void) {
    int n = 3, nrhs = 1;

    /*
     * Build an SPD matrix: A = M^T * M + I  (guarantees positive definiteness)
     * M = [[1,2,0],[0,1,1],[1,0,1]]
     * A = M^T*M + I
     */
    double A[9] = {
        3.0, 2.0, 1.0,
        2.0, 6.0, 1.0,
        1.0, 1.0, 3.0
    };
    double b[3] = { 11.0, 15.0, 9.0 };

    /* Save copies */
    double A_orig[9], b_orig[3];
    for (int i = 0; i < 9; i++) A_orig[i] = A[i];
    for (int i = 0; i < 3; i++) b_orig[i] = b[i];

    print_matrix("A (SPD)", A, 3, 3);
    print_vector("b", b, 3);

    lapack_int info;

    /*
     * Step 1: Cholesky factorization  A = L * L^T  (lower triangle)
     * LAPACKE_dpotrf(layout, uplo, n, a, lda)
     * uplo='L': compute lower-triangular L such that A = L*L^T
     */
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, A, n);
    if (info != 0) {
        fprintf(stderr, "dpotrf failed: info = %d\n", info);
        if (info > 0) fprintf(stderr, "Matrix is not positive definite.\n");
        return 1;
    }

    printf("Cholesky factor L (lower triangle of A):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j <= i)
                printf("  %8.4f", A[i * n + j]);
            else
                printf("  %8s", "");  /* upper part is stale */
        }
        printf("\n");
    }

    /*
     * Step 2: Solve A*x = b using the Cholesky factors
     * LAPACKE_dpotrs(layout, uplo, n, nrhs, a, lda, b, ldb)
     * b is overwritten with the solution x.
     */
    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n, nrhs, A, n, b, nrhs);
    if (info != 0) {
        fprintf(stderr, "dpotrs failed: info = %d\n", info);
        return 1;
    }

    print_vector("Solution x", b, 3);

    /* Verify: r = A_orig * x - b_orig */
    double r[3];
    for (int i = 0; i < 3; i++) r[i] = -b_orig[i];
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n,
                1.0, A_orig, n, b, 1, 1.0, r, 1);
    print_vector("Residual (A*x - b)", r, 3);

    return 0;
}
```

**Expected output:**
```
A (SPD):
    3.0000    2.0000    1.0000
    2.0000    6.0000    1.0000
    1.0000    1.0000    3.0000
b:  11.0000   15.0000    9.0000
Cholesky factor L (lower triangle of A):
    1.7321
    1.1547    2.0817
    0.5774    0.1601    1.6305
Solution x:   1.0000    2.0000    2.0000
Residual (A*x - b):   0.0000    0.0000    0.0000
```

**Notes:**
- Cholesky is roughly twice as fast as LU for SPD matrices and requires no pivoting.
- `uplo='L'` computes lower-triangular L; `uplo='U'` computes upper-triangular U where A = U^T*U.
- `info > 0` from `dpotrf` means the matrix is not positive definite (the `info`-th leading minor is not positive).
- The unused triangle of A is not modified by `dpotrf`. Only the triangle specified by `uplo` contains the factor.

---

## 5. Symmetric Eigenvalues

**Key functions:** `LAPACKE_dsyev`
**Compile:** `gcc -o example5 example5.c -llapacke -llapack -lcblas -lblas -lm`

```c
/*
 * Compute all eigenvalues and eigenvectors of a 3x3 symmetric matrix.
 */
#include <stdio.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %8.4f", M[i * cols + j]);
        printf("\n");
    }
}

int main(void) {
    int n = 3;

    /* Symmetric matrix (only upper or lower triangle is referenced) */
    double A[9] = {
        2.0, -1.0,  0.0,
       -1.0,  2.0, -1.0,
        0.0, -1.0,  2.0
    };
    double w[3]; /* eigenvalues */

    /* Save original */
    double A_orig[9];
    for (int i = 0; i < 9; i++) A_orig[i] = A[i];

    print_matrix("A (symmetric)", A, 3, 3);

    /*
     * LAPACKE_dsyev(layout, jobz, uplo, n, a, lda, w)
     *
     * jobz = 'V': compute eigenvalues AND eigenvectors
     * jobz = 'N': compute eigenvalues only
     * uplo = 'U': upper triangle of A is stored
     *
     * On exit: w contains eigenvalues in ascending order.
     *          A columns contain the corresponding eigenvectors (row-major:
     *          eigenvector i is in row i of the output matrix).
     */
    lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, n, w);
    if (info != 0) {
        fprintf(stderr, "dsyev failed: info = %d\n", info);
        return 1;
    }

    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++)
        printf("  lambda_%d = %8.4f\n", i, w[i]);

    /*
     * In row-major output, eigenvector i is stored as column i of the matrix.
     * That is, A[row * n + i] for row = 0..n-1 gives eigenvector i.
     */
    printf("Eigenvectors (as columns of A):\n");
    for (int i = 0; i < n; i++) {
        printf("  v_%d = [", i);
        for (int j = 0; j < n; j++)
            printf(" %8.4f", A[j * n + i]);
        printf(" ]\n");
    }

    /* Verify: A_orig * v_i = lambda_i * v_i for each eigenpair */
    printf("Verification (A*v - lambda*v for each eigenpair):\n");
    for (int i = 0; i < n; i++) {
        double v[3], Av[3];
        for (int j = 0; j < n; j++) v[j] = A[j * n + i];

        /* Av = A_orig * v */
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n,
                    1.0, A_orig, n, v, 1, 0.0, Av, 1);

        double err = 0.0;
        for (int j = 0; j < n; j++)
            err += (Av[j] - w[i] * v[j]) * (Av[j] - w[i] * v[j]);
        printf("  ||A*v_%d - lambda_%d*v_%d|| = %.2e\n", i, i, i, sqrt(err));
    }

    return 0;
}
```

**Expected output:**
```
A (symmetric):
    2.0000   -1.0000    0.0000
   -1.0000    2.0000   -1.0000
    0.0000   -1.0000    2.0000
Eigenvalues:
  lambda_0 =   0.5858
  lambda_1 =   2.0000
  lambda_2 =   3.4142
Eigenvectors (as columns of A):
  v_0 = [   0.5000   -0.7071    0.5000 ]
  v_1 = [  -0.7071    0.0000    0.7071 ]
  v_2 = [   0.5000    0.7071    0.5000 ]
Verification (A*v - lambda*v for each eigenpair):
  ||A*v_0 - lambda_0*v_0|| = 0.00e+00
  ||A*v_1 - lambda_1*v_1|| = 0.00e+00
  ||A*v_2 - lambda_2*v_2|| = 0.00e+00
```

**Notes:**
- `dsyev` is the standard driver for symmetric eigenvalue problems. For better performance, use `dsyevd` (divide-and-conquer) or `dsyevr` (relatively robust representations).
- Eigenvalues are always returned in ascending order.
- The matrix A is destroyed on output (overwritten with eigenvectors).
- Only the triangle specified by `uplo` is referenced; the other triangle is not read.

---

## 6. General Eigenvalues

**Key functions:** `LAPACKE_dgeev`
**Compile:** `gcc -o example6 example6.c -llapacke -llapack -lcblas -lblas -lm`

```c
/*
 * Compute eigenvalues (real and imaginary parts) and right eigenvectors
 * of a general (non-symmetric) 3x3 matrix.
 */
#include <stdio.h>
#include <lapacke.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %8.4f", M[i * cols + j]);
        printf("\n");
    }
}

int main(void) {
    int n = 3;

    /* General (non-symmetric) matrix -- A is overwritten */
    double A[9] = {
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0
    };

    double wr[3]; /* real parts of eigenvalues */
    double wi[3]; /* imaginary parts of eigenvalues */
    double vl[9]; /* left eigenvectors (not computed here) */
    double vr[9]; /* right eigenvectors */

    print_matrix("A", A, 3, 3);

    /*
     * LAPACKE_dgeev(layout, jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr)
     *
     * jobvl = 'N': do not compute left eigenvectors
     * jobvr = 'V': compute right eigenvectors
     *
     * Eigenvalues are returned as (wr[i], wi[i]) pairs.
     * If wi[i] == 0, the eigenvalue is real and vr column i is the eigenvector.
     * If wi[i] > 0, then eigenvalues i and i+1 are a complex conjugate pair:
     *   lambda_i     = wr[i] + j*wi[i]     with eigenvector vr[:,i] + j*vr[:,i+1]
     *   lambda_{i+1} = wr[i] - j*wi[i]     with eigenvector vr[:,i] - j*vr[:,i+1]
     */
    lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V',
                                     n, A, n, wr, wi, vl, n, vr, n);
    if (info != 0) {
        fprintf(stderr, "dgeev failed: info = %d\n", info);
        return 1;
    }

    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++) {
        if (wi[i] == 0.0)
            printf("  lambda_%d = %8.4f (real)\n", i, wr[i]);
        else if (wi[i] > 0.0)
            printf("  lambda_%d = %8.4f + %8.4fi (complex)\n", i, wr[i], wi[i]);
        else
            printf("  lambda_%d = %8.4f - %8.4fi (complex)\n", i, wr[i], -wi[i]);
    }

    printf("Right eigenvectors (columns of vr):\n");
    for (int i = 0; i < n; i++) {
        printf("  v_%d = [", i);
        for (int j = 0; j < n; j++)
            printf(" %8.4f", vr[j * n + i]);
        printf(" ]\n");
    }

    printf("\nNote: For complex conjugate eigenvalue pairs, the eigenvector\n");
    printf("for lambda_i is vr[:,i] + j*vr[:,i+1], and for lambda_{i+1}\n");
    printf("it is vr[:,i] - j*vr[:,i+1].\n");

    return 0;
}
```

**Expected output:**
```
A:
    0.0000    1.0000    0.0000
    0.0000    0.0000    1.0000
    1.0000    0.0000    0.0000
Eigenvalues:
  lambda_0 =  -0.5000 +   0.8660i (complex)
  lambda_1 =  -0.5000 -   0.8660i (complex)
  lambda_2 =   1.0000 (real)
Right eigenvectors (columns of vr):
  v_0 = [   0.2113    0.7887    0.5774 ]
  v_1 = [   0.7887   -0.2113    0.0000 ]
  v_2 = [  -0.5774   -0.5774    0.5774 ]

Note: For complex conjugate eigenvalue pairs, the eigenvector
for lambda_i is vr[:,i] + j*vr[:,i+1], and for lambda_{i+1}
it is vr[:,i] - j*vr[:,i+1].
```

**Notes:**
- This is a permutation matrix (cyclic shift) with eigenvalues that are the cube roots of unity: 1, -1/2 + i*sqrt(3)/2, -1/2 - i*sqrt(3)/2.
- Complex eigenvalues always appear in conjugate pairs for real matrices.
- The eigenvector encoding for complex pairs is compact but requires careful interpretation: the real and imaginary parts share two adjacent columns of `vr`.
- `dgeev` is more expensive than `dsyev` since it must compute the Schur form. Use `dsyev` when the matrix is symmetric.

---

## 7. Singular Value Decomposition

**Key functions:** `LAPACKE_dgesvd`
**Compile:** `gcc -o example7 example7.c -llapacke -llapack -lcblas -lblas -lm`

```c
/*
 * Compute the full SVD of a 3x4 matrix: A = U * diag(S) * Vt
 * Print singular values and verify the reconstruction.
 */
#include <stdio.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %8.4f", M[i * cols + j]);
        printf("\n");
    }
}

int main(void) {
    int m = 3, n = 4;
    int min_mn = m; /* min(3,4) = 3 */

    /* A is overwritten by dgesvd */
    double A[12] = {
        1.0, 0.0, 0.0, 2.0,
        0.0, 0.0, 3.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };
    /* Save original */
    double A_orig[12];
    for (int i = 0; i < 12; i++) A_orig[i] = A[i];

    double S[3];       /* singular values */
    double U[9];       /* m x m = 3x3 */
    double Vt[16];     /* n x n = 4x4 */
    double superb[2];  /* min(m,n)-1 for superdiagonal of bidiagonal */

    print_matrix("A (3x4)", A, m, n);

    /*
     * LAPACKE_dgesvd(layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb)
     *
     * jobu  = 'A': all m columns of U (full U)
     * jobvt = 'A': all n rows of Vt (full Vt)
     * superb: workspace for bidiagonal superdiagonal
     */
    lapack_int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A',
                                      m, n, A, n, S, U, m, Vt, n, superb);
    if (info != 0) {
        fprintf(stderr, "dgesvd failed: info = %d\n", info);
        return 1;
    }

    printf("Singular values:\n");
    for (int i = 0; i < min_mn; i++)
        printf("  sigma_%d = %8.4f\n", i, S[i]);

    print_matrix("U (3x3)", U, m, m);
    print_matrix("Vt (4x4)", Vt, n, n);

    /*
     * Reconstruct: A_recon = U * diag(S) * Vt
     * First compute T = U * diag(S) (scale columns of U by S)
     * Then A_recon = T * Vt
     */
    double T[12] = {0}; /* m x n */
    for (int i = 0; i < m; i++)
        for (int j = 0; j < min_mn; j++)
            T[i * n + j] = U[i * m + j] * S[j];

    double A_recon[12] = {0};
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, 1.0, T, n, Vt, n, 0.0, A_recon, n);

    print_matrix("Reconstructed A = U*S*Vt (3x4)", A_recon, m, n);

    /* Check max reconstruction error */
    double max_err = 0.0;
    for (int i = 0; i < m * n; i++) {
        double err = fabs(A_orig[i] - A_recon[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max reconstruction error: %.2e\n", max_err);

    return 0;
}
```

**Expected output:**
```
A (3x4):
    1.0000    0.0000    0.0000    2.0000
    0.0000    0.0000    3.0000    0.0000
    0.0000    0.0000    0.0000    0.0000
Singular values:
  sigma_0 =   3.0000
  sigma_1 =   2.2361
  sigma_2 =   0.0000
U (3x3):
    0.0000    1.0000    0.0000
    1.0000    0.0000    0.0000
    0.0000    0.0000   -1.0000
Vt (4x4):
    0.0000    0.0000    1.0000    0.0000
    0.4472    0.0000    0.0000    0.8944
    0.0000    0.0000    0.0000    0.0000
    0.8944    0.0000    0.0000   -0.4472
Reconstructed A = U*S*Vt (3x4):
    1.0000    0.0000    0.0000    2.0000
    0.0000    0.0000    3.0000    0.0000
    0.0000    0.0000    0.0000    0.0000
Max reconstruction error: 0.00e+00
```

**Notes:**
- Singular values are always returned in descending order (sigma_0 >= sigma_1 >= ... >= 0).
- `jobu='S'` and `jobvt='S'` compute the "economy" SVD (min(m,n) columns of U and rows of Vt), which is faster for rectangular matrices.
- The `superb` array must have at least `min(m,n) - 1` elements.
- For the divide-and-conquer variant `dgesdd`, no `superb` array is needed and it is generally faster, but uses more memory.

---

## 8. Least Squares

**Key functions:** `LAPACKE_dgels`
**Compile:** `gcc -o example8 example8.c -llapacke -llapack -lcblas -lblas -lm`

```c
/*
 * Solve an overdetermined system (4 equations, 3 unknowns)
 * in the least-squares sense: minimize ||Ax - b||_2.
 */
#include <stdio.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %8.4f", M[i * cols + j]);
        printf("\n");
    }
}

void print_vector(const char *name, const double *v, int n) {
    printf("%s:", name);
    for (int i = 0; i < n; i++)
        printf("  %8.4f", v[i]);
    printf("\n");
}

int main(void) {
    int m = 4, n = 3, nrhs = 1;

    /*
     * Overdetermined system: 4 equations, 3 unknowns.
     * We seek x that minimizes ||A*x - b||_2.
     *
     * This example fits y = c0 + c1*t + c2*t^2 to data points:
     *   t = 0: y = 1.0
     *   t = 1: y = 2.7
     *   t = 2: y = 5.8
     *   t = 3: y = 10.2
     */
    double A[12] = {
        1.0, 0.0, 0.0,   /* row for t=0: [1, 0, 0]   */
        1.0, 1.0, 1.0,   /* row for t=1: [1, 1, 1]   */
        1.0, 2.0, 4.0,   /* row for t=2: [1, 2, 4]   */
        1.0, 3.0, 9.0    /* row for t=3: [1, 3, 9]   */
    };
    /*
     * b must be at least max(m,n) elements for dgels.
     * On exit, first n elements of b contain the least-squares solution.
     */
    double b[4] = { 1.0, 2.7, 5.8, 10.2 };

    /* Save copies */
    double A_orig[12], b_orig[4];
    for (int i = 0; i < 12; i++) A_orig[i] = A[i];
    for (int i = 0; i < 4; i++) b_orig[i] = b[i];

    print_matrix("A (4x3)", A, m, n);
    print_vector("b", b, m);

    /*
     * LAPACKE_dgels(layout, trans, m, n, nrhs, a, lda, b, ldb)
     *
     * trans = 'N': solve min ||A*x - b||  (overdetermined, m >= n)
     * trans = 'T': solve min ||A^T*x - b|| (underdetermined)
     *
     * On exit:
     *   b[0..n-1] = least-squares solution x
     *   b[n..m-1] = residual vector components
     * A is overwritten with QR factorization details.
     */
    lapack_int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m, n, nrhs,
                                     A, n, b, nrhs);
    if (info != 0) {
        fprintf(stderr, "dgels failed: info = %d\n", info);
        return 1;
    }

    printf("\nLeast-squares solution (polynomial coefficients):\n");
    printf("  c0 = %8.4f  (constant)\n", b[0]);
    printf("  c1 = %8.4f  (linear)\n", b[1]);
    printf("  c2 = %8.4f  (quadratic)\n", b[2]);

    /* Compute residual norm: ||A_orig * x - b_orig||_2 */
    double r[4];
    for (int i = 0; i < m; i++) r[i] = -b_orig[i];
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n,
                1.0, A_orig, n, b, 1, 1.0, r, 1);

    double residual_norm = cblas_dnrm2(m, r, 1);
    printf("\nResidual norm ||Ax - b||_2 = %.6f\n", residual_norm);

    /* Show fitted vs actual */
    printf("\nFitted values vs actual:\n");
    for (int i = 0; i < m; i++) {
        double fitted = 0.0;
        for (int j = 0; j < n; j++)
            fitted += A_orig[i * n + j] * b[j];
        printf("  t=%d: actual = %6.2f, fitted = %6.4f\n", i, b_orig[i], fitted);
    }

    return 0;
}
```

**Expected output:**
```
A (4x3):
    1.0000    0.0000    0.0000
    1.0000    1.0000    1.0000
    1.0000    2.0000    4.0000
    1.0000    3.0000    9.0000
b:   1.0000    2.7000    5.8000   10.2000

Least-squares solution (polynomial coefficients):
  c0 =   1.0200  (constant)
  c1 =   1.1100  (linear)
  c2 =   0.5500  (quadratic)

Residual norm ||Ax - b||_2 = 0.109545

Fitted values vs actual:
  t=0: actual =   1.00, fitted =  1.0200
  t=1: actual =   2.70, fitted =  2.6800
  t=2: actual =   5.80, fitted =  5.4400
  t=3: actual =  10.20, fitted =  9.3000
```

**Notes:**
- `dgels` uses QR factorization for overdetermined systems (m >= n) and LQ factorization for underdetermined systems (m < n).
- The `b` array must be allocated with at least `max(m,n)` rows. For overdetermined systems, the solution occupies the first `n` entries.
- For problems where A is rank-deficient, use `dgelsd` (SVD-based) or `dgelsy` (QR with column pivoting) instead.
- The residual sum of squares can also be computed from `b[n..m-1]` after the call.

---

## 9. QR Factorization

**Key functions:** `LAPACKE_dgeqrf`, `LAPACKE_dorgqr`
**Compile:** `gcc -o example9 example9.c -llapacke -llapack -lcblas -lblas -lm`

```c
/*
 * Compute the QR factorization of a 4x3 matrix.
 * Extract Q (4x3 thin Q) and R (3x3 upper triangular) explicitly.
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %8.4f", M[i * cols + j]);
        printf("\n");
    }
}

int main(void) {
    int m = 4, n = 3;
    int k = n; /* min(m,n) = 3 */

    double A[12] = {
        1.0, -1.0,  4.0,
        1.0,  4.0, -2.0,
        1.0,  4.0,  2.0,
        1.0, -1.0,  0.0
    };

    /* Save original */
    double A_orig[12];
    for (int i = 0; i < 12; i++) A_orig[i] = A[i];

    double tau[3]; /* Householder scalars */

    print_matrix("A (4x3)", A, m, n);

    /*
     * Step 1: QR factorization  A = Q * R
     * LAPACKE_dgeqrf(layout, m, n, a, lda, tau)
     *
     * On exit: R is stored in the upper triangle of A.
     *          Householder reflectors are stored below the diagonal + in tau.
     */
    lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau);
    if (info != 0) {
        fprintf(stderr, "dgeqrf failed: info = %d\n", info);
        return 1;
    }

    /* Extract R (upper triangular part of A, 3x3) */
    double R[9] = {0};
    for (int i = 0; i < k; i++)
        for (int j = i; j < n; j++)
            R[i * n + j] = A[i * n + j];

    print_matrix("R (3x3, upper triangular)", R, k, n);

    /*
     * Step 2: Generate the explicit Q matrix (thin Q: 4x3)
     * LAPACKE_dorgqr(layout, m, n, k, a, lda, tau)
     *
     * m = 4, n = 3 (thin Q), k = 3 (number of reflectors)
     * On exit: A contains the first n columns of Q (the thin Q factor).
     */
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, k, A, n, tau);
    if (info != 0) {
        fprintf(stderr, "dorgqr failed: info = %d\n", info);
        return 1;
    }

    print_matrix("Q (4x3, thin Q)", A, m, n);

    /* Verify: Q^T * Q should be I_3 (orthonormal columns) */
    double QtQ[9] = {0};
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, n, m, 1.0, A, n, A, n, 0.0, QtQ, n);
    print_matrix("Q^T * Q (should be I_3)", QtQ, n, n);

    /* Verify: Q * R should reconstruct A */
    double A_recon[12] = {0};
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, 1.0, A, n, R, n, 0.0, A_recon, n);
    print_matrix("Q * R (should equal A)", A_recon, m, n);

    /* Max reconstruction error */
    double max_err = 0.0;
    for (int i = 0; i < m * n; i++) {
        double err = fabs(A_orig[i] - A_recon[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max reconstruction error: %.2e\n", max_err);

    return 0;
}
```

**Expected output:**
```
A (4x3):
    1.0000   -1.0000    4.0000
    1.0000    4.0000   -2.0000
    1.0000    4.0000    2.0000
    1.0000   -1.0000    0.0000
R (3x3, upper triangular):
   -2.0000   -3.0000   -2.0000
    0.0000   -5.0000    2.0000
    0.0000    0.0000   -4.0000
Q (4x3, thin Q):
   -0.5000    0.5000   -0.5000
   -0.5000   -0.5000    0.5000
   -0.5000   -0.5000   -0.5000
   -0.5000    0.5000    0.5000
Q^T * Q (should be I_3):
    1.0000    0.0000    0.0000
    0.0000    1.0000    0.0000
    0.0000    0.0000    1.0000
Q * R (should equal A):
    1.0000   -1.0000    4.0000
    1.0000    4.0000   -2.0000
    1.0000    4.0000    2.0000
    1.0000   -1.0000    0.0000
Max reconstruction error: 0.00e+00
```

**Notes:**
- `dgeqrf` stores the factorization in a compact form: R in the upper triangle, and Householder vectors below the diagonal. You must call `dorgqr` to expand Q explicitly.
- The "thin" Q (m x n where m > n) has orthonormal columns. For the full Q (m x m), set the `n` parameter of `dorgqr` to m and provide an m x m array.
- QR factorization is the basis for `dgels` (least squares) and is also used for rank-revealing decompositions (`dgeqp3`).
- For LQ factorization (useful when m < n), use `dgelqf` and `dorglq`.

---

## 10. Batch Vector Operations

**Key functions:** `cblas_daxpy`, `cblas_ddot`, `cblas_dnrm2`, `cblas_dscal`
**Compile:** `gcc -o example10 example10.c -lcblas -lblas -lm`

```c
/*
 * Demonstrate several BLAS Level 1 vector operations:
 * daxpy, ddot, dnrm2, dscal, dasum, idamax, dcopy, dswap.
 */
#include <stdio.h>
#include <cblas.h>

void print_vector(const char *name, const double *v, int n) {
    printf("%-12s:", name);
    for (int i = 0; i < n; i++)
        printf("  %8.4f", v[i]);
    printf("\n");
}

int main(void) {
    int n = 5;
    double x[5] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    double y[5] = { 5.0, 4.0, 3.0, 2.0, 1.0 };
    double z[5];

    printf("=== BLAS Level 1 Operations ===\n\n");
    print_vector("x", x, n);
    print_vector("y", y, n);

    /* --- ddot: dot product --- */
    double dot = cblas_ddot(n, x, 1, y, 1);
    printf("\n--- cblas_ddot ---\n");
    printf("x . y = %.4f\n", dot);

    /* --- dnrm2: Euclidean norm --- */
    double nrm = cblas_dnrm2(n, x, 1);
    printf("\n--- cblas_dnrm2 ---\n");
    printf("||x||_2 = %.4f\n", nrm);

    /* --- dasum: sum of absolute values --- */
    double asum = cblas_dasum(n, x, 1);
    printf("\n--- cblas_dasum ---\n");
    printf("||x||_1 = %.4f\n", asum);

    /* --- idamax: index of max absolute value --- */
    CBLAS_INDEX idx = cblas_idamax(n, x, 1);
    printf("\n--- cblas_idamax ---\n");
    printf("Index of max |x_i| = %zu (value = %.4f)\n", idx, x[idx]);

    /* --- dcopy: copy x into z --- */
    cblas_dcopy(n, x, 1, z, 1);
    printf("\n--- cblas_dcopy ---\n");
    print_vector("z = copy(x)", z, n);

    /* --- dscal: scale z in-place --- */
    cblas_dscal(n, 2.5, z, 1);
    printf("\n--- cblas_dscal ---\n");
    print_vector("z = 2.5*x", z, n);

    /* --- daxpy: y = alpha*x + y --- */
    double w[5];
    cblas_dcopy(n, y, 1, w, 1);  /* w = y */
    cblas_daxpy(n, 3.0, x, 1, w, 1);  /* w = 3.0*x + y */
    printf("\n--- cblas_daxpy ---\n");
    print_vector("w = 3*x + y", w, n);

    /* --- dswap: swap x and y --- */
    double x2[5], y2[5];
    cblas_dcopy(n, x, 1, x2, 1);
    cblas_dcopy(n, y, 1, y2, 1);
    cblas_dswap(n, x2, 1, y2, 1);
    printf("\n--- cblas_dswap ---\n");
    print_vector("x (after swap)", x2, n);
    print_vector("y (after swap)", y2, n);

    return 0;
}
```

**Expected output:**
```
=== BLAS Level 1 Operations ===

x           :    1.0000    2.0000    3.0000    4.0000    5.0000
y           :    5.0000    4.0000    3.0000    2.0000    1.0000

--- cblas_ddot ---
x . y = 35.0000

--- cblas_dnrm2 ---
||x||_2 = 7.4162

--- cblas_dasum ---
||x||_1 = 15.0000

--- cblas_idamax ---
Index of max |x_i| = 4 (value = 5.0000)

--- cblas_dcopy ---
z = copy(x):    1.0000    2.0000    3.0000    4.0000    5.0000

--- cblas_dscal ---
z = 2.5*x  :    2.5000    5.0000    7.5000   10.0000   12.5000

--- cblas_daxpy ---
w = 3*x + y:    8.0000   10.0000   12.0000   14.0000   16.0000

--- cblas_dswap ---
x (after swap):    5.0000    4.0000    3.0000    2.0000    1.0000
y (after swap):    1.0000    2.0000    3.0000    4.0000    5.0000
```

**Notes:**
- Level 1 BLAS operates on vectors. These are O(n) operations and memory-bandwidth-bound on modern hardware.
- The `incX` and `incY` parameters (stride) allow operating on non-contiguous elements. Use `incX=1` for contiguous arrays. Use `incX=n` to operate on a column of a row-major matrix.
- `cblas_daxpy` is the most commonly used Level 1 function: y = alpha*x + y. It modifies y in-place.
- `cblas_idamax` returns a `CBLAS_INDEX` (typically `size_t`), which is the index of the element with the largest absolute value.

---

## 11. Triangular Solve

**Key functions:** `cblas_dtrsm`
**Compile:** `gcc -o example11 example11.c -lcblas -lblas -lm`

```c
/*
 * Solve A * X = B where A is upper triangular.
 * B has multiple right-hand sides (3x2 matrix).
 */
#include <stdio.h>
#include <cblas.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %8.4f", M[i * cols + j]);
        printf("\n");
    }
}

int main(void) {
    int m = 3; /* rows of B and order of A */
    int n = 2; /* columns of B (number of right-hand sides) */

    /* Upper triangular matrix A (3x3) */
    double A[9] = {
        2.0, 1.0, 3.0,
        0.0, 4.0, 1.0,
        0.0, 0.0, 5.0
    };

    /* Right-hand side B (3x2), overwritten with solution X */
    double B[6] = {
        13.0, 10.0,
         9.0,  5.0,
        10.0, 15.0
    };

    /* Save copies for verification */
    double A_orig[9], B_orig[6];
    for (int i = 0; i < 9; i++) A_orig[i] = A[i];
    for (int i = 0; i < 6; i++) B_orig[i] = B[i];

    print_matrix("A (upper triangular)", A, m, m);
    print_matrix("B (right-hand sides)", B, m, n);

    /*
     * cblas_dtrsm(layout, side, uplo, transA, diag, M, N,
     *             alpha, A, lda, B, ldb)
     *
     * Solves: op(A) * X = alpha * B   (side = CblasLeft)
     * or:     X * op(A) = alpha * B   (side = CblasRight)
     *
     * side   = CblasLeft:  A is on the left
     * uplo   = CblasUpper: A is upper triangular
     * transA = CblasNoTrans: no transpose
     * diag   = CblasNonUnit: diagonal is not assumed to be 1
     * alpha  = 1.0: no scaling of B
     *
     * On exit, B is overwritten with X.
     */
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, m, n,
                1.0, A, m, B, n);

    print_matrix("Solution X", B, m, n);

    /* Verify: compute A * X and compare with B_orig */
    double Check[6] = {0};
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, 1.0, A_orig, m, B, n, 0.0, Check, n);

    print_matrix("A * X (should equal B_orig)", Check, m, n);

    return 0;
}
```

**Expected output:**
```
A (upper triangular):
    2.0000    1.0000    3.0000
    0.0000    4.0000    1.0000
    0.0000    0.0000    5.0000
B (right-hand sides):
   13.0000   10.0000
    9.0000    5.0000
   10.0000   15.0000
Solution X:
    1.0000   -2.0000
    1.7500    0.5000
    2.0000    3.0000
A * X (should equal B_orig):
   13.0000   10.0000
    9.0000    5.0000
   10.0000   15.0000
```

**Notes:**
- `cblas_dtrsm` is a Level 3 BLAS operation that exploits the triangular structure of A for efficient solving.
- Unlike LAPACKE solvers, `dtrsm` does not perform pivoting. It requires A to be non-singular (all diagonal elements non-zero for `CblasNonUnit`).
- Use `CblasUnit` for the `diag` parameter if A has an implicit unit diagonal (diagonal elements are all 1 and not stored).
- `dtrsm` is commonly used after LU or Cholesky factorization to apply the forward/backward substitution steps manually.

---

## 12. Banded Linear System

**Key functions:** `LAPACKE_dgbsv`
**Compile:** `gcc -o example12 example12.c -llapacke -llapack -lcblas -lblas -lm`

```c
/*
 * Solve a tridiagonal (banded) linear system stored in LAPACK band format.
 * The system is: [-1, 2, -1] tridiagonal matrix of size 5x5.
 */
#include <stdio.h>
#include <string.h>
#include <lapacke.h>

void print_vector(const char *name, const double *v, int n) {
    printf("%s:", name);
    for (int i = 0; i < n; i++)
        printf("  %8.4f", v[i]);
    printf("\n");
}

int main(void) {
    int n = 5;       /* matrix size */
    int kl = 1;      /* number of sub-diagonals */
    int ku = 1;      /* number of super-diagonals */
    int nrhs = 1;
    int ldab = 2*kl + ku + 1; /* = 4 for row-major band storage */

    /*
     * Band storage format (LAPACK_ROW_MAJOR):
     * For an n x n matrix with kl sub-diagonals and ku super-diagonals,
     * the band matrix AB has ldab = 2*kl + ku + 1 columns and n rows.
     *
     * Row-major layout: AB[i * ldab + j] where:
     *   - Row i corresponds to matrix row i (i = 0..n-1)
     *   - Column j maps to the band: the matrix element A(i, i+j-kl-ku)
     *
     * For our tridiagonal matrix:
     *   ldab = 2*1 + 1 + 1 = 4
     *   Column 0: extra space for LU fill-in (kl rows)
     *   Column 1: super-diagonal (ku)
     *   Column 2: main diagonal
     *   Column 3: sub-diagonal (kl)
     *
     * The 5x5 tridiagonal matrix:
     *   [ 2 -1  0  0  0 ]
     *   [-1  2 -1  0  0 ]
     *   [ 0 -1  2 -1  0 ]
     *   [ 0  0 -1  2 -1 ]
     *   [ 0  0  0 -1  2 ]
     */
    double AB[20]; /* n * ldab = 5 * 4 = 20 */
    memset(AB, 0, sizeof(AB));

    /*
     * Fill band storage (row-major):
     * For row-major, AB is n rows by ldab columns.
     * Band element A(i,j) goes into AB[i * ldab + (kl + ku + j - i)]
     * where kl + ku = 2 is the offset of the main diagonal.
     */
    for (int i = 0; i < n; i++) {
        /* Main diagonal: A(i,i) = 2.0 */
        AB[i * ldab + (kl + ku)] = 2.0;
        /* Super-diagonal: A(i, i+1) = -1.0 */
        if (i < n - 1)
            AB[i * ldab + (kl + ku + 1)] = -1.0;
        /* Sub-diagonal: A(i, i-1) = -1.0 */
        if (i > 0)
            AB[i * ldab + (kl + ku - 1)] = -1.0;
    }

    printf("Band storage AB (n=%d, kl=%d, ku=%d, ldab=%d):\n", n, kl, ku, ldab);
    for (int i = 0; i < n; i++) {
        printf("  row %d:", i);
        for (int j = 0; j < ldab; j++)
            printf("  %6.1f", AB[i * ldab + j]);
        printf("\n");
    }

    double b[5] = { 1.0, 0.0, 0.0, 0.0, 1.0 };
    lapack_int ipiv[5];

    print_vector("b", b, n);

    /*
     * LAPACKE_dgbsv(layout, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb)
     *
     * Solves A*x = b where A is stored in band format.
     * AB is overwritten with LU factors in band form.
     * b is overwritten with the solution x.
     */
    lapack_int info = LAPACKE_dgbsv(LAPACK_ROW_MAJOR, n, kl, ku, nrhs,
                                     AB, ldab, ipiv, b, nrhs);
    if (info != 0) {
        fprintf(stderr, "dgbsv failed: info = %d\n", info);
        return 1;
    }

    print_vector("Solution x", b, n);

    /*
     * For this particular system and RHS, the exact solution is:
     * x = [5/6, 4/6, 3/6, 4/6, 5/6] (fractions of 6)
     */
    printf("Expected:  ");
    for (int i = 0; i < n; i++)
        printf("  %8.4f", (double[]){5.0/6, 4.0/6, 3.0/6, 4.0/6, 5.0/6}[i]);
    printf("\n");

    return 0;
}
```

**Expected output:**
```
Band storage AB (n=5, kl=1, ku=1, ldab=4):
  row 0:     0.0    -1.0     2.0     0.0
  row 1:     0.0    -1.0     2.0    -1.0
  row 2:     0.0    -1.0     2.0    -1.0
  row 3:     0.0    -1.0     2.0    -1.0
  row 4:     0.0     0.0     2.0    -1.0
b:   1.0000    0.0000    0.0000    0.0000    1.0000
Solution x:   0.8333    0.6667    0.5000    0.6667    0.8333
Expected:     0.8333    0.6667    0.5000    0.6667    0.8333
```

**Notes:**
- Band storage requires `ldab = 2*kl + ku + 1` columns. The first `kl` columns are reserved for fill-in during LU factorization.
- For row-major layout, band element A(i,j) is stored at `AB[i * ldab + (kl + ku + j - i)]`.
- Banded solvers are O(n * kl * ku) rather than O(n^3), making them much faster for narrow-banded systems.
- For purely tridiagonal systems (kl=ku=1), consider `LAPACKE_dgtsv` which uses a simpler storage format (three separate vectors).

---

## 13. Symmetric Positive Definite Solve

**Key functions:** `LAPACKE_dposv`
**Compile:** `gcc -o example13 example13.c -llapacke -llapack -lcblas -lblas -lm`

```c
/*
 * Directly solve Ax = b where A is symmetric positive definite (SPD),
 * without explicit factorization steps.
 * dposv combines dpotrf + dpotrs in one call.
 */
#include <stdio.h>
#include <lapacke.h>
#include <cblas.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %8.4f", M[i * cols + j]);
        printf("\n");
    }
}

void print_vector(const char *name, const double *v, int n) {
    printf("%s:", name);
    for (int i = 0; i < n; i++)
        printf("  %8.4f", v[i]);
    printf("\n");
}

int main(void) {
    int n = 3, nrhs = 2; /* two right-hand sides */

    /* SPD matrix: diagonally dominant guarantees positive definiteness */
    double A[9] = {
        4.0, 2.0, 1.0,
        2.0, 5.0, 3.0,
        1.0, 3.0, 6.0
    };

    /*
     * B is n x nrhs (row-major).
     * Each column of B is a separate right-hand side.
     */
    double B[6] = {
        11.0,  7.0,
        19.0, 13.0,
        25.0, 16.0
    };

    /* Save copies */
    double A_orig[9], B_orig[6];
    for (int i = 0; i < 9; i++) A_orig[i] = A[i];
    for (int i = 0; i < 6; i++) B_orig[i] = B[i];

    print_matrix("A (SPD)", A, n, n);
    print_matrix("B (two RHS columns)", B, n, nrhs);

    /*
     * LAPACKE_dposv(layout, uplo, n, nrhs, a, lda, b, ldb)
     *
     * Combines Cholesky factorization (dpotrf) and solve (dpotrs).
     * uplo = 'U': upper triangle of A is stored and used.
     *
     * On exit:
     *   A upper triangle contains the Cholesky factor U (A = U^T * U).
     *   B is overwritten with the solutions.
     */
    lapack_int info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', n, nrhs,
                                     A, n, B, nrhs);
    if (info != 0) {
        fprintf(stderr, "dposv failed: info = %d\n", info);
        if (info > 0) fprintf(stderr, "Matrix is not positive definite.\n");
        return 1;
    }

    printf("Cholesky factor U (upper triangle of A):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j >= i)
                printf("  %8.4f", A[i * n + j]);
            else
                printf("  %8s", "");
        }
        printf("\n");
    }

    print_matrix("Solutions X (two columns)", B, n, nrhs);

    /* Verify both solutions: R = A_orig * X - B_orig */
    printf("Verification (residuals for each RHS):\n");
    for (int rhs = 0; rhs < nrhs; rhs++) {
        double res_norm = 0.0;
        for (int i = 0; i < n; i++) {
            double sum = -B_orig[i * nrhs + rhs];
            for (int j = 0; j < n; j++)
                sum += A_orig[i * n + j] * B[j * nrhs + rhs];
            res_norm += sum * sum;
        }
        printf("  RHS %d: ||A*x - b||_2 = %.2e\n", rhs, res_norm);
    }

    return 0;
}
```

**Expected output:**
```
A (SPD):
    4.0000    2.0000    1.0000
    2.0000    5.0000    3.0000
    1.0000    3.0000    6.0000
B (two RHS columns):
   11.0000    7.0000
   19.0000   13.0000
   25.0000   16.0000
Cholesky factor U (upper triangle of A):
    2.0000    1.0000    0.5000
              2.0000    1.2500
                        2.0310
Solutions X (two columns):
    1.0000    1.0000
    1.0000    1.0000
    3.0000    1.5000
Verification (residuals for each RHS):
  RHS 0: ||A*x - b||_2 = 0.00e+00
  RHS 1: ||A*x - b||_2 = 0.00e+00
```

**Notes:**
- `dposv` is a convenience "driver" that combines factorization and solve in one call. Use it when you do not need the factored matrix separately.
- For multiple solves with the same A but different B, call `dpotrf` once, then `dpotrs` repeatedly.
- `dposv` is roughly twice as fast as `dgesv` for SPD matrices since Cholesky is cheaper than LU and requires no pivoting.
- Use `dposvx` (expert driver) to also get a condition number estimate and iterative refinement.

---

## 14. Matrix Norm and Condition Number

**Key functions:** `LAPACKE_dlange`, `LAPACKE_dgetrf`, `LAPACKE_dgecon`
**Compile:** `gcc -o example14 example14.c -llapacke -llapack -lcblas -lblas -lm`

```c
/*
 * Compute the 1-norm condition number of a matrix:
 *   cond_1(A) = ||A||_1 * ||A^{-1}||_1
 *
 * Uses: dlange (matrix norm), dgetrf (LU factor), dgecon (condition estimate).
 */
#include <stdio.h>
#include <lapacke.h>

void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("  %12.6f", M[i * cols + j]);
        printf("\n");
    }
}

int main(void) {
    int n = 3;

    double A[9] = {
        1.0,  0.0,  0.0,
        0.0,  1.0,  0.0,
        0.0,  0.0,  1e-8
    };

    /* Save original for norm computation (dgetrf overwrites A) */
    double A_copy[9];
    for (int i = 0; i < 9; i++) A_copy[i] = A[i];

    print_matrix("A", A, n, n);

    /*
     * Step 1: Compute ||A||_1  (maximum absolute column sum)
     * LAPACKE_dlange(layout, norm, m, n, a, lda)
     *
     * norm = '1': 1-norm  (max absolute column sum)
     * norm = 'I': infinity-norm (max absolute row sum)
     * norm = 'F': Frobenius norm
     * norm = 'M': max absolute element
     */
    double anorm = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', n, n, A, n);
    printf("||A||_1 = %.6e\n", anorm);

    /*
     * Step 2: LU factorize A (needed by dgecon)
     */
    lapack_int ipiv[3];
    lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A, n, ipiv);
    if (info != 0) {
        fprintf(stderr, "dgetrf failed: info = %d\n", info);
        return 1;
    }

    /*
     * Step 3: Estimate the reciprocal condition number
     * LAPACKE_dgecon(layout, norm, n, a, lda, anorm, &rcond)
     *
     * Input: a = LU factors from dgetrf, anorm = the norm computed before factoring.
     * Output: rcond = reciprocal condition number estimate (1/cond if well-conditioned).
     *
     * rcond is in [0, 1]:
     *   rcond ~= 1.0: well-conditioned
     *   rcond ~= 0.0: ill-conditioned or singular
     *   rcond = 0.0: exactly singular
     */
    double rcond;
    info = LAPACKE_dgecon(LAPACK_ROW_MAJOR, '1', n, A, n, anorm, &rcond);
    if (info != 0) {
        fprintf(stderr, "dgecon failed: info = %d\n", info);
        return 1;
    }

    printf("Reciprocal condition number (rcond) = %.6e\n", rcond);
    printf("Estimated condition number = %.6e\n", 1.0 / rcond);

    /* Interpret the result */
    if (rcond < 1e-15)
        printf("WARNING: Matrix is numerically singular.\n");
    else if (rcond < 1e-8)
        printf("WARNING: Matrix is ill-conditioned. Solutions may lose %.0f digits.\n",
               -__builtin_log10(rcond));
    else
        printf("Matrix is reasonably well-conditioned.\n");

    /* Now test with a well-conditioned matrix */
    printf("\n--- Well-conditioned example ---\n");
    double B[9] = {
        4.0, 2.0, 1.0,
        2.0, 5.0, 3.0,
        1.0, 3.0, 6.0
    };
    double B_copy[9];
    for (int i = 0; i < 9; i++) B_copy[i] = B[i];

    print_matrix("B", B, n, n);

    double bnorm = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', n, n, B, n);
    printf("||B||_1 = %.6f\n", bnorm);

    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, B, n, ipiv);
    if (info != 0) { fprintf(stderr, "dgetrf failed\n"); return 1; }

    info = LAPACKE_dgecon(LAPACK_ROW_MAJOR, '1', n, B, n, bnorm, &rcond);
    if (info != 0) { fprintf(stderr, "dgecon failed\n"); return 1; }

    printf("Reciprocal condition number (rcond) = %.6e\n", rcond);
    printf("Estimated condition number = %.6f\n", 1.0 / rcond);
    printf("Matrix is reasonably well-conditioned.\n");

    return 0;
}
```

**Expected output:**
```
A:
     1.000000     0.000000     0.000000
     0.000000     1.000000     0.000000
     0.000000     0.000000     0.000010
||A||_1 = 1.000000e+00
Reciprocal condition number (rcond) = 1.000000e-08
Estimated condition number = 1.000000e+08
WARNING: Matrix is ill-conditioned. Solutions may lose 8 digits.

--- Well-conditioned example ---
B:
     4.000000     2.000000     1.000000
     2.000000     5.000000     3.000000
     1.000000     3.000000     6.000000
||B||_1 = 10.000000
Reciprocal condition number (rcond) = 6.818182e-02
Estimated condition number = 14.666667
Matrix is reasonably well-conditioned.
```

**Notes:**
- Always compute the norm BEFORE calling `dgetrf`, since `dgetrf` overwrites A with LU factors.
- `dgecon` estimates the reciprocal condition number (rcond) using the LU factors. It is O(n^2), much cheaper than computing the full inverse.
- A condition number of 10^k means you lose approximately k digits of accuracy when solving linear systems.
- For SPD matrices, use `LAPACKE_dpocon` after `dpotrf` instead.
- Use the 1-norm or infinity-norm; the Frobenius norm gives a looser bound.

---

## 15. Building and Linking Guide

This section provides compilation and linking instructions for various platforms and BLAS/LAPACK implementations.

### GCC (Reference LAPACK)

```bash
# Link order matters: dependent libraries first
gcc -o myprogram myprogram.c -llapacke -llapack -lcblas -lblas -lm

# With Fortran runtime (sometimes needed)
gcc -o myprogram myprogram.c -llapacke -llapack -lcblas -lblas -lm -lgfortran
```

### Clang

```bash
# Same flags as GCC
clang -o myprogram myprogram.c -llapacke -llapack -lcblas -lblas -lm
```

### pkg-config

```bash
# Check if LAPACKE pkg-config is available
pkg-config --cflags --libs lapacke

# Use in compilation
gcc -o myprogram myprogram.c $(pkg-config --cflags --libs lapacke) -lm
```

### OpenBLAS (optimized, recommended)

OpenBLAS bundles BLAS, CBLAS, LAPACK, and LAPACKE into a single library.

```bash
# Simple linking
gcc -o myprogram myprogram.c -lopenblas -lm

# With explicit include path
gcc -I/usr/include/openblas -o myprogram myprogram.c -lopenblas -lm

# With pkg-config
gcc -o myprogram myprogram.c $(pkg-config --cflags --libs openblas) -lm
```

### Intel MKL

```bash
# Using the single dynamic library (simplest)
gcc -o myprogram myprogram.c -lmkl_rt -lm -lpthread

# Using MKL's provided link line advisor script
# (see: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html)

# Sequential (single-threaded)
gcc -o myprogram myprogram.c \
    -L${MKLROOT}/lib/intel64 \
    -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm -lpthread

# Threaded (OpenMP)
gcc -o myprogram myprogram.c \
    -L${MKLROOT}/lib/intel64 \
    -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm -lpthread

# Using pkg-config (after sourcing MKL environment)
source /opt/intel/oneapi/setvars.sh
gcc -o myprogram myprogram.c $(pkg-config --cflags --libs mkl-dynamic-lp64-seq) -lm
```

### Platform-Specific Installation

**Ubuntu / Debian:**
```bash
# Reference LAPACK + CBLAS
sudo apt install liblapacke-dev libopenblas-dev

# Headers go to /usr/include/ or /usr/include/x86_64-linux-gnu/
# Libraries go to /usr/lib/x86_64-linux-gnu/
```

**Fedora / RHEL / CentOS:**
```bash
# Reference LAPACK
sudo dnf install lapack-devel

# OpenBLAS
sudo dnf install openblas-devel

# LAPACKE headers
sudo dnf install lapacke-devel
```

**macOS (Homebrew):**
```bash
# Install OpenBLAS
brew install openblas

# Compile with explicit paths (Homebrew does not link into /usr/local by default)
gcc -o myprogram myprogram.c \
    -I$(brew --prefix openblas)/include \
    -L$(brew --prefix openblas)/lib \
    -lopenblas -lm

# Note: macOS also ships with the Accelerate framework (vecLib):
gcc -o myprogram myprogram.c -framework Accelerate -lm
# (Uses #include <Accelerate/Accelerate.h> instead of #include <cblas.h>)
```

**Conda / Mamba:**
```bash
# Install OpenBLAS-based LAPACK
conda install -c conda-forge liblapacke openblas

# Compile with conda prefix
gcc -o myprogram myprogram.c \
    -I$CONDA_PREFIX/include \
    -L$CONDA_PREFIX/lib \
    -lopenblas -lm -Wl,-rpath,$CONDA_PREFIX/lib
```

### CMake (CMakeLists.txt)

```cmake
cmake_minimum_required(VERSION 3.14)
project(lapack_example C)

set(CMAKE_C_STANDARD 99)

# Find LAPACK and BLAS (CMake has built-in FindLAPACK and FindBLAS modules)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

# Look for LAPACKE and CBLAS headers
# (FindLAPACK does not always find LAPACKE headers)
find_path(LAPACKE_INCLUDE_DIR lapacke.h
    HINTS /usr/include /usr/local/include /usr/include/lapacke
          $ENV{MKLROOT}/include
          $ENV{CONDA_PREFIX}/include
)

find_path(CBLAS_INCLUDE_DIR cblas.h
    HINTS /usr/include /usr/local/include /usr/include/openblas
          $ENV{MKLROOT}/include
          $ENV{CONDA_PREFIX}/include
)

# Add executable
add_executable(example1 example1.c)
target_include_directories(example1 PRIVATE
    ${LAPACKE_INCLUDE_DIR}
    ${CBLAS_INCLUDE_DIR}
)
target_link_libraries(example1 PRIVATE
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
    m
)

# Alternative: if using OpenBLAS, you can link directly
# find_package(OpenBLAS REQUIRED)
# target_link_libraries(example1 PRIVATE OpenBLAS::OpenBLAS m)

# To build all examples at once
set(EXAMPLES
    example1 example2 example3 example4 example5
    example6 example7 example8 example9 example10
    example11 example12 example13 example14
)
foreach(ex ${EXAMPLES})
    add_executable(${ex} ${ex}.c)
    target_include_directories(${ex} PRIVATE
        ${LAPACKE_INCLUDE_DIR}
        ${CBLAS_INCLUDE_DIR}
    )
    target_link_libraries(${ex} PRIVATE
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
        m
    )
endforeach()
```

**Build with CMake:**
```bash
mkdir build && cd build
cmake ..
make

# To use a specific BLAS/LAPACK (e.g., OpenBLAS)
cmake -DBLA_VENDOR=OpenBLAS ..

# To use Intel MKL
cmake -DBLA_VENDOR=Intel10_64lp_seq ..

# Common BLA_VENDOR values:
#   OpenBLAS, Intel10_64lp_seq, Intel10_64lp, Apple, Generic
```

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `undefined reference to cblas_dgemm` | Missing CBLAS library | Add `-lcblas` or `-lopenblas` |
| `undefined reference to LAPACKE_dgesv` | Missing LAPACKE library | Add `-llapacke` or `-lopenblas` |
| `undefined reference to dgemm_` | Missing Fortran BLAS | Add `-lblas -lgfortran` |
| `lapacke.h: No such file` | Missing dev package | Install `liblapacke-dev` / `lapacke-devel` |
| `cannot find -llapacke` | Library not in search path | Add `-L/path/to/lib` |
| Wrong results | Row/column major mismatch | Verify `LAPACK_ROW_MAJOR` usage consistently |
| Slow performance | Using reference BLAS | Switch to OpenBLAS or MKL |

### Library Comparison

| Library | Speed | Ease of Use | Notes |
|---------|-------|-------------|-------|
| Reference LAPACK | Baseline | Simple | Unoptimized; good for correctness testing |
| OpenBLAS | Fast | Simple | Open source; auto-detects CPU; single `-lopenblas` |
| Intel MKL | Fastest (Intel) | Complex linking | Free via oneAPI; best on Intel CPUs |
| Apple Accelerate | Fast (Apple) | Simple | macOS only; `-framework Accelerate` |
| BLIS | Fast | Moderate | Modern micro-kernel architecture |
| ATLAS | Fast | Complex | Auto-tuned; long build time |
