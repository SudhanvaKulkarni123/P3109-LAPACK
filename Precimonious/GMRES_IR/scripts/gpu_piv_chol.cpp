#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>


#define CUDA_CHECK(err)                                               \
    do {                                                              \
        cudaError_t e = (err);                                        \
        if (e != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(e));       \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

#define CUBLAS_CHECK(err)                                             \
    do {                                                              \
        cublasStatus_t s = (err);                                     \
        if (s != CUBLAS_STATUS_SUCCESS) {                             \
            fprintf(stderr, "cuBLAS error %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)




void luFactorizationBlocked(
    int n,
    int b,            // block size
    float* dA,
    int lda,
    int* pivot,       // pivot array on the host (for simplicity)
    cublasHandle_t cublasHandle)
{
    // Set constants for cuBLAS calls:
    const float alpha = -1.0f;
    const float beta  = 1.0f;
    const float one   = 1.0f;

    // We'll need some host workspace to perform pivoting decisions.
    std::vector<float> hWork(n);

    for (int k = 0; k < n; k += b) {
        // Size of the current block:
        int pb = std::min(b, n - k);

        // ----------------------------------------------------------
        // 1) Factor the panel (unblocked) - partial pivot on CPU.
        //    We'll do it column by column for the panel.
        // ----------------------------------------------------------
        for (int col = 0; col < pb; ++col) {
            // Download the current column from device to host (from row k+col to n-1).
            // We'll do a small chunk (just for pivot searching):
            // The column index in the global matrix is (k+col).
            CUDA_CHECK(
                cudaMemcpy(hWork.data() + (k + col),
                           dA + (k+col)*lda + (k+col), // device pointer offset
                           (n - (k + col)) * sizeof(float),
                           cudaMemcpyDeviceToHost)
            );

    
            // Now we do the "division" step to make L:
            // A[k+col+1 : n, k+col] = A[k+col+1 : n, k+col] / A[k+col, k+col]
            // We'll fetch the pivot element from device to host just for a single float.
            float pivotVal;
            CUDA_CHECK(
                cudaMemcpy(&pivotVal,
                           dA + (k+col)*lda + (k+col),
                           sizeof(float),
                           cudaMemcpyDeviceToHost)
            );
            // Avoid division by zero: 
            if (fabsf(pivotVal) < 1e-20f) {
                fprintf(stderr, "Numerical error: pivot is too small.\n");
                // handle error or partial pivot fallback
            }

            float invPivot = 1.0f / pivotVal;

            

            // Next step, we do the rank-1 update for the panel strictly *in the same block*.
            //   A[k+col+1 : n, k+col+1 : k+pb] -= A[k+col+1 : n, k+col] * A[k+col, k+col+1 : k+pb]
            // This is a small update (still part of the "panel factorization").

            int trailingColsPanel = pb - (col + 1);
            if (trailingColsPanel > 0 && numRowsBelow > 0) {
                // cublasSger is a BLAS-2 routine. For the panel part, we can do it unblocked with ger.
                // However, strictly speaking, a "pure BLAS-3" approach typically lumps this into TRSM, etc.
                // For clarity, let's do a small ger:

                float minusOne = -1.0f;
                CUBLAS_CHECK(
                    cublasSger(
                        cublasHandle,
                        numRowsBelow,            // M
                        trailingColsPanel,       // N
                        &minusOne,
                        dA + (k+col)*lda + (k+col+1), // x pointer
                        1,
                        dA + (k + col + 1)*lda + (k + col), // y pointer
                        lda,
                        dA + (k + col + 1)*lda + (k + col + 1), // A pointer
                        lda
                    )
                );
            }
        }

        // ----------------------------------------------------------
        // 2) TRSM: solve for the block to the right of the panel
        //    U-part: L^-1 * A_panel_right
        // ----------------------------------------------------------
        int trailingCols = n - (k + pb);
        if (pb > 0 && trailingCols > 0) {
            // We call TRSM on the block: "Left, Lower, Not-trans, Unit diag"
            // This effectively does: A[k:n, k+pb : n] = L^-1 * (that block)
            // But we do it only for the block rows: from k to k+pb.
            // So the operation is:
            //   A[k : k+pb, k+pb : n] = L[k : k+pb, k : k+pb]^-1 * A[k : k+pb, k+pb : n]
            CUBLAS_CHECK(
                cublasStrsm(
                    cublasHandle,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,       // no transpose
                    CUBLAS_DIAG_UNIT,  // L has unit diagonal in the standard LU decomposition
                    pb,
                    trailingCols,
                    &one,
                    dA + k*lda + k, // L block start
                    lda,
                    dA + k*lda + (k + pb), // portion of A to solve
                    lda
                )
            );
        }

        // ----------------------------------------------------------
        // 3) TRSM for the block below the panel (the L part), or
        //    directly do the next step for the trailing submatrix update:
        //    A[k+pb : n, k+pb : n] -= A[k+pb : n, k : k+pb] * A[k : k+pb, k+pb : n]
        // ----------------------------------------------------------
        int trailingRows = n - (k + pb);
        if (trailingRows > 0 && trailingCols > 0) {
            // A[k+pb : n, k+pb : n] = A[k+pb : n, k+pb : n] - A[k+pb : n, k : k+pb] * A[k : k+pb, k+pb : n]
            CUBLAS_CHECK(
                cublasSgemm(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    trailingRows,    // M
                    trailingCols,    // N
                    pb,              // K
                    &alpha,          // alpha = -1
                    dA + (k)*lda + (k + pb),   // A pointer => offset (k+pb) columns, k rows
                    lda,
                    dA + (k + pb)*lda + k,     // B pointer => offset (k+pb) rows, k cols
                    lda,
                    &beta,           // beta = 1
                    dA + (k + pb)*lda + (k + pb), // C pointer => trailing submatrix
                    lda
                )
            );
        }
    }
}

int main()
{
    // Example usage:
    // Let's factor an n x n matrix with block size b.
    int n = 8;   // small example
    int b = 2;   // block size

    // Host array (row-major in C, but we treat it as col-major for cuBLAS).
    // For simplicity, let's define lda = n.
    std::vector<float> hA(n*n);

    // Fill hA with some data (for example).
    for (int i = 0; i < n*n; ++i) {
        hA[i] = static_cast<float>(rand() % 10 + 1);
    }

    // Device memory
    float* dA = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, n*n*sizeof(float)));

    // Copy input matrix to device
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), n*n*sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    // Pivot array
    std::vector<int> pivot(n);

    // Call our blocked LU routine
    luFactorizationBlocked(n, b, dA, n, pivot.data(), cublasHandle);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(hA.data(), dA, n*n*sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
    CUDA_CHECK(cudaFree(dA));

    // Print out result (L and U combined in hA).
    // The upper triangle (including diagonal) is U,
    // The lower triangle (below diagonal) is L (with implied 1s on diagonal).
    printf("LU-factorized matrix (combined L and U):\n");
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            printf("%8.4f ", hA[col*n + row]); // careful if col-major indexing
        }
        printf("\n");
    }

    // Print pivots
    printf("Pivot indices:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", pivot[i]);
    }
    printf("\n");

    return 0;
}
