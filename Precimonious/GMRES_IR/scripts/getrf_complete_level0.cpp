///@author Sudhanva Kulkarni, UC Berkeley
///this file contains level-1 BLAS code for finding LU factroization of a matrix with complete pivoting.
///So it returns A = PLUQ

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/ger.hpp"


template<TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR piv_t>
int LU_complete(matrix_t& A, piv_t& left_piv, piv_t& right_piv) {
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    
    int n = nrows(A);

    for(int i = 0; i < n; i++) {
        left_piv[i] = i;
        right_piv[i] = i;
    }
    for(int k = 0; k < n; i++) {

    double max_val = 0.0;
    int pivot_row = k, pivot_col = k;

    for (int i = k; i < n; i++) {
        for (int j = k; j < n; j++) {
            if (fabs(A(i, j)) > max_val) {
                max_val = fabs(A(i, j));
                pivot_row = i;
                pivot_col = j;
            }
        }
    }

    if (max_val == 0) {
        throw runtime_error("Matrix is singular and cannot be decomposed");
        return -1;
    }
    if (pivot_row != k) {
        swap(row_perm[k], row_perm[pivot_row]);  // Track row swap
        for (int j = 0; j < n; j++) {
            swap(A(k, j), A(pivot_row, j));      // Swap rows in matrix
        }
    }
    
    if (pivot_col != k) {
        swap(col_perm[k], col_perm[pivot_col]);  // Track column swap
        for (int i = 0; i < n; i++) {
            swap(A(i, k), A(i, pivot_col));      // Swap columns in matrix
        }
    }

    // Step 3: LU Decomposition: Perform the elimination
    for (int i = k + 1; i < n; i++) {
        A(i, k) /= A(k, k);   // Compute the multiplier
        for (int j = k + 1; j < n; j++) {
            A(i, j) -= A(i, k) * A(k, j);  // Update remaining submatrix
        }
    }

    }

    return 0;
}





void LU_complete_test() 
{

    vector<float> A_(36, 0.0);
    Create<LegacyMatrix<float, int>> gemm_matrix;
    auto A = gemm_matrix(A_, 6, 6);


    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 6; j++) {
            A(i,j) = float(rand())/float(RAND_MAX);
        }
    }
    

}


template<TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR piv_t>
int getrf_complte_blocked(matrix_t& A, piv_t& piv_left, piv_t& piv_right, int r = 32, int stopping_point = 9999999999, double tol = 0.0001)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;
    using gemm_type = mult_type;
    using gemm_type_2 = float;
    using trsm_type_1 = float;
    using trsm_type_2 = float;
    using trsm_type_3 = Eigen::half;

Create<LegacyMatrix<gemm_type, idx_t>> gemm_matrix;
Create<LegacyMatrix<gemm_type_2, idx_t>> gemm_2_matrix;

// Using the following lines to pass the abs function to iamax
IamaxOpts optsIamax([](const T& x) -> real_t { return abs(x); });

// constants
const idx_t m = nrows(A);
const idx_t n = ncols(A);
const idx_t k = min(m, n);
assert(n == m);
assert(n % r == 0);
int q = n / r;

// Pivots for both rows and columns
for (idx_t i = 0; i < n; ++i) {
    piv_right[i] = i;
    piv_left[i] = i;  // Initialize column pivots

}

if (n <= r) {
    return LU_complete(A, piv_left, piv_right);
}

auto maxA = tlapack::lange(tlapack::Norm::Max, A);

for (int k = 0; k < q; k++) {

    auto piv1_left = tlapack::slice(piv_left, range(k * r, (k + 1) * r));
    auto piv1_right = tlapack::slice(piv_right, range(k * r, (k + 1) * r));
    auto my_block = tlapack::slice(A, range(k * r, n), range(k * r, (k + 1) * r));

    for(int kk = k*r; kk < n; kk++) {

    double max_val = 0.0;
    int pivot_row = k, pivot_col = k;

    for (int i = k; i < n; i++) {
        for (int j = k; j < n; j++) {
            if (fabs(A(i, j)) > max_val) {
                max_val = fabs(A(i, j));
                pivot_row = i;
                pivot_col = j;
            }
        }
    }

    if (max_val == 0) {
        throw runtime_error("Matrix is singular and cannot be decomposed");
        return -1;
    }
    if (pivot_row != k) {
        swap(row_perm[k], row_perm[pivot_row]);  // Track row swap
        for (int j = 0; j < n; j++) {
            swap(A(k, j), A(pivot_row, j));      // Swap rows in matrix
        }
    }
    
    if (pivot_col != k) {
        swap(col_perm[k], col_perm[pivot_col]);  // Track column swap
        for (int i = 0; i < n; i++) {
            swap(A(i, k), A(i, pivot_col));      // Swap columns in matrix
        }
    }

    // Step 3: LU Decomposition: Perform the elimination
    for (int i = k + 1; i < n; i++) {
        A(i, k) /= A(k, k);   // Compute the multiplier
        for (int j = k + 1; j < n; j++) {
            A(i, j) -= A(i, k) * A(k, j);  // Update remaining submatrix
        }
    }

    }

    // Apply row pivoting to right and left blocks as before
    auto right_block = tlapack::slice(A, range((k) * r, n), range((k + 1) * r, n));
    for (idx_t j = 0; j < r; j++) {
        if ((idx_t)piv1[j] != j) {
            auto vect1 = tlapack::row(right_block, j);
            auto vect2 = tlapack::row(right_block, piv1[j]);
            tlapack::swap(vect1, vect2);
        }
    }

    auto left_block = tlapack::slice(A, range(k * r, n), range(0, k * r));
    for (idx_t j = 0; j < r; j++) {
        if ((idx_t)piv1[j] != j) {
            auto vect1 = tlapack::row(left_block, j);
            auto vect2 = tlapack::row(left_block, piv1[j]);
            tlapack::swap(vect1, vect2);
        }
    }

    // Apply column pivoting
    for (idx_t j = 0; j < r; j++) {
        // Find the column pivot index
        idx_t max_idx = j;
        real_t max_val = fabs(A(k * r + j, k * r + j));

        for (idx_t i = j + 1; i < r; ++i) {
            if (fabs(A(k * r + i, k * r + j)) > max_val) {
                max_val = fabs(A(k * r + i, k * r + j));
                max_idx = i;
            }
        }

        // Swap columns in A if necessary
        if (max_idx != j) {
            piv_col[k * r + j] = max_idx;  // Record column pivot
            for (idx_t i = 0; i < n; ++i) {
                swap(A(i, k * r + j), A(i, k * r + max_idx));
            }
        }
    }

    // Apply the updated column pivots to the right block
    auto top_block = tlapack::slice(A, range(0, k * r), range((k)*r, n));
    for (idx_t j = 0; j < r; j++) {
        if ((idx_t)piv_col[k * r + j] != j) {
            auto vect1 = tlapack::col(top_block, j);
            auto vect2 = tlapack::col(top_block, piv_col[k * r + j]);
            tlapack::swap(vect1, vect2);
        }
    }

    // LU decomposition continues with TRSM and GEMM steps
    for (idx_t i = 0; i < r; i++) {
        piv1[i] += k * r;
    }

    auto A01 = tlapack::slice(A, range((k) * r, (k + 1) * r), range(k * r + r, n));
    auto A00 = tlapack::slice(A, range((k) * r, (k + 1) * r), range(k * r, k * r + r));
    tlapack::trsm(tlapack::LEFT_SIDE, tlapack::LOWER_TRIANGLE, tlapack::NO_TRANS, tlapack::UNIT_DIAG, 1.0, A00, A01);

    auto A10 = tlapack::slice(A, range((k + 1) * r, n), range((k) * r, (k + 1) * r));
    auto A11 = tlapack::slice(A, range((k + 1) * r, n), range((k + 1) * r, n));

    std::vector<gemm_type> buf_L_(r * (n - (k + 1) * r));
    auto buf_L = gemm_2_matrix(buf_L_, n - (k + 1) * r, r);
    std::vector<gemm_type> buf_U_(r * (n - (k + 1) * r));
    auto buf_U = gemm_2_matrix(buf_U_, r, n - (k + 1) * r);

    block_gemm(A10, A01, A11, buf_L, buf_U);
}

return 0;



}

