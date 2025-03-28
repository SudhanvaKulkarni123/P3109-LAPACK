#include <fstream>
#include <queue>
#include <algorithm>
#include <set>
#include <cmath>
#include <random>
#include <iostream>

using namespace tlapack;

/* ---------------------------------------------------------
 * RBF kernel helper and norm/distance computations
 * ---------------------------------------------------------*/

template<typename T>
float DiffNrm(const T& x, const T& y, int n)
{
    float norm = 0.0f;
    for(int i = 0; i < n; i++) {
        float d = x[i] - y[i];
        norm += d * d;
    }
    return std::sqrt(norm);
}

template <typename Matrix, typename T>
T rbf_value(const Matrix& points, int i, int j, T sigma, int img_size)
{
    auto col_i = tlapack::col(points, i);
    auto col_j = tlapack::col(points, j);
    float dist = DiffNrm(col_i, col_j, img_size);
    // RBF kernel: exp(-(dist^2)/(2 sigma^2))
    return std::exp( - (dist * dist) / (2 * sigma * sigma) );
}

/* ---------------------------------------------------------
 * Sampling diag(A) for pivot indices
 * ---------------------------------------------------------*/

template<typename T>
void sample_indices(const std::vector<T>& A, int r, std::set<int>& samples) 
{
    std::vector<double> probabilities(A.size());
    float sum_A = 0.0f;
    for(auto val : A) sum_A += val;

    for (size_t i = 0; i < A.size(); i++) {
        probabilities[i] = A[i] / sum_A;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

    while (samples.size() < (size_t) r) {
        samples.insert(dist(gen));
    }
}

/* ---------------------------------------------------------
 * Packing columns/rows
 * ---------------------------------------------------------*/

// Pack the columns in 'cols' of the RBF kernel into dst (n-by-r).
template<typename Matrix, typename data_t>
void pack_RBF_cols(Matrix& dst, const std::set<int>& cols, const data_t& points, float sigma = 1.0f)
{
    using idx_t = size_type<Matrix>;
    idx_t n = nrows(dst); // number of rows
    idx_t r = ncols(dst); // number of columns in dst

    int colCount = 0;
    for (auto c : cols) {
        for(int i = 0; i < int(n); i++) {
            dst(i, colCount) = static_cast<type_t<Matrix>>(
                rbf_value(points, i, c, sigma, 28*28)
            );
        }
        colCount++;
    }
}

// Copy only the pivot rows (S') of src into dst.  That is, dst is r-by-k if S' has size r.
template<typename MatrixIn, typename MatrixOut>
void pack_rows(const MatrixIn& src, MatrixOut& dst, const std::set<int>& rows)
{
    int rowCount = 0;
    for(auto r : rows) {
        for(int j = 0; j < (int)ncols(src); j++) {
            dst(rowCount, j) = src(r, j);
        }
        rowCount++;
    }
}

/* ---------------------------------------------------------
 * Minimal fix: copy only the pivot rows from G1 into G3
 * ---------------------------------------------------------*/
template<typename Matrix>
void copy_matrix(Matrix& src, Matrix& dst, const std::set<int>& pivotset)
{
    // We assume dst is r-by-r, where r = pivotset.size().
    // We copy G1(rows in pivotset, 0..r-1) into G3(0..r-1,0..r-1).
    std::vector<int> piv(pivotset.begin(), pivotset.end());
    int r = (int) pivotset.size();
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < r; j++) {
            dst(i, j) = src(piv[i], j);
        }
    }
}

/* ---------------------------------------------------------
 * Basic Cholesky + TRSM in float
 * ---------------------------------------------------------*/

// Add eps*trace(A) to diag, then normal Cholesky
template<typename Matrix>
void perturbed_cholesky(Matrix& A, float trace_A, int n, float eps = 0.125f)
{
    for(int i = 0; i < n; i++) {
        A(i, i) += trace_A*eps;
    }
    // basic unblocked Cholesky
    for(int j = 0; j < n; j++) {
        for(int k = 0; k < j; k++) {
            A(j, j) -= A(j, k)*A(j, k);
        }
        A(j, j) = std::sqrt(A(j, j));
        for(int i = j+1; i < n; i++) {
            for(int k = 0; k < j; k++) {
                A(i, j) -= A(i, k)*A(j, k);
            }
            A(i, j) /= A(j, j);
        }
    }

    // Copy lower to upper
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < i; j++) {
            A(j, i) = A(i, j);
        }
    }
}

// B = B * A^{-1}, where A is (k-by-k) upper‐tri, B is (n-by-k).
template<typename fp32_Matrix>
void fp32_TRSM(fp32_Matrix& A, fp32_Matrix& B, int k, int n)
{
    // For each row of B, do row*(inv A).
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < k; j++) {
            for(int l = 0; l < j; l++) {
                B(i, j) -= A(j, l)*B(i, l);
            }
            B(i, j) /= A(j, j);
        }
    }
}

/* ---------------------------------------------------------
 * GEMM in float from low-precision inputs
 * ---------------------------------------------------------*/

template<typename fp8_Matrix, typename fp32_Matrix, typename scaling_mat>
void fp8fp32_GEMM(
    fp8_Matrix& A, fp8_Matrix& B,
    scaling_mat& A_exp, scaling_mat& B_exp,
    float alpha, float beta,
    fp32_Matrix& C, int m, int n, int k, int r)
{
    // C = alpha * (A * B^T) + beta*C
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++) {
            for(int l = 0; l < n; l++) {
                // minimal fix: accumulate instead of just = 
                C(i, j) = alpha
                          * (static_cast<float>(A(i, l))*A_exp(i, l/r))
                          * (static_cast<float>(B(j, l))*B_exp(j, l/r));
                          
            }
            C(i,j) += beta * C(i,j);
        }
    }
}

/* ---------------------------------------------------------
 * fp8 <-> fp32 casting
 * ---------------------------------------------------------*/

// Convert an n-by-r block from float -> fp8
template<typename hi_Matrix, typename lo_Matrix, typename Scaling_Mat>
void fp32Tofp8(
    hi_Matrix& src, lo_Matrix& dst,
    Scaling_Mat& scaling_mat,
    int n, int k, int r, int col_offset=0, int row_offset=0)
{
    // Very simplistic block scaling. 
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < k/r; j++) {

        float maxVal = 0.f;
            for(int cc = 0; cc < r; cc++) {
                float val = src(i, j*r + cc);
                maxVal = std::max(maxVal, std::fabs(val));
            }
        
        scaling_mat(i, j) = maxVal > 0.f ? maxVal : 1.f;

            for(int cc = 0; cc < r; cc++) {
                float scaled = src(i, j*r + cc) / scaling_mat(i, j);
                dst(i + row_offset, j*r + cc + col_offset) =
                    static_cast<type_t<lo_Matrix>>(scaled);
            }
        
        }
        
    }
}

/* ---------------------------------------------------------
 * The main algorithm (block variant)
 * ---------------------------------------------------------*/

 //might need a more aggressive block scaing that has a smaller block size-> perhaps a common exp for every r elements in a row

template<typename lo_matrix_t,
         typename scal_matrix_t,
         typename hi_matrix_t,
         typename data_t>
void rand_piv_RBF(
    lo_matrix_t& A,   // n-by-k in fp8: final factor F built block by block
    scal_matrix_t& A_exp,   // scaling matrix of int8_t for A that is size n-by-k/r
    hi_matrix_t& G1,  // n-by-r (float) for newly sampled columns
    lo_matrix_t& G2,  // r-by-k (fp8) for F(S',:)
    scal_matrix_t G2_exp, // scaling matrix of int8_t for G2 that is size r-by-k/r
    hi_matrix_t& G3,  // r-by-r (float) for Cholesky block
    data_t& diag_A,   // length n, diagonal of residual
    hi_matrix_t& points, // input data for RBF
    int n,
    int k,
    std::set<int>& S, // pivot set
    int r      = 32,
    float sigma= 1.0f )
{
    // 1) Initialize diag(A) if not already
    std::cout << "starting RBF kernel\n";
    float trace = 0.0f;
    for(int i = 0; i < n; i++) {
        diag_A[i] = rbf_value(points, i, i, sigma, 28*28);
        trace += diag_A[i];
    }

    // 2) Loop over blocks
    int nblocks = k / r;  // assume k multiple of r

    for(int iBlk = 0; iBlk < nblocks; iBlk++) 
    {
        // (a) Sample r pivots from diag_A
        std::set<int> S_prime;
        sample_indices(diag_A, r, S_prime);
        for(auto elem : S_prime) {
            S.insert(elem);
        }

        // (b) Build G1 = A(:, S') (n * s) in float
        // clear G1
        for(int rr = 0; rr < n; rr++) {
            for(int cc = 0; cc < r; cc++) {
                G1(rr, cc) = 0.f;
            }
        }
        pack_RBF_cols(G1, S_prime, points, sigma);

        int s = S_prime.size();

        // (c) Subtract overlap: G1 <- G1 - [F * F(S')^T]
        // We have F in A (fp8 and n*k) and F(S')^T in G2 (also fp8), so pack rows:
        int builtCols = iBlk * r;  // how many columns of F are already built
        if(builtCols > 0) {
            // G2 should be r-by-builtCols => pack from A
            // (S', 0..builtCols-1)
            for(int rr=0; rr<int(r); rr++) {
                int pivotIdx = *std::next(S_prime.begin(), rr);
                for(int cc=0; cc<builtCols; cc++){
                    G2(rr, cc) = A(pivotIdx, cc);
                    if(cc % r == 0) {
                        G2_exp(rr, cc/r) = A_exp(pivotIdx, cc/r);
                    }
                }
            }
            // Now do G1 = G1 - A * G2^T in float
            fp8fp32_GEMM(A, G2, A_exp, G2_exp, -1.0f, 1.0f,
                         G1, n, builtCols, r);
        }

        // (d) Form R-block = G1(S',:) in G3, do perturbed Cholesky
        {
            // copy pivot rows from G1 into G3
            // G3 is r-by-r (or s-by-s if dups sampling)
            copy_matrix(G1, G3, S_prime);
            float eps = 1e-7f * trace;  // small shift
            perturbed_cholesky(G3, eps, s);
        }

        // (e) Solve G1 = G1 * inv(R-block), store back to A in fp8
        {
            // G1 = G1 * inv(G3)
            fp32_TRSM(G3, G1, s, n);
            // Then cast that block into A’s columns [iBlk*r .. iBlk*r + r - 1]
            fp32Tofp8(G1, A, A_exp, n, r, iBlk*r, 0);
        }

        // (f) Update diag_A: subtract row‐norms^2 of the new block in G1
        for(int row = 0; row < n; row++) {
            float sqsum = 0.f;
            for(int col = 0; col < r; col++) {
                float val = G1(row, col);
                sqsum += val * val;
            }
            diag_A[row] = std::max(0.f, diag_A[row] - sqsum);
        }
    }
    // End.  A now contains the factor F in fp8 form.
    
}

