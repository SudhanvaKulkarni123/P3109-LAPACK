/// This file conatins code for a blocked LU factorization with partial pivoting
/// the input matrix is n-by-n and blocks are r-by-r
/// @author: Sudhanva Kulkarni, UC berkeley
#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/rscl.hpp"
#include "tlapack/lapack/getrf.hpp"




template <TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR piv_t>
int getrf_blocked(matrix_t& A, piv_t& piv, int r=128)
{

    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;
    using gemm_type = Eigen::half;
    using trsm_type_1 = float;
    using trsm_type_2 = float;
    using trsm_type_3 = Eigen::half;
 
    

    Create<LegacyMatrix<gemm_type, idx_t>> gemm_matrix;

        // Using the following lines to pass the abs function to iamax
    IamaxOpts optsIamax([](const T& x) -> real_t { return abs(x); });

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    assert(n == m);
    assert(n % r == 0);
    int q = n / r;

    if (n <= r) {
        return getrf(A, piv);
    }



    

   

    for(int k = 0; k < q; k++) {

        //getrf(A(k,k))
        auto piv1 = tlapack::slice(piv, range(k*r, (k+1)*r));
        auto my_block = tlapack::slice(A, range(k*r, n), range(k*r, (k+1)*r));
        tlapack::getrf(my_block, piv1);


    
       

        
        //need to apply pivot after this step
        
            auto right_block = tlapack::slice(A, range((k)*r, n), range((k+1)*r, n));

            for (idx_t j = 0 ; j < r ; j++) {
                if ((idx_t)piv1[j] != j) {
                    auto vect1 = tlapack::row(right_block, j);
                    auto vect2 = tlapack::row(right_block, piv1[j]);
                    tlapack::swap(vect1, vect2);
                }
            }

            auto left_block = tlapack::slice(A, range(k*r, n), range(0, k*r));
            for (idx_t j = 0 ; j < r ; j++) {
                if ((idx_t)piv1[j] != j) {
                    auto vect1 = tlapack::row(left_block, j);
                    auto vect2 = tlapack::row(left_block, piv1[j]);
                    tlapack::swap(vect1, vect2);
                }
            }

         

         for (idx_t i = 0; i < r; i++) {
            piv1[i] += k*r;
        }
        // Shift piv1, so piv will have the accurate representation of overall
        // pivots
        

   

        auto A01 = tlapack::slice(A, range((k)*r, (k+1)*r), range(k*r + r, n));
        auto A00 = tlapack::slice(A, range((k)*r, (k+1)*r), range(k*r, k*r + r));
        tlapack::trsm(tlapack::LEFT_SIDE, tlapack::LOWER_TRIANGLE, tlapack::NO_TRANS, tlapack::UNIT_DIAG, 1.0,A00, A01);
        auto A10 = tlapack::slice(A, range((k+1)*r, n), range((k)*r, (k+1)*r));
        auto A11 = tlapack::slice(A, range((k+1)*r, n), range((k+1)*r, n));
        std::vector<gemm_type> buf_L_(r * (n - (k+1)*r));
        auto buf_L = gemm_matrix(buf_L_, n - (k+1)*r, r);
        std::vector<gemm_type> buf_U_(r * (n - (k+1)*r));
        auto buf_U = gemm_matrix(buf_U_, r, n - (k+1)*r);
        block_gemm(A10, A01, A11, buf_L, buf_U);
        //tlapack::gemm(tlapack::NO_TRANS, tlapack::NO_TRANS, (-1), A10, A01,(1), A11);

    
    }

    return 0;
}