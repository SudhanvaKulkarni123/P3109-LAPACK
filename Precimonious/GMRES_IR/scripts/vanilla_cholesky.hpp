/// @author Sudhanva Kulkarni, UC Berkeley
/// literally just a regular Cholesky in double precision


#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/lapack/rscl.hpp"
#include "tlapack/lapack/getrf.hpp"



template<TLAPACK_MATRIX matrix_t> 
void vanilla_cholesky(matrix_t& A, int block_size = 32) 
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<int, int>;
    int n = nrows(A);
    int q = n/block_size;

    using gemm_type = double;

    for(int i = 0; i < q; i++) {

        auto A00 = tlapack::slice(A, range(i*r, (i+1)*r), range(i*r, (i+1)*r));
        auto A01 = tlapack::slice(A ,  range((i)*r, (i+1)*r), range((i+1)*r, n));
    
        tlapack::trsm(tlapack::LEFT_SIDE, Uplo::Lower, tlapack::NO_TRANS, tlapack::Diag::NonUnit, 1.0, A00, A01);
        auto A10 = tlapack::slice(A,  range((i+1)*r, n) ,range(i*r, (i+1)*r));

        
        for(int s = 0; s < nrows(A10); s++) {
            for(int t = 0; t < ncols(A10); t++) {
                A10(s, t) = A01(t, s);
            }
        }

        auto A11 = tlapack::slice(A, range(i*r + r, n), range(i*r + r, n));

        std::vector<gemm_type> buf_L_(r * (n - (i+1)*r));
        auto buf_L = gemm_matrix(buf_L_, n - (i+1)*r, r);
        std::vector<gemm_type> buf_U_(r * (n - (i+1)*r));
        auto buf_U = gemm_matrix(buf_U_, r, n - (i+1)*r);
        block_gemm(A10, A01, A11, buf_L, buf_U, r);





    }

    return;


}