///this file contains code for Rank Revealing QR factorization with column pivoting
#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/ger.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/plugins/legacyArray.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/nrm2.hpp"

template<TLAPACK_VECTOR vector_t>
int min(const vector_t& x)
{
    using idx_t = size_type<vector_t>;
    using T = type_t<vector_t>;
    using real_t = real_type<T>;
    real_t min = x(0);
    int index = 0;
    for(idx_t i = 0; i < size(x); i++)
    {
        min = std::min(min, x(i));
        if(min == x(i))
        {
            index = i;
        }
    }
    return index;
}

template<TLAPACK_MATRIX matrix_t, TLAPACK_MATRIX matrixP_t, TLAPACK_MATRIX matrixV_t, TLAPACK_MATRIX matrixW_t, TLAPACK_VECTOR piv_t, TLAPACK_VECTOR norm_t>
void RRQR(matrix_t& A, matrixP_t& P, matrixV_t& V, matrixW_t& W, norm_t& norms, int block_size)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;
    int m = nrows(A);
    int n = ncols(A);
    for(int j = 0; j < n; j+=block_size)
    {
        for(int k = j; k < j+block_size; k++)
        {   
            int i = min(tlapack::slice(norms, range(k, n)));
            if(k != i){
                auto vect1 = tlapack::col(A, k);
                auto vect2 = tlapack::col(A, i);
                tlapack::swap(vect1, vect2);
                vect1 = tlapack::col(P, k);
                vect2 = tlapack::col(P, i);
                tlapack::swap(vect1, vect2);
            }

            auto Akmk = tlapack::slice(A, ranhe(k,m), k);
            auto Vkmj = tlapack::slice(V, range(k,m), range(j, k-1));
            auto Wkmj = tlapack::slice(W, range(j,k-1), k);

            tlapack::gemv(tlapack::NO_TRANS, -1.0, Vkmj, Wkmj, 1.0, Akmk);

            //compute householder vector to annihilate bottom m-k+1 entries of column k
            auto hh_vec = tlapack::slice(A, range(k,m), k);
            double tau = tlapack::nrm2(hh_vec);
            int sgn = (hh_vec[0] >= 0) ? 1 : -1;
            tau = sgn * tau;
            hh_vec[0] += tau;
            auto Wkpokn = tlapack::slice(W, range(k+1, n), k);
            tlapack::gemv(tlapack::TRANSPOSE, tau, Wkpokn, hh_vec, 0.0, Wkpokn);
            tlapack::gemv(tlapack::TRANSPOSE, tau, tlapack::cols(V, range(j, k-1)), hh_vec, 0.0, tlapack::col(V, k));



            




        }

    }

}