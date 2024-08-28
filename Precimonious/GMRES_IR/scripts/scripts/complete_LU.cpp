/// This file conatins code for a blocked LU factorization with complete pivoting
/// the input matrix is n-by-n and blocks are r-by-r
/// @author: Sudhanva Kulkarni, UC berkeley
#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/rscl.hpp"
#include "tlapack/lapack/getrf.hpp"
#include <fstream>






std::ofstream growth_log("growth_log.txt");
template <TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR piv_t>
int getrf_complete_blocked(matrix_t& A, piv_t& piv, int r=256, int stopping_point = 999999)
{

    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;
    using gemm_type =  ml_dtypes::float8_ieee_p<5>;
    using gemm_type_2 = double;
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

    if (n <= r) {
        return getrf(A, piv);
    }

    auto maxA = tlapack::lange(tlapack::Norm::Max, A); 



    
    

    for(int k = 0; k < q; k++) {
        if(k > stopping_point) {
            std::cout << "stooing LU at iteration number " << k << std::endl;
            //fill in the Schur complement with 0s
            for(int i = k*r; i < n; i++) {
                for(int j = k*r ; j < n; j++) {
                    A(i, j) = 0;
                }
                A(i,i) = 1.0;
            }
            //fill the rest of piv[i] with i
            for(int i = k*r; i < n; i++) {
                piv[i] = i;
            }
            break;

        }
        //

        //getrf(A(k,k))
        growth_log << "growth factor at iter k : " << double(max_upper(A, k*r))/double(maxA) << std::endl;
        auto piv1 = tlapack::slice(piv, range(k*r, (k+1)*r));
        auto my_block = tlapack::slice(A, range(k*r, n), range(k*r, (k+1)*r));
        tlapack::getrf(my_block, piv1);
        //flops.add_float_flops(2*n);


    
       

        
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
        //flops.add_float_flops(2*r*r*(n - (k+1)*r));
        auto A10 = tlapack::slice(A, range((k+1)*r, n), range((k)*r, (k+1)*r));
        auto A11 = tlapack::slice(A, range((k+1)*r, n), range((k+1)*r, n));
        
            std::vector<gemm_type> buf_L_(r * (n - (k+1)*r));
            auto buf_L = gemm_matrix(buf_L_, n - (k+1)*r, r);
            std::vector<gemm_type> buf_U_(r * (n - (k+1)*r));
            auto buf_U = gemm_matrix(buf_U_, r, n - (k+1)*r);
            std::vector<double> obuf_L_(r * (n - (k+1)*r));
            auto obuf_L = gemm_2_matrix(obuf_L_, n - (k+1)*r, r);
            std::vector<double> obuf_U_(r * (n - (k+1)*r));
            auto obuf_U = gemm_2_matrix(obuf_U_, r, n - (k+1)*r);
            std::vector<double> buf_res((n - (k+1)*r)*(n - (k+1)*r));
            auto res = gemm_2_matrix(buf_res, n - (k+1)*r, n - (k+1)*r);
            std::vector<float> A_exp_(r * (n - (k+1)*r));
            auto A_exp = gemm_2_matrix(A_exp_, n - (k+1)*r, r);
            std::vector<float> B_exp_(r * (n - (k+1)*r));
            auto B_exp = gemm_2_matrix(B_exp_, r, n - (k+1)*r);
            block_gemm(A10, A01, A11, buf_L, buf_U, obuf_L, obuf_U, res, A_exp, B_exp);
        
        
        //tlapack::gemm(tlapack::NO_TRANS, tlapack::NO_TRANS, (-1), A10, A01,(1), A11);

        //flops.add_fp8_flops(2*r*(n - (k+1)*r)*(n - (k+1)*r));


    }


    return 0;
}