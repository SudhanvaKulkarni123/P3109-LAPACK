/// @author Sudhanva Kulkarni
///This file contains code for a mixed precision Incomplete Cholesky-0. 
/// It is hard to say what eps_prime means in the case of an incomplete Cholesky since we drop a bunch of terms.
/// Regardless, we will still use the same swithcing criterion as the dense case. Intuitively, the incomplete Choesky is less likely to switch to lower precision for the same epsilon prime. This is because some l_ik terms are dropped to match the sparsity pattern of A

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/lapack/rscl.hpp"
#include "tlapack/lapack/getrf.hpp"
#include <fstream>
#include <queue>



//level-0 function for ILU-0 cholesky.
template<TLAPACK_SCALAR eps_t, TLAPACK_MATRIX matrix_t>
int incomplete_cholesky_kernel(matrix_t& A, float eta = 1.0, float ksi = 1.0, int N = 1024, chol_mod chol_modif = chol_mod::NONE, bool phase2 = false, float prev_err = 0.0)
{   
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    idx_t n = nrows(A);
    real_t tmp = real_t(0.0);
    for (int i = 0; i < n; i++) {
      
    tmp = A(i,i);
    
    

    for (int k = 0; k < i; k++) {
        tmp -= A(i,k) * A(i,k); 
    }

    auto b_norm = 0.0;
    auto b_inf = 0.0;
    for(int j = i+1; j < n; j++) {
        b_norm += abs(A(j,i));
        b_inf = (b_inf > abs(A(j,i))) ? b_inf : abs(A(j, i));
    }
    // if (tmp <= 0) {
    //     cout << "Matrix is not positive definite, perturbing diagonal to restore positveness\n";
    //     tmp += 0.0625;
    //     cout << "tmp after prturbation : " << tmp << "\n";
    // }


 

    auto mach_eps = (float)std::numeric_limits<eps_t>::epsilon();
    auto NUN = (float)std::numeric_limits<eps_t>::min();
    auto DUN = (float)std::numeric_limits<eps_t>::denorm_min();
    auto beta_sq = std::max(eta, std::max(ksi/ (float)sqrt(N*N - 1), (float)mach_eps));
    // if( tmp <= 0 ) { 
    //     tmp = A(i,i) + (float)std::numeric_limits<eps_t>::epsilon(); std::cout << "negative entry! perturbing to preserve +veness";
    // } 
    auto prev_tmp = tmp;
    if(phase2 && chol_modif == chol_mod::GMW81)
    {
        tmp = abs(tmp) > DUN*2.0 ? abs(tmp) : DUN*2.0;
        tmp = tmp > b_inf*b_inf/ beta_sq ? tmp : b_inf*b_inf/ beta_sq;
        tmp = tmp > prev_err ? tmp : prev_err;
        prev_err = std::max(prev_err, tmp - prev_tmp);
    }
    else if (phase2 && chol_modif == chol_mod::SE90)
    {
        auto pert = std::max((float)0.0, std::max((float)b_norm - tmp, prev_err));
        tmp = tmp + pert;
        prev_err = pert;
    }
    else if (phase2 && chol_modif == chol_mod::SE99) 
    {
        float buf = std::max((float)b_norm, prev_err);
        auto pert = std::max((float)0.0, -tmp + buf);
        tmp = tmp + pert;
        prev_err = pert;
    }
    
    
    

    A(i,i) = sqrt((tmp));  
    

    for (int j = i+1; j < n; j++) {
        tmp = A(j,i);
        for (int k = 0; k < i; k++) {
            tmp -= A(j,k) * A(i,k);  
        }

        A(j,i) = tmp / A(i,i);  
    }


}

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < i; j++) {
            A(j,i) = A(i,j);
        }
    }


    return 0;

}
