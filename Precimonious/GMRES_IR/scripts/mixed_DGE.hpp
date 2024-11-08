/// @author Sudhanva Kulkarni
/// This file contains code for mixed precision LU for diagonally dominant matrices.
/// The idea is the same as Cholesky. Prepivot the diagonal so that values are decreasing 
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


template<TLAPACK_VECTOR vector_t>
int max_index(vector_t& v, int start) {
    int to_ret = start;
    int n = size(v);
    for(int i = start; i < n; i++) {
        if(abs(v[i]) > abs(v[to_ret])) to_ret = i;
    }
    return to_ret;
}



template<TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
void trailing_min_arr(matrix_t& A, vector_t min_vec, int r = 32) {

    int n = nrows(A);
    int q = n/r;

    for(int i = 0; i < n; i++) min_vec[i] = INFINITY;

    for(int t = q-1; t >=0; t--) {
        for(int i = n-1; i  >= t*r ; i--) {
            for(int j = n-1; j >= t*r; j--) {
                    min_vec[i] = std::min(min_vec[i], abs(A(i,j)));
            }
        }
    }

    return;
}

//function for regular unblocked Cholesky decomp. returns decomp in lower part of matrix
template<TLAPACK_MATRIX matrix_t>
int lu_kernel(matrix_t& A)
{   
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    idx_t n = nrows(A);
    real_t tmp = real_t(0.0);

    for (idx_t k = 0; k < n; ++k)
    {
        if (A(k, k) == T(0))
        {
            return k + 1; 
        }
        for (idx_t i = k + 1; i < n; ++i)
        {
            A(i, k) /= A(k, k);
        }

        for (idx_t i = k + 1; i < n; ++i)
        {
            for (idx_t j = k + 1; j < n; ++j)
            {
                A(i, j) -= A(i, k) * A(k, j);
            }
        }
    }

    return 0;

}



template<TLAPACK_SCALAR scalar_t, TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
void update_diag(matrix_t& A, matrix_t& L, matrix_t& U, vector_t& updated_A)
{
    using idx_t = size_type<matrix_t>;
    int n = nrows(L);
    int b = ncols(L);

    for(int i = 0; i < n; i++) updated_A[i] = A(i,i);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < b; j++) {
            updated_A[i] -= L(i,j)*U(j,i);
        }
    }


    return;

}

template<TLAPACK_SCALAR scalar_t, TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
bool can_use_type_general(matrix_t& A, vector_t& diag_A, vector_t& updated_A, double eps_prime)
{
    using idx_t = size_type<matrix_t>;
    int n = nrows(A);
    double mach_eps = (double)std::numeric_limits<scalar_t>::epsilon();

    for(int i = 0; i < n; i++) {     
        if (!(mach_eps*abs(updated_A[i]) < eps_prime*diag_A[i])) return false;
    }
    
    return true;   
}



template<TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR piv_t>
int diag_LU(matrix_t& A, piv_t& left_piv, piv_t& right_piv, n_flops& flop_counter,  int r = 32, double tol = 0.0000000000001)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<int, int>;


    using gemm_type = float;
    using gemm_type2 =  Eigen::half;
    using gemm_type3 = ml_dtypes::float8_ieee_p<7>;



    Create<LegacyMatrix<gemm_type, idx_t>> gemm_matrix;
    Create<LegacyMatrix<gemm_type2, idx_t>> gemm_2_matrix;
    Create<LegacyMatrix<gemm_type3, idx_t>> gemm_3_matrix;

    idx_t n = nrows(A);
    int q = n/r;


    using val_idx = pair<T, int>;

    if ( n <= r) return lu_kernel(A);


    vector<T> diag_A(n);
    for(int i = 0; i < n; i++) diag_A[i] = abs(A(i,i));

    vector<T> updated_diag(n);
    for(int i = 0; i < n; i++) updated_diag[i] = A(i,i);


    // cholesky_kernel(A);
    // return 0;
  

    for(int i  = 0; i < n; i++) {

        int curr_idx = max_index(diag_A, i);
        auto tmp_scal = diag_A[curr_idx];
        diag_A[curr_idx] = diag_A[i];
        diag_A[i] = tmp_scal;
        auto tmp = tlapack::col(A, curr_idx);
        auto tmp2 = tlapack::col(A, i);
        tlapack::swap(tmp, tmp2);
        auto tmp3 = tlapack::row(A, curr_idx);
        auto tmp4 = tlapack::row(A, i);
        tlapack::swap(tmp3, tmp4);
        left_piv[i] = curr_idx;
        right_piv[i] = curr_idx;
        
    }

    bool sorted = true;
    for(int i = 0; i < n-1; i++) {
        if(diag_A[i] < diag_A[i+1]) { sorted = false; cout << "not sorted"; break;}
    }

    if (sorted) cout << " array is sorted!";
    cout << "min diag elem = " << A(n-1, n-1) << " , max diag elem = " << A(0,0) << "\n";
    cout << "ration of indices is : " << A(0,0)/A(n-1, n-1) << endl;


    //Now, that that's done, we construct mins_A, where the i-th entry of mins_A is the minimum by abs in row i%q of block "A[i//q :, i//q : ]"
    //where q is block size.

    std::vector<T> mins_A(n, INFINITY);

    for(int i = n-1; i >= 0; i--) {
        for(int j = i/q; j < n; j++) {
            for(int k = i/q; k < n; k++) {
                mins_A[i] = std::min(mins_A[i], abs(A(j,k)));
            }
        }
    }



    
    // cout << "after sorted diag : \n";
   

    //now the matrix is ready to be Cholesky'd

    for(int i = 0; i < q; i++) 
    {
        auto A00 = tlapack::slice(A, range(i*r, (i+1)*r), range(i*r, (i+1)*r));
         

        lu_kernel(A00);
        flop_counter.add_float_flops(2*r*r*r/3);

        auto A01 = tlapack::slice(A ,  range((i)*r, (i+1)*r), range((i+1)*r, n));
        auto A10 = tlapack::slice(A,  range((i+1)*r, n) ,range(i*r, (i+1)*r));

        tlapack::trsm(tlapack::LEFT_SIDE, Uplo::Lower, tlapack::NO_TRANS, tlapack::Diag::Unit, 1.0, A00, A01);
        flop_counter.add_float_flops(r*r*r*(q - i - 1));

        tlapack::trsm(tlapack::RIGHT_SIDE, Uplo::Upper, Op::NoTrans, Diag::NonUnit, 1.0, A00, A10);
        flop_counter.add_float_flops(r*r*r*(q - i - 1));


        //need to provide alternate check where eps_8*A(i, i) < eps'*diag_A[i] for indices in the Schur complement



        double err_bnd = tol*diag_A[(i+1)*r]/(A((i+1)*r, (i+1)*r));
        flop_counter.add_double_flops(1);
        

        auto A11 = tlapack::slice(A, range(i*r + r, n), range(i*r + r, n));
        auto diag_left = std::vector<T>(diag_A.begin() + i*r + r, diag_A.end());
        auto update_left = std::vector<T>(updated_diag.begin() + i*r + r, updated_diag.end());
        auto mins_left = std::vector<T>(mins_A.begin() + i*r + r, mins_A.end());

        update_diag<float>(A11, A10, A01, update_left);
        int swich = 2;

        if (can_use_type_general<gemm_type3>(A11, diag_left , update_left, tol)) {
            swich = 0;
            cout << " using lowest precision \n"; 
            std::vector<gemm_type3> buf_L_(r * (n - (i+1)*r));
            auto buf_L = gemm_3_matrix(buf_L_, n - (i+1)*r, r);
            std::vector<gemm_type3> buf_U_(r * (n - (i+1)*r));
            auto buf_U = gemm_3_matrix(buf_U_, r, n - (i+1)*r);
            squeezing_matmul(A10, A01, A11, buf_L, buf_U, -1.0, 1.0);
            //block_gemm(A10, A01, A11, buf_L, buf_U);
            //scaled_matmul(A10, A01, A11, buf_L, buf_U, r);
            flop_counter.add_fp8_flops(r*(n - (i+1)*r)*(n - (i+1)*r));
        } else if(can_use_type_general<gemm_type2>(A11, diag_left , update_left, tol)) {
            swich = 1;
            cout << " using middle precision \n"; 
            std::vector<gemm_type2> buf_L_(r * (n - (i+1)*r));
            auto buf_L = gemm_2_matrix(buf_L_, n - (i+1)*r, r);
            std::vector<gemm_type2> buf_U_(r * (n - (i+1)*r));
            auto buf_U = gemm_2_matrix(buf_U_, r, n - (i+1)*r);
            block_gemm(A10, A01, A11, buf_L, buf_U);
            //squeezing_matmul(A10, A01, A11, buf_L, buf_U, -1.0, 1.0);
            flop_counter.add_half_flops(r*(n - (i+1)*r)*(n - (i+1)*r));
        } else {
            swich = 2;
            cout << " using highest precision \n"; 
            std::vector<gemm_type> buf_L_(r * (n - (i+1)*r));
            auto buf_L = gemm_matrix(buf_L_, n - (i+1)*r, r);
            std::vector<gemm_type> buf_U_(r * (n - (i+1)*r));
            auto buf_U = gemm_matrix(buf_U_, r, n - (i+1)*r);
            block_gemm(A10, A01, A11, buf_L, buf_U);
            flop_counter.add_float_flops(r*(n - (i+1)*r)*(n - (i+1)*r));

        }

  

        
    }

    return 0;


}
