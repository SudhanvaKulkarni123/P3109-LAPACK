/// This file contains code for a pre-pivoted left looking cholesky routine that sorts the values on the diagonal first and 
/// then switches based on the diagonal value we are at  
/// @author: Sudhanva Kulkarni, UC berkeley
#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/lapack/rscl.hpp"
#include "tlapack/lapack/getrf.hpp"
#include "pivoted_cholesky.hpp"
#include <fstream>
#include <queue>





template<TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR piv_t>
int pivoted_cholesky_left(matrix_t& A, piv_t& left_piv, piv_t& right_piv, chol_mod& chol_modif, n_flops& flop_counter,  int r = 32, double tol = 0.0000000000001, float dropping_prob = 0.1)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<int, int>;


    using gemm_type = float;
    using gemm_type2 =  Eigen::half;
    using gemm_type3 = ml_dtypes::float8_ieee_p<5>;


    #ifdef STOCHASTIC_ROUND
    std::cout << "using stochastic rounding!" << std::endl;
    #endif



    Create<LegacyMatrix<gemm_type, idx_t>> gemm_matrix;
    Create<LegacyMatrix<gemm_type2, idx_t>> gemm_2_matrix;
    Create<LegacyMatrix<gemm_type3, idx_t>> gemm_3_matrix;

    idx_t n = nrows(A);
    int q = n/r;
    bool phase2 = false;

    T ksi = -std::numeric_limits<T>::infinity();
    T eta = -std::numeric_limits<T>::infinity();

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(j != i) ksi = (ksi > abs(A(i,j))) ? ksi : abs(A(i,j));
        }
    }


    using val_idx = pair<T, int>;

    if ( n <= r) return cholesky_kernel<float>(A);


    vector<T> diag_A(n);
    for(int i = 0; i < n; i++) diag_A[i] = A(i,i);

    vector<T> updated_diag(n);
    for(int i = 0; i < n; i++) updated_diag[i] = A(i,i);

    vector<T> masks(n);
    for(int i = 0; i < n; i++) masks[i] = 1.0;


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

    eta = diag_A[0];


    bool sorted = true;
    for(int i = 0; i < n-1; i++) {
        if(diag_A[i] < diag_A[i+1]) { sorted = false; cout << "not sorted"; break;}
    }

    if (sorted) cout << " array is sorted!";
    cout << "min diag elem = " << A(n-1, n-1) << " , max diag elem = " << A(0,0) << "\n";
    cout << "ration of indices is : " << A(0,0)/A(n-1, n-1) << endl;



    
    // cout << "after sorted diag : \n";
   

    //now the matrix is ready to be Cholesky'd
    int swich = 3;
    for(int i = 0; i < q; i++) 
    {
        
        auto A00 = tlapack::slice(A, range(i*r, (i+1)*r), range(i*r, (i+1)*r));
        auto A10 = tlapack::slice(A,  range((i+1)*r, n) ,range(i*r, (i+1)*r));
        auto A01 = tlapack::slice(A ,  range((i)*r, (i+1)*r), range((i+1)*r, n));
        auto A11 = tlapack::slice(A, range(i*r + r, n), range(i*r + r, n));

        if (can_use_type<gemm_type3>(A11, diag_left ,update_left, tol)) {
            swich = 0;
            cout << " using lowest precision \n"; //8 bit float fp8
            std::vector<gemm_type3> buf_L_(r * (n - (i+1)*r));
            auto buf_L = gemm_3_matrix(buf_L_, n - (i+1)*r, r);
            std::vector<gemm_type3> buf_U_(r * (n - (i+1)*r));
            auto buf_U = gemm_3_matrix(buf_U_, r, n - (i+1)*r);
            //diff_matmul<5>(A10, A01, A11, buf_L, buf_U, -1.0, 1.0, r);
            squeezing_matmul(A10, A01, A11, buf_L, buf_U, -1.0, 1.0, r);
            //block_gemm(A10, A01, A11, buf_L, buf_U);
            //scaled_matmul(A10, A01, A11, buf_L, buf_U, r);
            if(dropping_prob != 1.0) flop_counter.add_fp8_flops(r*(n - (i+1)*r)*(n - (i+1)*r));
        } else if(can_use_ext_type<gemm_type3>(A11, diag_left ,update_left, tol)) {
            swich = 1;
            cout << " using lowest extended precision \n"; //8 bit float fp8
            std::vector<gemm_type3> buf_L_(r * (n - (i+1)*r));
            auto buf_L = gemm_3_matrix(buf_L_, n - (i+1)*r, r);
            std::vector<gemm_type3> buf_U_(r * (n - (i+1)*r));
            auto buf_U = gemm_3_matrix(buf_U_, r, n - (i+1)*r);
            diff_matmul<5>(A10, A01, A11, buf_L, buf_U, -1.0, 1.0, r);
            if(dropping_prob != 1.0) flop_counter.add_fp8_flops(2*r*(n - (i+1)*r)*(n - (i+1)*r));
            
        } else if(can_use_type<gemm_type2>(A11, diag_left , update_left, tol)) {
            swich = 2;
            cout << " using middle precision \n"; //16 bit float
            std::vector<gemm_type2> buf_L_(r * (n - (i+1)*r));
            auto buf_L = gemm_2_matrix(buf_L_, n - (i+1)*r, r);
            std::vector<gemm_type2> buf_U_(r * (n - (i+1)*r));
            auto buf_U = gemm_2_matrix(buf_U_, r, n - (i+1)*r);
            block_gemm(A10, A01, A11, buf_L, buf_U, r);
            //squeezing_matmul(A10, A01, A11, buf_L, buf_U, -1.0, 1.0);
            flop_counter.add_half_flops(r*(n - (i+1)*r)*(n - (i+1)*r));
        } else {
            swich = 3;
            cout << " using highest precision \n"; //32 bit float 
            std::vector<gemm_type> buf_L_(r * (n - (i+1)*r));
            auto buf_L = gemm_matrix(buf_L_, n - (i+1)*r, r);
            std::vector<gemm_type> buf_U_(r * (n - (i+1)*r));
            auto buf_U = gemm_matrix(buf_U_, r, n - (i+1)*r);
            block_gemm(A10, A01, A11, buf_L, buf_U, r);
            flop_counter.add_float_flops(r*(n - (i+1)*r)*(n - (i+1)*r));

        }
         

        
        if(swich == 0 || swich == 1) cholesky_kernel<gemm_type3>(A00, eta, ksi, n, chol_modif, phase2);
        else if(swich == 2) cholesky_kernel<gemm_type2>(A00, eta, ksi, n, chol_modif, phase2);
        else cholesky_kernel<gemm_type>(A00, eta, ksi, n, chol_modif, phase2);
        flop_counter.add_float_flops(r*r*r/3);
        


        if(!is_symm(A00)) {
            throw std::runtime_error("A00 is not SPD!"); 
        }

        
    

        tlapack::trsm(tlapack::LEFT_SIDE, Uplo::Lower, tlapack::NO_TRANS, tlapack::Diag::NonUnit, 1.0, A00, A01);
        flop_counter.add_float_flops(r*r*r*(q - i - 1));

        // if(swich == 0) {
        //     for(int i = 0; i < nrows(A01); i++) {
        //         for(int j = 0; j < ncols(A01); j++) {
        //             A01(i,j) = static_cast<float>(static_cast<gemm_type2>(A01(i,j)));
        //         }
        //     }
       
        // }


        //need to provide alternate check where eps_8*A(i, i) < eps'*diag_A[i] for indices in the Schur complement



        double err_bnd = tol*diag_A[(i+1)*r]/(A((i+1)*r, (i+1)*r));
        flop_counter.add_double_flops(1);
        for(int s = 0; s < nrows(A10); s++) {
            for(int t = 0; t < ncols(A10); t++) {
                A10(s, t) = A01(t, s);
            }
        }

        auto diag_left = std::vector<T>(diag_A.begin() + i*r + r, diag_A.end());
        auto update_left = std::vector<T>(updated_diag.begin() + i*r + r, updated_diag.end());
        auto masks_left = std::vector<T>(masks.begin() + i*r + r, masks.end());
        if(i*r + r == n) return 0;

        //tau = 0.25, mu = 0.001
        phase2 = update_diag<float>(A10, update_left, A11, diag_left, masks_left, chol_modif, 0.00025, eta, 0.001);

        


   

        

  

        
    }

   

    return 0;


}
