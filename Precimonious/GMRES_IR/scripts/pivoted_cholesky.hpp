/// This file contains code for a pre-pivoted cholesky routine that sorts the values on the diagonal first and 
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
#include <fstream>
#include <queue>



template<TLAPACK_MATRIX matrix_t>
bool is_symm(matrix_t& A) {
    
    int n = nrows(A);
    for(int i = 0; i < n; i++) {
        for(int j= 0; j  <n; j++) {
            if(A(i,j) != A(j,i)) return false;
        }
    }

    return true;

}


template<TLAPACK_VECTOR vector_t>
int max_index(vector_t& v, int start) {
    int to_ret = start;
    int n = size(v);
    for(int i = start; i < n; i++) {
        if(abs(v[i]) > abs(v[to_ret])) to_ret = i;
    }
    return to_ret;
}
//function for regular unblocked Cholesky decomp. returns decomp in lower part of matrix
template<TLAPACK_SCALAR eps_t, TLAPACK_MATRIX matrix_t>
int cholesky_kernel(matrix_t& A, float eta = 1.0, float ksi = 1.0, int N = 1024, chol_mod chol_modif = chol_mod::NONE, bool phase2 = false, float prev_err = 0.0)
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
    if(chol_modif == chol_mod::GMW81)
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


void cholesky_kernel_test() 
{
    vector<float> A_(9, 0.0);
    Create<LegacyMatrix<float, int>> gemm_matrix;
    auto A = gemm_matrix(A_, 3, 3);
    
    A(0,0) = 4.0;
    A(1,0) = 12.0; A(0,1) = 12.0;
    A(2,0) = -16.0; A(0,2) = -16.0;
    A(1,1) = 37.0;
    A(2,2) = 98.0;
    A(2,1) = -43.0; A(1,2) = -43.0;

    cholesky_kernel<float>(A);
  

    vector<float> B_(9, 0.0);
    auto B = gemm_matrix(B_, 3, 3);
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < i; j++) {
            B(j,i) = A(i,j);
        }
        B(i,i) = A(i,i);
    }

    tlapack::trmm(tlapack::LEFT_SIDE, Uplo::Lower, tlapack::NO_TRANS, tlapack::Diag::NonUnit, 1.0, A, B);



}

std::ofstream logfile("mgs.txt");

/// function to updated diagonal entry and check for update precision and modified cholesky phase switching
template<TLAPACK_SCALAR scalar_t, TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
bool update_diag(matrix_t& L, vector_t& updated_A, matrix_t& A, vector_t& diag_A, vector_t& masks, chol_mod chol_modif, float tau = 1.0, float gamma = 1.0, float mu = 1.0)
{
    using idx_t = size_type<matrix_t>;
    int n = nrows(L);
    int b = ncols(L);

    bool to_ret = true;
   

    for(int i = 0; i < n; i++) updated_A[i] = A(i,i);

    double to_print_sum = 0.0;

    for(int i = 0; i < n; i++) {
        to_print_sum = 0.0;
        for(int j = 0; j < b; j++) {
            updated_A[i] -= L(i,j)*L(i,j);
            to_print_sum += L(i,j)*L(i,j);
            
        }
        if(chol_modif == chol_mod::SE99) {
            to_ret = to_ret & (updated_A[i] < tau*gamma);
        } else if(chol_modif == chol_mod::SE90) {
            to_ret = to_ret & (updated_A[i] < -mu*diag_A[i] || updated_A[i] < -mu*gamma);
        } else if(chol_modif == chol_mod::GMW81) {
            to_ret = to_ret & (updated_A[i] < tau*diag_A[i]);
        }



        logfile << "to_print_sum = " << to_print_sum << "\n";
        logfile <<  "sample element = " << L(0,0) << std::endl;
        logfile << "diag we are updating = " << updated_A[i] << std::endl;
        to_print_sum = 0.0;

    }


    for(int i = 0; i < n; i++) {
        if(updated_A[i] <= 0) {masks[i] = 0.0;}
        else masks[i] = 1.0;
    }


    return to_ret;

}

template<TLAPACK_SCALAR scalar_t, TLAPACK_VECTOR vector_t, TLAPACK_MATRIX matrix_t>
bool can_use_type(matrix_t& A, vector_t& diag_A, vector_t& updated_A, double eps_prime)
{
    int n = size(diag_A);

    
    double mach_eps = (double)std::numeric_limits<scalar_t>::epsilon();
    
    for(int i = 0; i < n; i++) {     
        if (!(mach_eps*updated_A[i] < eps_prime*diag_A[i])) return false;
    }
    
    return true;   
}

template<TLAPACK_SCALAR scalar_t, TLAPACK_VECTOR vector_t, TLAPACK_MATRIX matrix_t>
bool can_use_ext_type(matrix_t& A, vector_t& diag_A, vector_t& updated_A, double eps_prime)
{
    int n = size(diag_A);

    
    double mach_eps_sq = (double)std::numeric_limits<scalar_t>::epsilon();
    mach_eps_sq *= mach_eps_sq;
    
    for(int i = 0; i < n; i++) {     
        if (!(mach_eps_sq*updated_A[i] < eps_prime*diag_A[i])) return false;
    }
    
    return true;   
}





template<typename T>
class Compare {
public:
    bool operator()(pair<T, int> below, pair<T, int> above)
    {
        if (below.first < above.first) {
            return true;
        }
        return false;
    }
};



template<TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR piv_t>
int pivoted_cholesky(matrix_t& A, piv_t& left_piv, piv_t& right_piv, chol_mod& chol_modif, n_flops& flop_counter,  int r = 32, double tol = 0.0000000000001, float dropping_prob = 0.1)
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
         

        
        if(swich == 0 || swich == 1) cholesky_kernel<gemm_type3>(A00, eta, ksi, n, chol_modif, phase2);
        else if(swich == 2) cholesky_kernel<gemm_type2>(A00, eta, ksi, n, chol_modif, phase2);
        else cholesky_kernel<gemm_type>(A00, eta, ksi, n, chol_modif, phase2);
        flop_counter.add_float_flops(r*r*r/3);
        


        if(!is_symm(A00)) {
            throw std::runtime_error("A00 is not SPD!"); 
        }

        auto A01 = tlapack::slice(A ,  range((i)*r, (i+1)*r), range((i+1)*r, n));
    

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
        auto A10 = tlapack::slice(A,  range((i+1)*r, n) ,range(i*r, (i+1)*r));
        for(int s = 0; s < nrows(A10); s++) {
            for(int t = 0; t < ncols(A10); t++) {
                A10(s, t) = A01(t, s);
            }
        }

        auto A11 = tlapack::slice(A, range(i*r + r, n), range(i*r + r, n));
        auto diag_left = std::vector<T>(diag_A.begin() + i*r + r, diag_A.end());
        auto update_left = std::vector<T>(updated_diag.begin() + i*r + r, updated_diag.end());
        auto masks_left = std::vector<T>(masks.begin() + i*r + r, masks.end());
        if(i*r + r == n) return 0;

        //tau = 0.25, mu = 0.001
        phase2 = update_diag<float>(A10, update_left, A11, diag_left, masks_left, chol_modif, 0.00025, eta, 0.001);

        


   

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

  

        
    }

   

    return 0;


}
