/// This file contains code for a pre-pivoted cholesky routine that sorts the values on the diagonal first and then switches based on the diagonal value we are at  
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



//function for regular unblocked Cholesky decomp. returns decomp in lower part of matrix
template<TLAPACK_MATRIX matrix_t>
int cholesky_kernel(matrix_t& A)
{   
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    idx_t n = nrows(A);
    real_t tmp = real_t(0.0);
    for (int i = 0; i < n; i++) {
      
    tmp = A(i,i);
    if(i == 2) cout << tmp << endl;

    for (int k = 0; k < i; k++) {
        tmp -= A(i,k) * A(i,k); 
        if(i == 2) cout << tmp << endl; 
    }
    if (tmp <= 0) {
        throw std::runtime_error("Matrix is not positive definite");
    }

    A(i,i) = sqrt(tmp);  

    for (int j = i+1; j < n; j++) {
        tmp = A(j,i);
        for (int k = 0; k < i; k++) {
            tmp -= A(j,k) * A(i,k);  
        }
        A(j,i) = tmp / A(i,i);  
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

    cout << " A is : \n";
    printMatrix(A);
    cholesky_kernel(A);
    cout << " A after cholesky is  :  \n";
    printMatrix(A);

    vector<float> B_(9, 0.0);
    auto B = gemm_matrix(B_, 3, 3);
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < i; j++) {
            B(j,i) = A(i,j);
        }
        B(i,i) = A(i,i);
    }

    tlapack::trmm(tlapack::LEFT_SIDE, Uplo::Lower, tlapack::NO_TRANS, tlapack::Diag::NonUnit, 1.0, A, B);
    cout << "B is : \n"; 
    printMatrix(B);


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
int pivoted_cholesky(matrix_t& A, piv_t& left_piv, piv_t& right_piv, int r = 32, double tol = 0.0001)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<int, int>;

    using gemm_type = float;
    using gemm_type2 = Eigen::half;
    using gemm_type3 = ml_dtypes::float8_ieee_p<6>;



    Create<LegacyMatrix<gemm_type, idx_t>> gemm_matrix;
    Create<LegacyMatrix<gemm_type2, idx_t>> gemm_2_matrix;
    Create<LegacyMatrix<gemm_type3, idx_t>> gemm_3_matrix;

    idx_t n = nrows(A);
    int q = n/r;


    using val_idx = pair<T, int>;

    if ( n < r) return cholesky_kernel(A);

    priority_queue<val_idx, vector<val_idx>, Compare<T>> diag_A;
    


    for(int i = 0; i < n; i++) diag_A.push({A(i,i), i});

    for(int i = 0; i < n; i++) {
        left_piv[i] = diag_A.top().second;
        right_piv[i] = diag_A.top().second;
        diag_A.pop();
        auto tmp1 = tlapack::col(A, i);
        auto tmp2 = tlapack::col(A, right_piv[i]);
        tlapack::swap(tmp1, tmp2);
        tmp1 = tlapack::row(A,i);
        tmp2 = tlapack::row(A, left_piv[i]);
        tlapack::swap(tmp1, tmp2);
    }
    //now the matrix is ready to be Cholesky'd

    for(int i = 0; i < q; i++) 
    {
        auto A00 = tlapack::slice(A, range(i*r, (i+1)*r), range(i*r, (i+1)*r));
        cholesky_kernel(A00);
        auto A10 = tlapack::slice(A, range((i+1)*r, n), range((i)*r, (i+1)*r));
        tlapack::trsm(tlapack::RIGHT_SIDE, Uplo::Lower, tlapack::TRANSPOSE, tlapack::Diag::NonUnit, 1.0, A00, A10);
        //check the value of err and perform gemm update
        double err_bnd = tol/A(i*r,i*r);
        auto A01 = tlapack::slice(A, range(i*r, (i+1)*r), range((i+1)*r, n));
        for(int s = 0; s < nrows(A10); s++) {
            for(int t = 0; t < ncols(A10); t++) {
                A01(t, s) = A10(s, t);
            }
        }
        auto A11 = tlapack::slice(A, range(i*r, (i+1)*r), range(i*r, n));

        if (err_bnd > (2.0*pow(2.0, -7) + pow(2.0, -14))/(1.0 - 2.0*pow(2.0, -7) - pow(2.0, -14))) {
            std::vector<gemm_type3> buf_L_(r * (n - (i+1)*r));
            auto buf_L = gemm_3_matrix(buf_L_, n - (i+1)*r, r);
            std::vector<gemm_type3> buf_U_(r * (n - (i+1)*r));
            auto buf_U = gemm_3_matrix(buf_U_, r, n - (i+1)*r);
            block_gemm(A10, A01, A11, buf_L, buf_U);
        } else if(err_bnd > (2.0*pow(2.0, -11) + pow(2.0, -22))/(1.0 - 2.0*pow(2.0, -11) - pow(2.0, -22))) {
            std::vector<gemm_type2> buf_L_(r * (n - (i+1)*r));
            auto buf_L = gemm_2_matrix(buf_L_, n - (i+1)*r, r);
            std::vector<gemm_type2> buf_U_(r * (n - (i+1)*r));
            auto buf_U = gemm_2_matrix(buf_U_, r, n - (i+1)*r);
            block_gemm(A10, A01, A11, buf_L, buf_U);
        } else {
            std::vector<gemm_type> buf_L_(r * (n - (i+1)*r));
            auto buf_L = gemm_matrix(buf_L_, n - (i+1)*r, r);
            std::vector<gemm_type> buf_U_(r * (n - (i+1)*r));
            auto buf_U = gemm_matrix(buf_U_, r, n - (i+1)*r);
            block_gemm(A10, A01, A11, buf_L, buf_U);

        }

        
    }

    return 0;


}
