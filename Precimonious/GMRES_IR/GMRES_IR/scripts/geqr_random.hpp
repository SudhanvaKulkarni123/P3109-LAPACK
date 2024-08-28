#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/gemv.hpp>
#include <random>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/blas/nrm2.hpp>
#include <tlapack/blas/scal.hpp>
#include <tlapack/blas/ger.hpp>
#include <tlapack/blas/dot.hpp>


/// computes Psi*X where X is inout and Psi = [I, 0; 0, Omega]
template<TLAPACK_MATRIX matrixX_t, TLAPACK_MATRIX sketch_t, TLAPACK_MATRIX A_B_t>
void Sketch(matrixX_t& X, sketch_t& Omega, A_B_t& A_B)
{
    int n = nrows(X);
    int k = ncols(X);
    int m = n - ncols(Omega);
    using range =  std::pair<int, int>;
    auto A = tlapack::slice(X, range(0,m), range(0, k));
    auto B = tlapack::slice(X, range(m, n), range(0, k));
    auto new_B = tlapack::slice(A_B, range(m, n), range(0, k));
    auto new_A = tlapack::slice(A_B, range(0, m), range(0, k));

    tlapack::gemm(tlapack::NO_TRANS, tlapack::NO_TRANS, 1.0, Omega, B, 0.0, new_B);
    tlapack::lacpy(tlapack::Uplo::General, A, new_A);

    return;

}



//this algo gets a random householder vector w
template<TLAPACK_VECTOR vector_w_t, TLAPACK_VECTOR vector_y_t, TLAPACK_VECTOR vector_s_t, TLAPACK_VECTOR vector_u_t, TLAPACK_SCALAR scal_rho_t, TLAPACK_SCALAR scal_beta_t>
void HHvector(vector_w_t& w, vector_y_t& y, int& sigma, vector_s_t& s, vector_u_t& u, scal_rho_t& rho, scal_beta_t& beta, int j)
{
    int n = size(u);
    int lpm = size(y);
    sigma = (y[j] >= 0) ? 1 : -1;
    for(int i = 0; i < n; i++)
    {
        u[i] = (i < j-1) ? 0 : w[i];
    }
    for(int i = 0; i < lpm; i++)
    {
        s[i] = (i < j-1) ? 0 : y[i];
    }
    rho = tlapack::nrm2(s);

    u[j] += sigma*rho;
    s[j] += sigma*rho;  //this is gamma
    beta = static_cast<scal_beta_t>(1.0)/static_cast<scal_beta_t>(s[j]*rho);

    tlapack::scal(sqrt(beta), s);
    tlapack::scal(sqrt(beta), u);
    beta = 1;

    return;

}

//main algo for randQR on a tall amtrix
template<TLAPACK_MATRIX matrix_W_t, TLAPACK_MATRIX matrix_O_t, TLAPACK_MATRIX matrix_R_t, TLAPACK_MATRIX matrix_U_t, TLAPACK_MATRIX matrix_S_t, TLAPACK_MATRIX matrix_work_t, TLAPACK_VECTOR work_vector>
void randQR(matrix_W_t& W, matrix_O_t& Omega, matrix_R_t& R, matrix_U_t& U, matrix_S_t& S, matrix_work_t& work_matrix)
{   
    using range = std::pair<int, int>;
    int n = ncols(W);
    int l = nrows(R);
    for(int i = 0; i < n; i++)
    {
        auto w = tlapack::col(W, i);
        Sketch(W, Omega, work_matrix);
        auto y = tlapack::col(work_matrix, 0);
        auto Y = tlapack::cols(work_matrix, range(1, n));
        double rho = 0.0;
        double beta = 0.0;
        int sigma = 0;
        auto u = tlapack::col(U, i);
        auto s = tlapack::col(S, i);
        HHvector(w, y, sigma, s, u, rho, beta, i);
        auto r = tlapack::col(R, i);
        for(int j = 0; j < l; j++)
        {
            if(j < i - 1) r[j] = w[j];
            else if(j == i - 1) r[j] = -double(sigma)*rho;
            else r[j] = 0;
        }
        if(i < n - 1)
        {
            auto Wjm = tlapack::cols(W, range(i, n));
            double us = dot(u, s);
            for(int ii = 0; ii < n; ii++)
            {
                for(int jj = 0; jj < n - i; jj++)
                {
                    Wjm(ii, jj) -= beta*us*Y(ii,jj);
                }
            }

        }
        return;



    }

    return;

}

///this function uses randQR for arnoldi iter in GMRES. HEre w = b - Ax
template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixH_t, TLAPACK_MATRIX matrixQ_t, TLAPACK_MATRIX matrixO_t, TLAPACK_MATRIX matrixU_t, TLAPACK_MATRIX matrixS_t, TLAPACK_VECTOR vectorw_t, TLAPACK_VECTOR vectorz_t>
void rand_arnoldi(matrixA_t& A, matrixH_t& H, matrixQ_t& Q, matrixO_t& Omega, vectorw_t& w, vectorz_t& z, matrixU_t& U, matrixS_t& S)
{
    int m = nrows(Q);
    int n = ncols()
    Sketch(w, Omega, z);
    for(int i = 0; i <= m; i++)
    {
        int sigma;
        double rho, beta;
        auto s = tlapack::col(S, i);
        auto u = tlapack::col(U, i);
        HHvector(w, z, sigma, s, u, rho, beta, i);
        auto h = tlapack::col(H, i);

    }


}






