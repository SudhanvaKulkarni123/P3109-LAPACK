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
#include <iostream>
#include <vector>
#include <random>
#include <fstream>

template<TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR piv_t>
int random_comp_LU(matrix_t& A, matrix_t& Psi, matrix_t& Omega, piv_t& left_piv, piv_t& right_piv, int r, int b = 32)
{

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);  // N(0,1) distribution

    int n = nrows(A);
    int alpha = 0;


    // Fill the matrix with N(0,1) values
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < r; ++j) {
            Omega(i,j) = dist(gen);  // Generate random value from N(0,1)
        }
    }

    tlapack::gemm(tlapack::NO_TRANS, tlapack::NO_TRANS, 1.0, Omega, A, Psi);

    for(int k = 0; k < n; k+= b)
    {
        int k_up = k + min(b, n - k + 1) - 1;
        for(int v = k; k < k_up; k++) {

            if(n - v > r) {
                float max_norm = 0;
                for(int j = v; j < n; j++ ) {
                    if (tlapack::nrm2(tlapack::col(Psi, j)) > max_norm){
                        max_norm = tlapack::nrm2(tlapack::col(Psi, j));
                        alpha = j;
                    } 
                }
 
            } else {

            }

        } 
    }

    return 0;

}