#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/float8_iee_p.hpp>
#include <tlapack/plugins/eigen_bfloat.hpp>
#include <tlapack/plugins/eigen_half.hpp>
#include "opcounts.hpp"
#include <tlapack/blas/axpy.hpp>
#include <tlapack/blas/dot.hpp>
#include <tlapack/blas/nrm2.hpp>
#include <tlapack/blas/scal.hpp>
#include <tlapack/blas/trsv.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/gemv.hpp>
#include <tlapack/blas/ger.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/blas/trsm.hpp>
#include "getrf_blocked.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <string>


template<typename T>
void printMatrix(T& A, int n)
{
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            std::cout << A(i,j) << " ";
        }
        std::cout << std::endl;
    }
}


using namespace Eigen;
using namespace ml_dtypes;
int main(int argc, char** argv)
{
    int n = 50;
    std::vector<float> A_(n*n);
    std::vector<int> piv(n);
    tlapack::LegacyMatrix<float, int> A(n, n, A_.data(), n);
    std::vector<Eigen::half> B_(n*n);
    tlapack::LegacyMatrix<Eigen::half, int> B(n, n, B_.data(), n);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < 2; j++) {
            A(i,j) = Eigen::half(1.0);
        }
    }

    for(int i = 0; i < n; i++) {
        for(int j = 2; j < n; j++) {
            A(i,j) = float(rand())/float(RAND_MAX);
        }
        A(i,1) += float((double(rand())*1.0/(2048.0*double(RAND_MAX))));
        
    }

    

    printMatrix(A, n);

    tlapack::lacpy(tlapack::GENERAL, A, B);

    printMatrix(B, n);
    





    int info = tlapack::getrf(B , piv);

    if(info != 0) {
        std::cout << "Error in getrf in fp16" << std::endl;
    }   

    if(tlapack::getrf(A, piv) ==  0) std::cout << "factorization in fp32 worked" << std::endl;

    return 0;


}


