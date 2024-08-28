/// @author Sudhanva Kulkarni, University of California Berkeley, USA
/// This file contains a collection of scaling routines for the GMRES_IR algorithm.

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#ifndef TLAPACK_PREFERRED_MATRIX_LEGACY
#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/LegacyMatrix.hpp>
#include <tlapack/blas/nrm2.hpp>
#include <tlapack/blas/lange.hpp>
#endif


template<typename T>
T inline closest_power_of_two(T n) {
    return pow(2, floor(log2(n)));
}


template <typename matrixA_t, typename vectorS_t, typename vectorR_t>
void RnC_Equilibrate(matrixA_t &A, vectorS_t &s, vectorR_t &r, double tol = 1e-8) {
    int m = rows(A);
    int n = cols(A);
    using real_t = type_t<matrixA_t>;
    real_t max = real_t(0);
    for( int i = 0; i < m; i++) {
        for( int j = 0; j < n; j++) {
            if(abs(A(i,j)) > max) {
                max = abs(A(i,j));
            }
        }
        r[i] = 1/closest_power_of_two(max);
        max = real_t(0);
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            A(i,j) = A(i,j) * r[i];
        }
    }

    max = real_t(0);
    for( int j = 0; j < n; j++) {
        for( int i = 0; i < m; i++) {
            if(abs(A(i,j)) > max) {
                max = abs(A(i,j));
            }
        }
        s[j] = 1/closest_power_of_two(max);
        max = real_t(0);
    }

    for(int j = 0; j < n; j++) {
        for(int i = 0; i < m; i++) {
            A(i,j) = A(i,j) * s[j];
        }
    }
 
}


template <typename matrixA_t, typename vectorS_t, typename vectorR_t>
void Sym_Equilirate(matrixA_t &A, vectorS_t &s, vectorR_t &r, double tol = 1e-8) {
    int m = rows(A);
    int n = cols(A);
    using real_t = type_t<matrixA_t>;
    real_t maxR, maxS;
    std::vector<real_t> r_buf(m, 1.0);
    std::vector<real_t> s_buf(n, 1.0);
    
    

    int not_count = 0;
    while(true){
        for(int i = 0; i < n; i++){
            auto c1 = tlapack::rows(A,range(i,i+1));
            r[i] = 1/closest_power_of_two(sqrt(tlapack::lange(tlapack::Norm::Inf, c1)));
            auto c2 = tlapack::cols(A, range(i,i+1));
            s[i] = 1/closest_power_of_two(sqrt(tlapack::lange(tlapack::Norm::Inf, c2)));
            maxR = r[i] > maxR ? r[i] : maxR;
            maxS = s[i] > maxS ? s[i] : maxS;

        }
        for(int j = 0; j < m; j++){
            for(int k = 0; k < n; k++){
                A(j,k) = A(j,k)*(r[j])*(s[k]);
            }
        }
        for(int i = 0 ; i < n; i++){
            r_buf[i] = r_buf[i]*r[i];
            s_buf[i] = s_buf[i]*s[i];
        }
        //std::cout << maxR;
        not_count++;
        if(abs(maxR - 1) < 1 || abs(maxS - 1) < 1 || not_count > 10) break;
        }
    for(int i = 0; i < m; i++) r[i] = r_buf[i];
    for(int i = 0; i < n; i++) s[i] = s_buf[i];

    return;
 
}

template <typename matrixA_t, typename vectorS_t, typename vectorR_t>
void geom_Scaling(matrixA_t &A, vectorS_t &s, vectorR_t &r, double tol = 1e-8) {
    int m = rows(A);
    int n = cols(A);
    using real_t = type_t<matrixA_t>;
    real_t maxR, maxS;
    std::vector<real_t> r_buf(m, 1.0);
    std::vector<real_t> s_buf(n, 1.0);
    
    

    int not_count = 0;
    while(true){
        for(int i = 0; i < n; i++){
            auto c1 = tlapack::rows(A,range(i,i+1));
            r[i] = 1/closest_power_of_two(sqrt(tlapack::lange(tlapack::Norm::Inf, c1)));
            auto c2 = tlapack::cols(A, range(i,i+1));
            s[i] = 1/closest_power_of_two(sqrt(tlapack::lange(tlapack::Norm::Inf, c2)));
            maxR = r[i] > maxR ? r[i] : maxR;
            maxS = s[i] > maxS ? s[i] : maxS;

        }
        for(int j = 0; j < m; j++){
            for(int k = 0; k < n; k++){
                A(j,k) = A(j,k)*(r[j])*(s[k]);
            }
        }
        for(int i = 0 ; i < n; i++){
            r_buf[i] = r_buf[i]*r[i];
            s_buf[i] = s_buf[i]*s[i];
        }
        //std::cout << maxR;
        not_count++;
        if(abs(maxR - 1) < 1 || abs(maxS - 1) < 1 || not_count > 10) break;
        }
    for(int i = 0; i < m; i++) r[i] = r_buf[i];
    for(int i = 0; i < n; i++) s[i] = s_buf[i];

    return;
 
}
