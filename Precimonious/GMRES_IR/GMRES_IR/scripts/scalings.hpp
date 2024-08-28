//// @author Sudhanva Kulkarni UC Berkeley
/// this file contains some utility functions for scaling the input matrix

#include <tlapack/base/utils.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/blas/scal.hpp>





template<typename T>
T inline closest_power_of_two(T n) {
    return pow(2, floor(log2(n)));
}

template<TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vec_t>
void geom_mean_scal(matrix_t&A, vec_t& R, vec_t& S)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    assert(n == m);
    int k = 0; // number of iterations
    do {
        for(int i = 0 ; i < n; i++)
        {
           T maxim = T(0.0);
            T minim = T(99999999999.0);
            for(int j = 0; j < n; j++)
            {
                maxim = max(maxim, abs(A(i, j)));
                minim = min(minim, abs(A(i, j)));
            }
            R[i] = (sqrt(maxim * minim));
            auto row = tlapack::row(A, i);
            tlapack::rscl(R[i], row);
        }
        for(int i = 0 ; i < n; i++)
        {
            T maxim = T(0.0);
            T minim = T(99999999999.0);
            for(int j = 0; j < n; j++)
            {
                maxim = max(maxim, abs(A(j, i)));
                minim = min(minim, abs(A(j, i)));
            }
            S[i] = (sqrt(maxim * minim));
            auto col = tlapack::col(A, i);
            tlapack::rscl(S[i], col);
        }
        k++;
    } while(k < 2);

    return;


}

template<TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vec_t>
void arith_mean_scal(matrix_t&A, vec_t& R, vec_t& S)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    assert(n == m);
    int k = 0; // number of iterations
    do {
        for(int i = 0 ; i < n; i++)
        {
            T sum = T(0.0);
            for(int j = 0; j < n; j++)
            {
                sum += abs(A(i, j));
            }
            R[i] = (sum / n);
            auto row = tlapack::row(A, i);
            tlapack::rscl(R[i], row);
        }
        for(int i = 0 ; i < n; i++)
        {
            T sum = T(0.0);
            for(int j = 0; j < n; j++)
            {
                sum += abs(A(j, i));
            }
            S[i] = (sum / n);
            auto col = tlapack::col(A, i);
            tlapack::rscl(S[i], col);
        }
        k++;
    } while(k < 3);

    return;
}

template<TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vec_t, TLAPACK_SCALAR norm_type>
void norm_equilibration(matrix_t&A, vec_t& R, vec_t& S, norm_type normA)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    assert(n == m);
    int k = 0; // number of iterations
    do {
        for(int i = 0 ; i < n; i++)
        {
            T sum = T(0.0);
            for(int j = 0; j < n; j++)
            {
                sum += abs(A(i, j));
            }
            R[i] = (sum);
            auto row = tlapack::row(A, i);
            tlapack::rscl(R[i], row);
            tlapack::scal(normA, row);
        }
        for(int i = 0 ; i < n; i++)
        {
            T sum = T(0.0);
            for(int j = 0; j < n; j++)
            {
                sum += abs(A(j, i));
            }
            S[i] = (sum);
            auto col = tlapack::col(A, i);
            tlapack::rscl(S[i], col);
            tlapack::scal(normA, col);
        }
        k++;
    } while(k < 1);

    return;
}