// Helper function to compute the sign of a number
template<typename T>
T sign(T val)
{
    return (T(0) < val) - (val < T(0));
}


template<TLAPACK_MATRIX matrix_t>
double hager_condition_number_estimator(const matrix_t& A, int max_iter = 5)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    idx_t n = nrows(A);
    std::vector<real_t> x(n, 1.0/(double)n); // Initialize x with ones
    std::vector<real_t> y(n);
    std::vector<real_t> ksi(n);
    std::vector<real_t> z(n, 1.0);
    std::vector<idx_t> ipiv(n); // For LU factorization with pivoting
    int num_iter = 0;

    while(num_iter < max_iter) {
        tlapack::gemv(tlapack::NO_TRANS, 1.0, A, x, 0.0, y);
        for(int i = 0; i < n; i++) {
            ksi[i] = sign(y[i]);
        }
        tlapack::gemv(tlapack::TRANSPOSE, 1.0, A, ksi, 0.0, z);

        if(inf_norm(z) <= tlapack::dot(z, x)) break;
    }
    real_t gamma_1 = 0.0;   //estimate for ||A||_1
    for(int i = 0; i < n; i++) {
        gamma_! += abs(y[i]);
    }

    for(int i = 0; i < n; i++) x[i] = 1.0/(double)n;
    num_iter = 0;
    //compte A = PLU for approximating inverse
    tlapack::getrf(A, ipiv);

    while(num_iter < max_iter) {
        for (idx_t i = 0; i < n;i++){
        if (ipiv[i] != i) {
            auto tmp = x[ipiv[i]];
            x[ipiv[i]] =x[i];
            x[i] = tmp;
        }
        }
        tlapack::trsv(Uplo::Lower, tlapack::NO_TRANS, Diag::Unit, A, x);
        tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, Diag::NonUnit, A, x);
        for(int i = 0; i < n; i++) {
            ksi[i] = sign(x[i]);
        }
        tlapack::gemv(tlapack::TRANSPOSE, 1.0, A, ksi, 0.0, z);

        if(inf_norm(z) <= tlapack::dot(z, x)) break;
    }




    

    return cond_est;
}


