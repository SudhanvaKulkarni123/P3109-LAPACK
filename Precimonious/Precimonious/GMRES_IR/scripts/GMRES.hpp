/// @author Sudhanva Kulkarni, University of California Berkeley, USA
/// This file contains some relevant versions of the GMRES-IR routine.

enum class kernel_type
{
    LEFT_LU,
    RIGHT_LU,
    RIGHT_GMRES
};


template <typename matrix_t>
bool isNanorInf(matrix_t& A)
{
    for(int i = 0; i < nrows(A); i++)
    {
        for(int j = 0; j < ncols(A); j++)
        {
            if(isnan(A(i,j)) || isinf(A(i,j)))
            {
                return true;
            }
        }
    }
    return false;
}

template <typename matrix_t, typename vector_t>
void compute_residual(matrix_t &A, vector_t &x, vector_t &b, vector_t &r)
{
    // data traits
    using idx_t = size_type<matrix_t>;
    using scalar_t = type_t<matrix_t>;
    using range = pair<idx_t, idx_t>;
    int n = size(r);
    for (int i = 0; i < n; i++)
        r[i] = b[i];

    // compute r = b - A*x
    tlapack::gemv(Op::NoTrans, static_cast<scalar_t>(-1.0), A, x, static_cast<scalar_t>(1.0), r);
}

template <typename vector_t>
double inf_norm(vector_t &v)
{
    double max = 0;
    for (int i = 0; i < size(v); i++)
    {
        if (abs(v[i]) > max)
        {
            max = abs(v[i]);
        }
    }
    return max;
}

/// solution is returned in b
template <typename matrix_t, typename vector_t, typename idk>
void LU_solve(matrix_t &LU, std::vector<idk> &piv, vector_t &b)
{
    // data traits
    using idx_t = size_type<matrix_t>;
    using scalar_t = type_t<matrix_t>;
    using range = pair<idx_t, idx_t>;

    int n = size(b);

    for (int i = 0; i < n; i++)
    {
        if (piv[i] != i)
        {
            auto tmp = b[piv[i]];
            b[piv[i]] = b[i];
            b[i] = tmp;
        }
    }

    tlapack::trsv(Uplo::Lower, tlapack::NO_TRANS, Diag::Unit, LU, b);
    tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, Diag::NonUnit, LU, b);
    return;
}

// this function will convert H into an upper triangular R and b into Q^Tb. Then we can solve Rx = Q^Tb outside this function
template <typename matrix_t, typename vector_t>
void Hessenberg_qr(matrix_t H, vector_t b, int size)
{
    // data traits
    using idx_t = size_type<matrix_t>;
    using real_t = type_t<matrix_t>;
    using range = pair<idx_t, idx_t>;

    auto m = nrows(H);
    auto n = ncols(H);
    real_t c = 0.0;
    real_t s = 0.0;
    real_t temp = 0.0;

    auto da_num = n < size ? n : size - 1;
    for (int i = 0; i < da_num; i++)
    {
        c = H(i, i);
        s = -H(i + 1, i);
        H(i, i) = sqrt(H(i, i) * H(i, i) + H(i + 1, i) * H(i + 1, i));
        c = c / H(i, i);
        s = s / H(i, i);
        H(i + 1, i) = 0.0;
        for (int j = i + 1; j < n; j++)
        {
            temp = c * H(i, j) - s * H(i + 1, j);
            H(i + 1, j) = s * H(i, j) + c * H(i + 1, j);
            H(i, j) = temp;
        }
        temp = c * b[i] - s * b[i + 1];
        b[i + 1] = s * b[i] + c * b[i + 1];
        b[i] = temp;
    }
}

/// this function perform one step of arnoldi iter
template <TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixH_t, TLAPACK_MATRIX matrixQ_t, TLAPACK_MATRIX matrixLU_t, typename idk>
void arnoldi_iter(matrixA_t &A, matrixH_t &H, matrixQ_t &Q, matrixLU_t LU, std::vector<idk> piv, kernel_type ker, int k)
{
    // data traits
    using idx_t = size_type<matrixA_t>;
    using scalar_t = type_t<matrixA_t>;
    using range = pair<idx_t, idx_t>;
    using prod_type = type_t<matrixLU_t>;

    // constants
    const idx_t n = nrows(Q);
    const idx_t m = ncols(H);

    std::vector<prod_type> Q_tmp_(n * n);
    tlapack::LegacyMatrix<prod_type, idx_t> Q_tmp(n, n, Q_tmp_.data(), n);
    tlapack::lacpy(tlapack::GENERAL, Q, Q_tmp);
    std::vector<prod_type> A_tmp_(n * n);
    tlapack::LegacyMatrix<prod_type, idx_t> A_tmp(n, n, A_tmp_.data(), n);
    tlapack::lacpy(tlapack::GENERAL, A, A_tmp);

    // temporary storage
    std::vector<scalar_t> w(n);
    std::vector<prod_type> w1(n);
    std::vector<prod_type> w2(n);

    // one step of Arnoldi iteration
    // w = A * V[j]
    if (ker == kernel_type::LEFT_LU)
    {
        auto vec_tmp = slice(Q_tmp, range{0, m}, k);
        gemv(Op::NoTrans, static_cast<prod_type>(1.0), A_tmp, vec_tmp, static_cast<prod_type>(0), w1);
        // need to permute before applying LU

        for (int i = 0; i < n; i++)
        {
            if (piv[i] != i)
            {
                auto tmp = w1[piv[i]];
                w1[piv[i]] = w1[i];
                w1[i] = tmp;
            }
        }
        tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU, w1);
        tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU, w1);

        for (int i = 0; i < n; i++)
            w[i] = static_cast<scalar_t>(w1[i]);
    }
    else if (ker == kernel_type::RIGHT_LU)
    {

        // invert then apply A
        auto vec_tmp = slice(Q_tmp, range{0, m}, k);
        for (int i = 0; i < n; i++)
            w1[i] = static_cast<prod_type>(vec_tmp[i]);
        for (int i = 0; i < n; i++)
        {
            if (piv[i] != i)
            {
                auto tmp = w1[piv[i]];
                w1[piv[i]] = w1[i];
                w1[i] = tmp;
            }
        }
        tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU, w1);
        tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU, w1);

        tlapack::gemv(Op::NoTrans, static_cast<prod_type>(1.0), A_tmp, w1, static_cast<prod_type>(0), w2);
        for (int i = 0; i < n; i++)
            w[i] = static_cast<scalar_t>(w2[i]);
    }

    // H[j,0:j+1] = V[0:n] * w
    for (idx_t i = 0; i < k + 1; ++i)
        H(i, k) = dot(slice(Q, range{0, m}, i), w);

    // w = w - V[0:n] * H[0:j+1,j]
    for (idx_t i = 0; i < k + 1; ++i)
        axpy(-H(i, k), slice(Q, range{0, m}, i), w);

    if (k == n - 1)
        return;
    // H[k+1,k] = ||w||
    H(k + 1, k) = nrm2(w);

    if (H(k + 1, k) == scalar_t(0))
        std::cout << "breakdown occured" << std::endl;

    // Q[k+1] = w / H[k+1,k]

    rscl(H(k + 1, k), w);
    for (int i = 0; i < m; i++)
    {
        Q(i, k + 1) = w[i];
    }

    return;
}

/// this solves Ax = b by GMRES with initial guess x0. Workspace arrays Q,H and r must be provided by the user. Solution is returned in b
template <typename matrix_t, typename vector_t, typename idk>
void GMRES(matrix_t& A, matrix_t& Q, matrix_t& H, matrix_t& LU, std::vector<idk>& piv, vector_t& b, vector_t& x0, vector_t& x, vector_t& r, kernel_type ker, int m, int max_num_iter, double tol = 1e-8)
{

    using idx_t = size_type<matrix_t>;
    using scalar_t = type_t<matrix_t>;
    using range = pair<idx_t, idx_t>;

    
    auto n = size(r);
    int i = 0;
    do
    {
        compute_residual(A, x0, b, r);
        std::cout << "inf norm fo r : " << inf_norm(r) << std::endl;
        if (inf_norm(r) / (inf_norm(b) * inf_norm(x0)) < tol)
            break;
        if (ker == kernel_type::LEFT_LU)
        {
            LU_solve(LU, piv, r);
        }
        auto norm_r = tlapack::nrm2(r);
        for (i = 0; i < n; i++)
        {
            Q(i, 0) = r[i] / norm_r;
        }
        for (i = 0; i < m; i++)
        {
            arnoldi_iter(A, H, Q, LU, piv, ker, i);
        }
        for(int j = 0; j < n; j++) r[j] = (j == 0 ? norm_r : 0.0);
        auto da_tmp = tlapack::slice(r,range{0, i});
        Hessenberg_qr(tlapack::slice(H,range{0, i+1}, range{0,i}), tlapack::slice(r,range{0, i+1}), n); 
        tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS,tlapack::Diag::NonUnit ,tlapack::slice(H,range{0, i}, range{0,i}), da_tmp); 
        tlapack::gemv(tlapack::NO_TRANS, static_cast<scalar_t>(1.0), tlapack::slice(Q, range{0, n}, range{0,i+1}), da_tmp, static_cast<scalar_t>(0.0), x);
        if(ker == kernel_type::RIGHT_LU)
        {
            LU_solve(LU, piv, x);
        }
        tlapack::axpy(static_cast<scalar_t>(1.0), x, x0);

    } while (max_num_iter-- > 0);

    
}


template <TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixH_t, TLAPACK_MATRIX matrixQ_t, TLAPACK_MATRIX matrixZ_t, TLAPACK_MATRIX matrixLU_t, typename idk>
void flexible_arnoldi_iter(matrixA_t &A, matrixH_t &H, matrixQ_t &Q, matrixZ_t& Z, matrixLU_t& LU, std::vector<idk>& piv, kernel_type ker, int k)
{
    // data traits
    using idx_t = size_type<matrixA_t>;
    using scalar_t = type_t<matrixA_t>;
    using range = pair<idx_t, idx_t>;
    using prod_type = type_t<matrixLU_t>;

    if(ker == kernel_type::RIGHT_LU || ker == kernel_type::LEFT_LU)
    {
        arnoldi_iter(A, H, Q, LU, piv, ker, k);
    }
    else if(ker == kernel_type::RIGHT_GMRES)
    {
        // constants
    const idx_t n = nrows(Q);
    const idx_t m = ncols(H);

    std::vector<prod_type> Q_tmp_(n * n);
    tlapack::LegacyMatrix<prod_type, idx_t> Q_tmp(n, n, Q_tmp_.data(), n);
    tlapack::lacpy(tlapack::GENERAL, Q, Q_tmp);
    std::vector<prod_type> A_tmp_(n * n);
    tlapack::LegacyMatrix<prod_type, idx_t> A_tmp(n, n, A_tmp_.data(), n);
    tlapack::lacpy(tlapack::GENERAL, A, A_tmp);

    std::vector<prod_type> Q_work_(n*n);
    tlapack::LegacyMatrix<prod_type, idx_t> Q_work(n, n, Q_work_.data(), n);

    std::vector<prod_type> H_work_(n*n);
    tlapack::LegacyMatrix<prod_type, idx_t> H_work(n, n, Q_work_.data(), n);

    // temporary storage
    std::vector<scalar_t> w(n);
    std::vector<prod_type> w1(n);
    std::vector<prod_type> w2(n);
    std::vector<prod_type> x(n);
    std::vector<prod_type> r(n);

    auto vec_tmp = slice(Q_tmp, range{0, m}, k);
    for (int i = 0; i < n; i++){
        w1[i] = static_cast<prod_type>(vec_tmp[i]);
        w2[i] = static_cast<prod_type>(vec_tmp[i]);
    }
    std::cout << "norm of  w2 before LU ; " << tlapack::nrm2(w2) << std::endl;
    std::cout << "LU is nanorinf : " << isNanorInf(LU) << std::endl;
    //want to call GMRES here instead of LU -- but can call LU for initial guess
    std::cout << "norm of  w2 before GMRES ; " << tlapack::nrm2(w2) << std::endl;
    GMRES(A_tmp, Q_work, H_work, LU, piv, w1, w2, x, r , kernel_type::LEFT_LU, m, 15, 1e-8);
    for(int i = 0; i < n; i++)
    {
        Z(i,k) = w2[i];
    }
    

    tlapack::gemv(Op::NoTrans, static_cast<prod_type>(1.0), A_tmp, w2, static_cast<prod_type>(0), w1);
    for (int i = 0; i < n; i++)
        w[i] = static_cast<scalar_t>(w1[i]);

    for (idx_t i = 0; i < k + 1; ++i)
        H(i, k) = dot(slice(Q, range{0, m}, i), w);

    // w = w - V[0:n] * H[0:j+1,j]
    for (idx_t i = 0; i < k + 1; ++i)
        axpy(-H(i, k), slice(Q, range{0, m}, i), w);

    if (k == n - 1)
        return;
    // H[k+1,k] = ||w||
    H(k + 1, k) = nrm2(w);

    if (H(k + 1, k) == scalar_t(0))
        std::cout << "breakdown occured" << std::endl;

    // Q[k+1] = w / H[k+1,k]

    rscl(H(k + 1, k), w);
    for (int i = 0; i < m; i++)
    {
        Q(i, k + 1) = w[i];
    }
    }

    
}

template <typename matrix_t, typename vector_t, typename idk>
void flexible_GMRES(matrix_t &A, matrix_t &Q, matrix_t &H, matrix_t& Z, matrix_t &LU, std::vector<idk>& piv, vector_t &b, vector_t &x0, vector_t& x, vector_t &r, kernel_type ker, int m, int max_num_iter, double tol = 1e-8)
{

    using idx_t = size_type<matrix_t>;
    using scalar_t = type_t<matrix_t>;
    using range = pair<idx_t, idx_t>;
    auto n = size(r);
    int i = 0;
    do
    {
        compute_residual(A, x0, b, r);
        if (inf_norm(r) / (inf_norm(b) * inf_norm(x0)) < tol)
            break;
        if (ker == kernel_type::LEFT_LU)
        {
            LU_solve(LU, piv, r);
        }
        auto norm_r = tlapack::nrm2(r);
        for (i = 0; i < n; i++)
        {
            Q(i, 0) = r[i] / norm_r;
        }
        for (i = 1; i < m; i++)
        {
            flexible_arnoldi_iter(A, H, Q, Z, LU, piv, ker, i-1);
        }
        for(int j = 0; j < n; j++) r[j] = (j == 0 ? norm_r : 0.0);
        auto da_tmp = tlapack::slice(r,range{0, i});
        Hessenberg_qr(tlapack::slice(H,range{0, i+1}, range{0,i}), tlapack::slice(r,range{0, i+1}), n); 
        tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS,tlapack::Diag::NonUnit ,tlapack::slice(H,range{0, i}, range{0,i}), da_tmp); 
        if(ker == kernel_type::RIGHT_GMRES)
        {
        tlapack::gemv(tlapack::NO_TRANS, static_cast<scalar_t>(1.0), tlapack::slice(Z, range{0, n}, range{0, i+1}), r, static_cast<scalar_t>(0.0), x);
        } else {
        tlapack::gemv(tlapack::NO_TRANS, static_cast<scalar_t>(1.0), tlapack::slice(Q, range{0, n}, range{0, i+1}), r, static_cast<scalar_t>(0.0), x);
        }
        if(ker == kernel_type::RIGHT_LU)
        {
            LU_solve(LU, piv, x);
        }
        tlapack::axpy(static_cast<scalar_t>(1.0), x, x0);

    } while (max_num_iter-- > 0);

    
}