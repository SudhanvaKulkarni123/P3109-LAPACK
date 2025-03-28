
// cleaned_cg_ir.cpp
// Cleaned version of CG_IR with unused buffers removed but A_exp and A_F retained

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/blas/gemv.hpp>
#include <tlapack/blas/dot.hpp>
#include <tlapack/blas/trsv.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include "pivoted_cholesky.hpp"
#include "num_ops.hpp"
#include "n_flops.hpp"

template <typename F0, typename F1, typename T2, typename T3, typename T4, typename T5, typename T6>
double CG_IR(int n, double scale, float cond, factorization_type fact_type, double switching_val, double tolerance, double conv_thresh, bool is_symmetric, bool diag_dom, float dropping_prob, chol_mod chol_modif, int prec, int variant = 0, int max_CG_iter = 20, int inner_num_gmres_iter = 20, int num_IR_iter = 20, refinement_type method = refinement_type::CG_IR, int block_size = 128, int stopping_pos = 99999)
{
    using idx_t = size_t;
    using range = std::pair<idx_t, idx_t>;

    n_flops mixed_prec_flops;
    n_flops reg_flops;
    n_flops mixed_fact_flops;
    n_flops cholesky_flops;

    long nn = static_cast<long>(n);
    cholesky_flops.add_double_flops(nn * nn * (nn / 3));

    std::vector<float> A_(n * n);
    std::vector<int> A_exp(n, 0);
    tlapack::LegacyMatrix<float, idx_t> A(n, n, A_.data(), n);

    std::vector<double> FG_(n * n);
    tlapack::LegacyMatrix<double, idx_t> FG(n, n, FG_.data(), n);

    std::vector<T6> A_F_(n * n);
    tlapack::LegacyMatrix<T6, idx_t> A_F(n, n, A_F_.data(), n);

    std::vector<float> LL_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LL(n, n, LL_.data(), n);

    std::vector<T6> LL_copy_(n * n);
    tlapack::LegacyMatrix<T6, idx_t> LL_copy(n, n, LL_copy_.data(), n);

    std::vector<double> LL_double_(n * n);
    tlapack::LegacyMatrix<double, idx_t> LL_double(n, n, LL_double_.data(), n);

    std::vector<double> x_(n);
    tlapack::LegacyVector<double, idx_t> x(n, x_.data());

    std::vector<double> b_(n);
    tlapack::LegacyVector<double, idx_t> b(n, b_.data());

    std::vector<double> r_(n);
    tlapack::LegacyVector<double, idx_t> r(n, r_.data());

    constructMatrix<float>(n, cond, std::min(n - 1, static_cast<int>(std::ceil(cond / 5.0f))), false, FG, prec, is_symmetric, diag_dom, cond);
    double normA = tlapack::lange(tlapack::Norm::Inf, FG);
    reg_flops.add_double_flops(n * n);

    for (int i = 0; i < n * n; ++i)
        FG_(i) = scale * FG_(i);

    tlapack::lacpy(tlapack::Uplo::General, FG, A);

    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            A_F(i, j) = static_cast<T6>(FG(i, j));

    for (int i = 0; i < n; i++) {
        b[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    std::vector<int> left_piv(n), right_piv(n);
    for (int i = 0; i < n; i++) {
        left_piv[i] = i;
        right_piv[i] = i;
    }

    std::vector<double> D(n, 1.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A(i, j) *= D[i] * D[j];

    tlapack::lacpy(tlapack::GENERAL, A, LL);
    int info = pivoted_cholesky(LL, left_piv, right_piv, chol_modif, mixed_fact_flops, block_size, tolerance, dropping_prob);
    if (info != 0) {
        std::cerr << "Cholesky in low precision failed" << std::endl;
        return -1.0;
    }

    tlapack::lacpy(tlapack::GENERAL, LL, LL_copy);

    for (int i = 0; i < n; i++) {
        r[i] = b[i];
        x[i] = b[i];
    }

    tlapack::gemv(tlapack::NO_TRANS, -1.0, FG, x, 1.0, r);
    mixed_prec_flops.add_double_flops(2 * n * n);

    double norm_x = inf_norm(x);
    double initial_resid = inf_norm(r);
    double initial_backward_error = initial_resid / (norm_x * normA);
    std::cout << "Initial backward error: " << initial_backward_error << std::endl;

    return initial_backward_error;
}
